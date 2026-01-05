"""
Benchmark evaluation script for AutoCompressor models on MRQA-style datasets.

This script compresses the context into summary vectors (soft prompts) using an
AutoCompressor model (Llama/Qwen/OPT) and evaluates EM/F1 on MRQA datasets.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from auto_compressor import (
    LlamaAutoCompressorModel,
    OPTAutoCompressorModel,
    QwenAutoCompressorModel,
)

# Dataset name to file mapping (relative to --datasets-dir)
DATASET_FILES: Dict[str, str] = {
    "BioASQ": "BioASQ.jsonl.gz",
    "DROP": "DROP.jsonl.gz",
    "DuoRC": "DuoRC.jsonl.gz",
    "HotpotQA": "HotpotQA.jsonl.gz",
    "NaturalQuestions": "NaturalQuestions.jsonl.gz",
    "NewsQA": "NewsQA.jsonl.gz",
    "RACE": "RACE.jsonl.gz",
    "RelationExtraction": "RelationExtraction.jsonl.gz",
    "SearchQA": "SearchQA.jsonl.gz",
    "SQuAD": "SQuAD.jsonl.gz",
    "TextbookQA": "TextbookQA.jsonl.gz",
    "TriviaQA": "TriviaQA.jsonl.gz",
}


def format_prompt(question: str, context: str, template: str) -> str:
    return template.format(question=question, context=context)


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    import re
    import string

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    def lower(t: str) -> str:
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    prediction_counts: Dict[str, int] = {}
    ground_counts: Dict[str, int] = {}
    for tok in prediction_tokens:
        prediction_counts[tok] = prediction_counts.get(tok, 0) + 1
    for tok in ground_truth_tokens:
        ground_counts[tok] = ground_counts.get(tok, 0) + 1
    num_same = sum(min(prediction_counts.get(tok, 0), ground_counts.get(tok, 0)) for tok in prediction_counts)
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        scores_for_ground_truths.append(metric_fn(prediction, ground_truth))
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0


def compute_official_metrics(prediction: str, answers: Sequence[str]) -> Dict[str, float]:
    em = metric_max_over_ground_truths(exact_match_score, prediction, answers) * 100.0
    f1 = metric_max_over_ground_truths(f1_score, prediction, answers) * 100.0
    return {"exact_match": em, "f1": f1}


class MRQADataset(Dataset):
    """Minimal MRQA dataset wrapper that flattens each QA pair into a sample."""

    def __init__(
        self,
        dataset_path: Path,
        tokenizer,
        max_context_length: int,
        max_samples: int | None = None,
    ) -> None:
        self.samples: List[Dict[str, object]] = []
        total_samples = 0

        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                if "header" in example:
                    continue

                context = example.get("context", "")
                if not context:
                    continue

                context_tokens = tokenizer(
                    context,
                    truncation=True,
                    padding="max_length",
                    max_length=max_context_length,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.squeeze(0)

                for qa in example.get("qas", []):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    qid = qa.get("qid", f"sample-{len(self.samples)}")

                    if not question or not answers:
                        continue

                    attention_mask = (context_tokens != tokenizer.pad_token_id).long()

                    self.samples.append(
                        {
                            "id": qid,
                            "context_tokens": context_tokens,
                            "attention_mask": attention_mask,
                            "context": context,
                            "question": question,
                            "answers": answers,
                        }
                    )
                    total_samples += 1
                    if max_samples is not None and total_samples >= max_samples:
                        return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.samples[index]


def build_collate_fn():
    def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        return {
            "context_tokens": torch.stack([item["context_tokens"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "context_texts": [item["context"] for item in batch],
            "questions": [item["question"] for item in batch],
            "answers": [item["answers"] for item in batch],
            "ids": [item["id"] for item in batch],
        }

    return _collate


@dataclass
class ModelAdapter:
    tokenizer: object
    compress: callable
    predict: callable
    info: Dict[str, object]


def build_model(args: argparse.Namespace, device: torch.device) -> ModelAdapter:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_name_lower = args.model_path.lower()
    if "llama" in model_name_lower:
        ac_cls = LlamaAutoCompressorModel
    elif "qwen" in model_name_lower:
        ac_cls = QwenAutoCompressorModel
    else:
        ac_cls = OPTAutoCompressorModel

    config = AutoConfig.from_pretrained(args.model_path)
    if args.summary_length is not None:
        config.summary_length = args.summary_length
    if not hasattr(config, "summary_length"):
        raise ValueError(
            "Base model config is missing summary_length. "
            "Set it with --summary-length or use a checkpoint that includes it."
        )

    try:
        base_model = ac_cls.from_pretrained(args.model_path, config=config, dtype=dtype)
    except TypeError:
        # Fallback for older transformers that expect torch_dtype
        base_model = ac_cls.from_pretrained(args.model_path, config=config, torch_dtype=dtype)

    if args.lora_path:
        base_model = PeftModel.from_pretrained(base_model, args.lora_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = base_model.to(device)
    base_model.eval()

    def _compress(context_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = base_model(
                input_ids=context_tokens,
                attention_mask=(context_tokens != tokenizer.pad_token_id).long(),
                output_softprompt=True,
            )
        return out.softprompt

    def _predict(softprompt: torch.Tensor, prompts: List[str]) -> List[str]:
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_context_length,
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            outputs = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                softprompt=softprompt,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        outputs = outputs.cpu()
        prompt_lengths = attention_mask.sum(dim=1).cpu().tolist()
        responses: List[str] = []
        for idx, prompt_len in enumerate(prompt_lengths):
            generated_ids = outputs[idx, prompt_len:]
            responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        return responses

    return ModelAdapter(
        tokenizer=tokenizer,
        compress=_compress,
        predict=_predict,
        info={"summary_length": getattr(base_model.config, "summary_length", None)},
    )


def evaluate_single_dataset(
    dataset_name: str,
    dataset_path: Path,
    model: ModelAdapter,
    device: torch.device,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, float]:
    print(f"\nLoading dataset {dataset_name} from {dataset_path}")
    dataset = MRQADataset(dataset_path, model.tokenizer, args.max_context_length, args.max_samples)
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_name} is empty or unreadable.")
    print(f"Loaded {len(dataset)} QA pairs.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(),
    )

    predictions: Dict[str, str] = {}
    detailed_predictions: List[Dict[str, object]] = []
    em_total = 0.0
    f1_total = 0.0
    sample_count = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_start = time.perf_counter()

    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        context_tokens = batch["context_tokens"].to(device)
        softprompt = model.compress(context_tokens)

        prompts = [
            format_prompt(
                question=q,
                context=context,
                template=args.prompt_template,
            )
            for q, context in zip(batch["questions"], batch["context_texts"])
        ]

        predictions_text = model.predict(softprompt, prompts)

        for idx in range(len(batch["ids"])):
            qid = batch["ids"][idx]
            pred_text = predictions_text[idx].strip()
            gold_answers = batch["answers"][idx]
            metrics = compute_official_metrics(pred_text, gold_answers)

            predictions[qid] = pred_text
            detailed_predictions.append(
                {
                    "id": qid,
                    "question": batch["questions"][idx],
                    "context": batch["context_texts"][idx],
                    "answers": gold_answers,
                    "prediction": pred_text,
                    "prompt": prompts[idx],
                }
            )
            em_total += metrics["exact_match"]
            f1_total += metrics["f1"]
            sample_count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - eval_start

    em_avg = em_total / sample_count
    f1_avg = f1_total / sample_count

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    preds_path = dataset_dir / "predictions.json"
    metrics_path = dataset_dir / "metrics.json"
    full_preds_path = dataset_dir / "predictions_full.jsonl"

    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "samples": sample_count,
                "exact_match": em_avg,
                "f1": f1_avg,
                "total_time_s": total_time,
                "samples_per_second": sample_count / total_time if total_time > 0 else 0.0,
            },
            f,
            indent=2,
        )

    with open(full_preds_path, "w", encoding="utf-8") as f:
        for row in detailed_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"{dataset_name} — Samples: {sample_count}, "
        f"Exact Match: {em_avg:.2f}, F1: {f1_avg:.2f}, Time: {total_time:.1f}s"
    )

    return {
        "dataset": dataset_name,
        "samples": sample_count,
        "exact_match": em_avg,
        "f1": f1_avg,
        "total_time_s": total_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AutoCompressor QA on MRQA datasets.")
    parser.add_argument(
        "--datasets-dir",
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Directory containing MRQA jsonl.gz files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_FILES.keys()),
        default=sorted(DATASET_FILES.keys()),
        help="Datasets to evaluate.",
    )
    parser.add_argument("--model-path", required=True, help="Base AutoCompressor model name or path.")
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Optional path to a finetuned LoRA adapter checkpoint.",
    )
    parser.add_argument("--output-dir", default="./predictions/autocompressor", help="Directory to store benchmark outputs.")
    parser.add_argument("--max-context-length", type=int, default=512, help="Maximum number of context tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens to generate per answer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--device", default=None, help="Computation device (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on samples per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--prompt-template", default="Question: {question} Answer: ", help="Prompt template applied before generation.")
    parser.add_argument("--summary-length", type=int, default=None, help="Manually set summary_length if missing in the base config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model = build_model(args, device)

    summary_tag = model.info.get("summary_length", "softprompt")
    output_root = Path(args.output_dir) / f"autocompressor_{summary_tag}"
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []
    for dataset_name in args.datasets:
        dataset_file = DATASET_FILES[dataset_name]
        dataset_path = Path(args.datasets_dir) / dataset_file
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        results.append(
            evaluate_single_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                model=model,
                device=device,
                args=args,
                output_root=output_root,
            )
        )

    summary_path = output_root / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "summary_length": model.info.get("summary_length"),
                "datasets": results,
            },
            f,
            indent=2,
        )

    print("\nBenchmark evaluation finished. Summary:")
    for res in results:
        print(
            f"{res['dataset']}: Samples={res['samples']}, "
            f"Exact Match={res['exact_match']:.2f}, F1={res['f1']:.2f}, Time={res['total_time_s']:.1f}s"
        )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

# python evaluate_autocompressor_benchmark.py --summary-length 16 --lora-path /home/work/prompt/dpc/autocompressor/checkpoints/ac_Llama-3.1-8B-Instruct_sub2_seg2_sum16_lr8e-4_bsz64_rand_accu/checkpoint-2500 --model-path /home/work/prompt/models/Llama-3.1-8B-Instruct --datasets HotpotQA --datasets-dir /home/work/prompt/dpc/dataset/MRQA_benchmark --batch-size 4 --max-samples 32