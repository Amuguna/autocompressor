"""
Evaluate AutoCompressor models on PwC QA JSONL datasets.

- Loads a base AutoCompressor (Llama/Qwen/OPT) and optional LoRA.
- Compresses context into softprompt summary vectors, then generates answers.
- Reports Exact Match and token-level F1; writes predictions and summary metrics.

Dataset format (per JSONL row):
- context: one of `input` | `context` | `passage`
- question: one of `prompt` | `question`
- answer: one of `answer` | `target` | `output`
Rows missing any of these are skipped.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from auto_compressor import (
    LlamaAutoCompressorModel,
    OPTAutoCompressorModel,
    QwenAutoCompressorModel,
)


def normalize_answer(text: str) -> str:
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
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    overlap = sum(min(pred_counts.get(tok, 0), gold_counts.get(tok, 0)) for tok in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    return max(metric_fn(prediction, g) for g in ground_truths) if ground_truths else 0.0


@dataclass
class PwCSample:
    id: str
    context_tokens: torch.Tensor
    context_text: str
    question: str
    answers: List[str]


class PwCDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_context_length: int, max_samples: int | None = None) -> None:
        self.samples: List[PwCSample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                context = obj.get("input") or obj.get("context") or obj.get("passage")
                question = obj.get("prompt") or obj.get("question")
                answer = obj.get("answer") or obj.get("target") or obj.get("output")
                if not context or not question or not answer:
                    continue
                answers = answer if isinstance(answer, list) else [answer]
                tokenized = tokenizer(
                    context,
                    truncation=True,
                    padding="max_length",
                    max_length=max_context_length,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                context_tokens = tokenized.input_ids.squeeze(0)
                self.samples.append(
                    PwCSample(
                        id=str(obj.get("id", len(self.samples))),
                        context_tokens=context_tokens,
                        context_text=context,
                        question=question,
                        answers=answers,
                    )
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PwCSample:
        return self.samples[idx]


def build_collate_fn():
    def _collate(batch: List[PwCSample]) -> Dict[str, object]:
        return {
            "context_tokens": torch.stack([b.context_tokens for b in batch]),
            "context_texts": [b.context_text for b in batch],
            "questions": [b.question for b in batch],
            "answers": [b.answers for b in batch],
            "ids": [b.id for b in batch],
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

    name = args.model_path.lower()
    if "llama" in name:
        ac_cls = LlamaAutoCompressorModel
    elif "qwen" in name:
        ac_cls = QwenAutoCompressorModel
    else:
        ac_cls = OPTAutoCompressorModel

    config = AutoConfig.from_pretrained(args.model_path)
    if args.summary_length is not None:
        config.summary_length = args.summary_length
    if not hasattr(config, "summary_length"):
        raise ValueError("summary_length missing; set --summary-length or use a checkpoint that includes it.")
    if not hasattr(config, "accumulate_summary"):
        config.accumulate_summary = False
    if not hasattr(config, "segment_gradient_checkpointing"):
        config.segment_gradient_checkpointing = False

    try:
        base_model = ac_cls.from_pretrained(args.model_path, config=config, dtype=dtype)
    except TypeError:
        base_model = ac_cls.from_pretrained(args.model_path, config=config, torch_dtype=dtype)

    if args.lora_path:
        base_model = PeftModel.from_pretrained(base_model, args.lora_path)

    if getattr(base_model, "generation_config", None) is None:
        base_model.generation_config = GenerationConfig.from_model_config(base_model.config)
    if hasattr(base_model, "base_model") and getattr(base_model.base_model, "generation_config", None) is None:
        base_model.base_model.generation_config = base_model.generation_config

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


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args, device)

    dataset = PwCDataset(Path(args.dataset_path), model.tokenizer, args.max_context_length, args.max_samples)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty or unreadable.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=build_collate_fn())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions_full.jsonl"
    metrics_path = output_dir / "metrics_summary.json"

    em_total = 0.0
    f1_total = 0.0
    sample_count = 0

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for batch in tqdm(dataloader, desc="Evaluating PwC"):
            context_tokens = batch["context_tokens"].to(device)
            softprompt = model.compress(context_tokens)

            prompts = [args.prompt_template.format(question=q) for q in batch["questions"]]
            preds = model.predict(softprompt, prompts)

            for idx, pred in enumerate(preds):
                gold_answers = batch["answers"][idx]
                em = metric_max_over_ground_truths(exact_match, pred, gold_answers)
                f1 = metric_max_over_ground_truths(f1_score, pred, gold_answers)
                em_total += em
                f1_total += f1
                sample_count += 1

                pred_f.write(
                    json.dumps(
                        {
                            "id": batch["ids"][idx],
                            "question": batch["questions"][idx],
                            "context": batch["context_texts"][idx],
                            "answers": gold_answers,
                            "prediction": pred,
                            "em": em,
                            "f1": f1,
                            "prompt": prompts[idx],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "samples": sample_count,
        "exact_match": em_total / sample_count if sample_count else 0.0,
        "f1": f1_total / sample_count if sample_count else 0.0,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "summary_length": model.info.get("summary_length"),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(summary)
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AutoCompressor on PwC JSONL dataset.")
    parser.add_argument("--dataset-path", required=True, help="Path to PwC JSONL file.")
    parser.add_argument("--model-path", required=True, help="Base AutoCompressor model name or path.")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--output-dir", default="./predictions/pwc_autocompressor", help="Output directory.")
    parser.add_argument("--summary-length", type=int, default=None, help="Manually set summary_length if missing in the base config.")
    parser.add_argument("--max-context-length", type=int, default=512, help="Max context tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--prompt-template", default="Question: {question} Answer: ", help="Prompt template applied before generation.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
