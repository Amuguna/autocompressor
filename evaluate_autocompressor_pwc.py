"""
Evaluate AutoCompressor models on PwC QA JSONL datasets.

- Loads a base AutoCompressor (Llama/Qwen/OPT) and optional LoRA.
 - Compresses context into softprompt summary vectors, then generates answers.
 - Outputs CSVs aligned with predict_qa_500x.py: per-sample predictions, per-sample metrics, and summary metrics (ROUGE, BLEU, EM, F1, timings).

Dataset format (per JSONL row):
- context: one of `input` | `context` | `passage`
- question: one of `prompt` | `question`
- answer: one of `answer` | `target` | `output`
Rows missing any of these are skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
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

# Optional metric dependencies. We fall back to lightweight behavior if missing.
try:  # noqa: SIM105
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    _TOKENIZER_FN = nltk.word_tokenize
    _SMOOTHING_FN = SmoothingFunction().method1
except Exception:  # pragma: no cover - optional dependency guard
    nltk = None
    sentence_bleu = None
    _TOKENIZER_FN = lambda text: text.split()  # noqa: E731
    _SMOOTHING_FN = None

try:  # noqa: SIM105
    from rouge import Rouge

    _ROUGE = Rouge()
except Exception:  # pragma: no cover - optional dependency guard
    _ROUGE = None


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


def sanitize_lora_path(lora_path: str | None, model_path: str) -> str:
    """Create a stable, filesystem-safe run name from the LoRA/model path."""
    base_path = Path(lora_path) if lora_path else Path(model_path)
    tail_parts: Sequence[str]
    if len(base_path.parts) >= 2:
        tail_parts = base_path.parts[-2:]
    else:
        tail_parts = base_path.parts
    base = "__".join(tail_parts) if tail_parts else base_path.name
    safe_base = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in base)
    digest = hashlib.md5(str(base_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{safe_base}_{digest}"


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts.get(tok, 0), gold_counts.get(tok, 0)) for tok in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_metrics(reference: str, candidate: str) -> Dict[str, float]:
    reference = reference.strip()
    candidate = candidate.strip()

    metrics = {
        "rouge-1-p": 0.0,
        "rouge-1-r": 0.0,
        "rouge-1-f": 0.0,
        "rouge-2-p": 0.0,
        "rouge-2-r": 0.0,
        "rouge-2-f": 0.0,
        "rouge-l-p": 0.0,
        "rouge-l-r": 0.0,
        "rouge-l-f": 0.0,
        "bleu": 0.0,
        "exact_match": exact_match(candidate, reference),
        "f1": f1_score(candidate, reference),
    }

    if not reference or not candidate:
        return metrics

    if _ROUGE is not None:
        try:
            rouge_scores = _ROUGE.get_scores(candidate, reference)[0]
            for key in ("rouge-1", "rouge-2", "rouge-l"):
                metrics[f"{key}-p"] = rouge_scores[key]["p"]
                metrics[f"{key}-r"] = rouge_scores[key]["r"]
                metrics[f"{key}-f"] = rouge_scores[key]["f"]
        except ValueError:
            pass

    try:
        reference_tokens = _TOKENIZER_FN(reference) if reference else []
        candidate_tokens = _TOKENIZER_FN(candidate) if candidate else []
    except Exception:
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

    if reference_tokens and candidate_tokens and sentence_bleu is not None:
        metrics["bleu"] = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=_SMOOTHING_FN)

    return metrics


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

    def _predict(softprompt: torch.Tensor, prompts: List[str]) -> tuple[List[str], List[bool], List[int]]:
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
                return_dict_in_generate=True,
                output_scores=False,
            )

        sequences = outputs.sequences.detach().cpu()
        prompt_lengths = attention_mask.sum(dim=1).cpu().tolist()
        responses: List[str] = []
        generated_lengths: List[int] = []
        ended_flags: List[bool] = []
        for idx, prompt_len in enumerate(prompt_lengths):
            generated_ids = sequences[idx, prompt_len:]
            responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
            generated_lengths.append(int(len(generated_ids)))
            ended_flags.append(bool(tokenizer.eos_token_id in generated_ids.tolist()))
        return responses, ended_flags, generated_lengths

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

    run_name = sanitize_lora_path(args.lora_path, args.model_path)
    prediction_csv_path = output_dir / f"{run_name}_predictions.csv"
    metrics_csv_path = output_dir / f"{run_name}_metrics.csv"
    summary_csv_path = output_dir / f"{run_name}_metrics_summary.csv"

    metric_keys = [
        "rouge-1-p",
        "rouge-1-r",
        "rouge-1-f",
        "rouge-2-p",
        "rouge-2-r",
        "rouge-2-f",
        "rouge-l-p",
        "rouge-l-r",
        "rouge-l-f",
        "bleu",
        "exact_match",
        "f1",
    ]

    aggregate_metrics = {key: 0.0 for key in metric_keys}
    total_generated_tokens = 0.0
    total_compress_time = 0.0
    total_predict_time = 0.0

    detailed_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    sample_count = 0

    for batch in tqdm(dataloader, desc="Evaluating PwC"):
        context_tokens = batch["context_tokens"].to(device)

        _sync_if_cuda(device)
        compress_start = time.perf_counter()
        softprompt = model.compress(context_tokens)
        _sync_if_cuda(device)
        compress_duration = time.perf_counter() - compress_start
        per_sample_compress = compress_duration / len(batch["ids"])

        prompts = [args.prompt_template.format(question=q) for q in batch["questions"]]

        _sync_if_cuda(device)
        predict_start = time.perf_counter()
        preds, ended_flags, generated_lengths = model.predict(softprompt, prompts)
        _sync_if_cuda(device)
        predict_duration = time.perf_counter() - predict_start
        per_sample_predict = predict_duration / len(batch["ids"])

        for idx, pred in enumerate(preds):
            gold_answers = batch["answers"][idx]
            reference = gold_answers[0] if gold_answers else ""
            metrics = compute_metrics(reference, pred)

            for key in metric_keys:
                aggregate_metrics[key] += metrics[key]

            generated_len = generated_lengths[idx]
            total_generated_tokens += generated_len
            total_compress_time += per_sample_compress
            total_predict_time += per_sample_predict
            sample_count += 1

            detailed_rows.append(
                {
                    "sample_id": batch["ids"][idx],
                    **{key: metrics[key] for key in metric_keys},
                    "generated_tokens": generated_len,
                    "compress_time_s": per_sample_compress,
                    "predict_time_s": per_sample_predict,
                    "stopped_by_token": 1 if ended_flags[idx] else 0,
                }
            )

            prediction_rows.append(
                {
                    "sample_id": batch["ids"][idx],
                    "question": batch["questions"][idx],
                    "context": batch["context_texts"][idx],
                    "ground_truth": reference,
                    "prediction": pred,
                    "prompt": prompts[idx],
                    **{key: metrics[key] for key in metric_keys},
                    "generated_tokens": generated_len,
                    "compress_time_s": per_sample_compress,
                    "predict_time_s": per_sample_predict,
                    "stopped_by_token": 1 if ended_flags[idx] else 0,
                }
            )

    if sample_count == 0:
        raise ValueError("No samples evaluated. Please check dataset and filters.")

    for key in aggregate_metrics:
        aggregate_metrics[key] /= sample_count

    averages = {
        **aggregate_metrics,
        "generated_tokens_avg": total_generated_tokens / sample_count,
        "compress_time_s_avg": total_compress_time / sample_count,
        "predict_time_s_avg": total_predict_time / sample_count,
        "samples": sample_count,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "summary_length": model.info.get("summary_length"),
    }

    prediction_fieldnames = [
        "sample_id",
        "question",
        "context",
        "ground_truth",
        "prediction",
        "prompt",
        *metric_keys,
        "generated_tokens",
        "compress_time_s",
        "predict_time_s",
        "stopped_by_token",
    ]

    with prediction_csv_path.open("w", encoding="utf-8", newline="") as pred_file:
        writer = csv.DictWriter(pred_file, fieldnames=prediction_fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow(row)

    metric_fieldnames = [
        "sample_id",
        *metric_keys,
        "generated_tokens",
        "compress_time_s",
        "predict_time_s",
        "stopped_by_token",
    ]

    with metrics_csv_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=metric_fieldnames)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    with summary_csv_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["metric", "value"])
        for key, value in averages.items():
            writer.writerow([key, value])

    print("\nDone.")
    print(f"Samples: {sample_count}")
    print(f"Exact Match: {averages['exact_match']:.4f}")
    print(f"F1: {averages['f1']:.4f}")
    print(f"ROUGE-1 F1: {averages['rouge-1-f']:.4f}")
    print(f"ROUGE-2 F1: {averages['rouge-2-f']:.4f}")
    print(f"ROUGE-L F1: {averages['rouge-l-f']:.4f}")
    print(f"BLEU: {averages['bleu']:.4f}")
    print(f"Predictions: {prediction_csv_path}")
    print(f"Metrics: {metrics_csv_path}")
    print(f"Summary: {summary_csv_path}")


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

# CUDA_VISIBLE_DEVICES=0 python evaluate_autocompressor_pwc.py --lora-path /home/work/prompt/dpc/autocompressor/checkpoints/ac_Llama-3.1-8B-Instruct_sub2_seg2_sum16_lr8e-4_bsz64_rand_accu/checkpoint-5000 --summary-length 16 --batch-size 32 --output-dir ./predictions/pwc/llama8b --dataset-path /home/work/prompt/dpc/dataset/PwC/PwC_test.jsonl --model-path /home/work/prompt/models/Llama-3.1-8B-Instruct --max-samples 32
