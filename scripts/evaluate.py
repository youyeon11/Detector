from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.enums import DetectionLabel
from app.services.detector import DetectorService


POSITIVE_LABELS = {DetectionLabel.BLOCK.value, DetectionLabel.REVIEW.value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector quality metrics on a JSONL dataset.")
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "dataset" / "eval_sentences.jsonl",
        help="Path to evaluation JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the evaluation report. Supports .json and .md.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick local runs.",
    )
    return parser.parse_args()


def load_eval_dataset(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {line_number}: {path}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def evaluate_dataset(
    detector: DetectorService,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    label_confusion: Counter[tuple[str, str]] = Counter()

    tp = fp = tn = fn = 0

    for row in dataset:
        text = str(row["text"])
        expected_label = str(row["label"])
        detection = detector.detect(text)
        predicted_label = detection.label.value

        expected_positive = expected_label in POSITIVE_LABELS
        predicted_positive = predicted_label in POSITIVE_LABELS

        if expected_positive and predicted_positive:
            tp += 1
        elif (not expected_positive) and predicted_positive:
            fp += 1
        elif (not expected_positive) and (not predicted_positive):
            tn += 1
        else:
            fn += 1

        label_confusion[(expected_label, predicted_label)] += 1
        rows.append(
            {
                "text": text,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "score": detection.score,
                "matched_term": detection.matched_term,
                "normalized_text": detection.normalized_text,
                "reasons": detection.reasons,
                "source": row.get("source", "unknown"),
                "notes": row.get("notes", ""),
            }
        )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(fp, fp + tn)

    return {
        "summary": {
            "sample_count": len(rows),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(false_positive_rate, 4),
        },
        "label_confusion": build_label_confusion(label_confusion),
        "false_positives": [
            row
            for row in rows
            if row["expected_label"] == DetectionLabel.PASS.value
            and row["predicted_label"] in POSITIVE_LABELS
        ],
        "misses": [
            row
            for row in rows
            if row["expected_label"] in POSITIVE_LABELS
            and row["predicted_label"] == DetectionLabel.PASS.value
        ],
        "rows": rows,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_label_confusion(confusion: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (expected, predicted), count in sorted(confusion.items()):
        rows.append(
            {
                "expected_label": expected,
                "predicted_label": predicted,
                "count": count,
            }
        )
    return rows


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sample Count | {summary['sample_count']} |",
        f"| TP | {summary['tp']} |",
        f"| FP | {summary['fp']} |",
        f"| TN | {summary['tn']} |",
        f"| FN | {summary['fn']} |",
        f"| Precision | {summary['precision']:.4f} |",
        f"| Recall | {summary['recall']:.4f} |",
        f"| F1 | {summary['f1']:.4f} |",
        f"| False Positive Rate | {summary['false_positive_rate']:.4f} |",
        "",
        "## Label Confusion",
        "",
        "| Expected | Predicted | Count |",
        "|---|---|---:|",
    ]
    for row in report["label_confusion"]:
        lines.append(
            f"| {row['expected_label']} | {row['predicted_label']} | {row['count']} |"
        )

    lines.extend(
        [
            "",
            f"## False Positives ({len(report['false_positives'])})",
            "",
        ]
    )
    lines.extend(render_case_bullets(report["false_positives"]))
    lines.extend(
        [
            "",
            f"## Misses ({len(report['misses'])})",
            "",
        ]
    )
    lines.extend(render_case_bullets(report["misses"]))
    return "\n".join(lines)


def render_case_bullets(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [
        (
            f"- expected={row['expected_label']} predicted={row['predicted_label']} "
            f"score={row['score']}: {row['text']}"
        )
        for row in rows
    ]


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".md":
        path.write_text(render_markdown_report(report), encoding="utf-8")
        return
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Evaluation summary")
    print(f"sample_count={summary['sample_count']}")
    print(f"precision={summary['precision']:.4f}")
    print(f"recall={summary['recall']:.4f}")
    print(f"f1={summary['f1']:.4f}")
    print(f"false_positive_rate={summary['false_positive_rate']:.4f}")
    print(f"false_positives={len(report['false_positives'])}")
    print(f"misses={len(report['misses'])}")


def main() -> int:
    args = parse_args()
    dataset = load_eval_dataset(args.path, limit=args.limit)
    detector = DetectorService()
    report = evaluate_dataset(detector, dataset)
    print_summary(report)

    if args.output is not None:
        save_report(args.output, report)
        print(f"wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
