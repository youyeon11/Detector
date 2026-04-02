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

from app.domain.variation_types import VariationType
from app.services.variation_detection import VariationDetectionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate variation detection metrics on a JSONL dataset."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "dataset" / "variation_eval_messages.jsonl",
        help="Path to variation evaluation JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "v4_eval_report.md",
        help="Optional output path for the evaluation report. Supports .json and .md.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path for the evaluation report.",
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
    detector: Any,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    variation_confusion: Counter[tuple[str, str]] = Counter()

    tp = fp = tn = fn = 0
    canonical_correct = 0
    canonical_total = 0
    variation_correct = 0
    variation_total = 0

    for row in dataset:
        text = str(row["text"])
        expected_detected = bool(row["expected_detected"])
        expected_canonical = row.get("expected_canonical")
        expected_variation_type = row.get("expected_variation_type")
        detection = detector.detect_document(text)

        predicted_detected = bool(detection.profanity_detected)
        predicted_hit = detection.profanity_hits[0] if detection.profanity_hits else None
        predicted_canonical = predicted_hit.canonical if predicted_hit is not None else None
        predicted_variation_type = (
            predicted_hit.variation_type.value if predicted_hit is not None else None
        )

        if expected_detected and predicted_detected:
            tp += 1
        elif (not expected_detected) and predicted_detected:
            fp += 1
        elif (not expected_detected) and (not predicted_detected):
            tn += 1
        else:
            fn += 1

        if expected_detected:
            canonical_total += 1
            if expected_canonical == predicted_canonical:
                canonical_correct += 1

            if expected_variation_type is not None:
                variation_total += 1
                if expected_variation_type == predicted_variation_type:
                    variation_correct += 1
                variation_confusion[(str(expected_variation_type), str(predicted_variation_type))] += 1

        rows.append(
            {
                "text": text,
                "expected_detected": expected_detected,
                "predicted_detected": predicted_detected,
                "expected_canonical": expected_canonical,
                "predicted_canonical": predicted_canonical,
                "expected_variation_type": expected_variation_type,
                "predicted_variation_type": predicted_variation_type,
                "message_normalized": detection.message_normalized,
                "hit_count": len(detection.profanity_hits),
                "top_label": predicted_hit.label.value if predicted_hit is not None else None,
                "top_score": round(predicted_hit.score, 4) if predicted_hit is not None else None,
                "top_reasons": predicted_hit.reasons if predicted_hit is not None else [],
                "case_group": row.get("case_group", "unknown"),
                "notes": row.get("notes", ""),
            }
        )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(fp, fp + tn)
    canonical_match_accuracy = safe_divide(canonical_correct, canonical_total)
    variation_classification_accuracy = safe_divide(variation_correct, variation_total)

    false_positives = [
        row
        for row in rows
        if row["expected_detected"] is False and row["predicted_detected"] is True
    ]
    false_negatives = [
        row
        for row in rows
        if row["expected_detected"] is True and row["predicted_detected"] is False
    ]
    wrong_variation_type = [
        row
        for row in rows
        if row["expected_variation_type"] is not None
        and row["predicted_detected"] is True
        and row["expected_variation_type"] != row["predicted_variation_type"]
    ]

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
            "canonical_match_accuracy": round(canonical_match_accuracy, 4),
            "variation_classification_accuracy": round(variation_classification_accuracy, 4),
            "variation_case_count": variation_total,
        },
        "variation_type_confusion": build_variation_confusion(variation_confusion),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "wrong_variation_type": wrong_variation_type,
        "rows": rows,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_variation_confusion(confusion: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (expected, predicted), count in sorted(confusion.items()):
        rows.append(
            {
                "expected_variation_type": expected,
                "predicted_variation_type": predicted,
                "count": count,
            }
        )
    return rows


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Variation Evaluation Report",
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
        f"| Canonical Match Accuracy | {summary['canonical_match_accuracy']:.4f} |",
        f"| Variation Classification Accuracy | {summary['variation_classification_accuracy']:.4f} |",
        f"| Variation Case Count | {summary['variation_case_count']} |",
        "",
        "## Variation Type Confusion",
        "",
        "| Expected | Predicted | Count |",
        "|---|---|---:|",
    ]
    for row in report["variation_type_confusion"]:
        lines.append(
            f"| {row['expected_variation_type']} | {row['predicted_variation_type']} | {row['count']} |"
        )
    if not report["variation_type_confusion"]:
        lines.append("| none | none | 0 |")

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
            f"## False Negatives ({len(report['false_negatives'])})",
            "",
        ]
    )
    lines.extend(render_case_bullets(report["false_negatives"]))
    lines.extend(
        [
            "",
            f"## Wrong Variation Type ({len(report['wrong_variation_type'])})",
            "",
        ]
    )
    lines.extend(render_case_bullets(report["wrong_variation_type"]))
    return "\n".join(lines)


def render_case_bullets(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [
        (
            f"- expected_detected={row['expected_detected']} predicted_detected={row['predicted_detected']} "
            f"expected_canonical={row['expected_canonical']} predicted_canonical={row['predicted_canonical']} "
            f"expected_variation={row['expected_variation_type']} predicted_variation={row['predicted_variation_type']} "
            f"text={row['text']}"
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
    print("Variation evaluation summary")
    print(f"sample_count={summary['sample_count']}")
    print(f"precision={summary['precision']:.4f}")
    print(f"recall={summary['recall']:.4f}")
    print(f"f1={summary['f1']:.4f}")
    print(f"false_positive_rate={summary['false_positive_rate']:.4f}")
    print(f"canonical_match_accuracy={summary['canonical_match_accuracy']:.4f}")
    print(f"variation_classification_accuracy={summary['variation_classification_accuracy']:.4f}")
    print(f"false_positives={len(report['false_positives'])}")
    print(f"false_negatives={len(report['false_negatives'])}")
    print(f"wrong_variation_type={len(report['wrong_variation_type'])}")


def main() -> int:
    args = parse_args()
    dataset = load_eval_dataset(args.path, limit=args.limit)
    detector = VariationDetectionService()
    report = evaluate_dataset(detector, dataset)
    print_summary(report)

    if args.output is not None:
        save_report(args.output, report)
        print(f"wrote report: {args.output}")

    if args.json_output is not None:
        save_report(args.json_output, report)
        print(f"wrote report: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
