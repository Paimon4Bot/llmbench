from __future__ import annotations

import csv
import io
import json
from typing import Any


def flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_record(value, full_key))
        elif isinstance(value, list):
            flat[full_key] = json.dumps(value, ensure_ascii=True)
        else:
            flat[full_key] = value
    return flat


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [flatten_record(record) for record in records]


def records_to_csv(records: list[dict[str, Any]]) -> str:
    normalized = normalize_records(records)
    if not normalized:
        return ""
    headers = sorted({key for row in normalized for key in row.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for row in normalized:
        writer.writerow({header: row.get(header, "") for header in headers})
    return output.getvalue()


def records_to_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records)


def records_to_stdout(records: list[dict[str, Any]]) -> str:
    normalized = normalize_records(records)
    if not normalized:
        return "No benchmark records were produced."
    if len(normalized) == 1:
        row = normalized[0]
        width = max(len(key) for key in row.keys())
        return "\n".join(f"{key.ljust(width)} : {row[key]}" for key in sorted(row.keys()))

    headers = sorted({key for row in normalized for key in row.keys()})
    widths = {header: len(header) for header in headers}
    for row in normalized:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    lines = []
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    lines.append(header_line)
    lines.append(separator)
    for row in normalized:
        lines.append(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    return "\n".join(lines)
