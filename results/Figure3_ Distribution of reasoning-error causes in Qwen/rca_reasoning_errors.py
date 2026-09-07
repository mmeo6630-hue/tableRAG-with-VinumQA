import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "TableRAG_Qwen_DeepSeek_LLM_Judge_Details.xlsx"
OUT_DIR = ROOT / "outputs"
DETAIL_CSV = OUT_DIR / "reasoning_error_rca_details.csv"
REPORT_MD = OUT_DIR / "reasoning_error_rca_report.md"

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SHEETS = {"Qwen": "sheet2.xml", "DeepSeek": "sheet3.xml"}


def norm(text):
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", text.lower())).strip()


def contains_answer(answer, text):
    a = norm(answer)
    t = norm(text)
    if not a:
        return False
    if a in t:
        return True
    parts = [p.strip() for p in re.split(r"\bor\b|/|,|;", a) if p.strip()]
    return bool(parts) and any(p in t for p in parts if len(p) >= 3)


def compact(text, limit=260):
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def read_sheet_rows(sheet_xml):
    with zipfile.ZipFile(WORKBOOK) as zf:
        root = ET.fromstring(zf.read(f"xl/worksheets/{sheet_xml}"))
    raw_rows = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib["r"]
            col = "".join(ch for ch in ref if ch.isalpha())
            values[col] = cell.findtext("x:v", default="", namespaces=NS)
        raw_rows.append(values)

    header = raw_rows[0]
    col_by_name = {name: col for col, name in header.items()}
    rows = []
    for i, raw in enumerate(raw_rows[1:], start=1):
        row = {name: raw.get(col, "") for name, col in col_by_name.items()}
        row["_index"] = i
        rows.append(row)
    return rows


def load_trace(model, index):
    trace_dir = "traces_qwen" if model == "Qwen" else "traces_deepseek"
    path = ROOT / trace_dir / f"q_{index:06d}.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data, path


def flatten_sql_events(trace):
    events = trace.get("sql_events") or []
    out = []
    for ev in events:
        resp = ev.get("response") or {}
        sql = resp.get("sql_str") or resp.get("nl2sql_response") or ""
        result = resp.get("sql_execution_result")
        out.append((sql, "" if result is None else str(result), ev.get("query", "")))
    return out


def classify(row, trace):
    gold = row["Gold_Answer"]
    pred = row["Prediction"]
    question = row["Question"]
    context = trace.get("retrieved_context") or ""
    retrieved_files = trace.get("retrieved_files") or []
    subqueries = trace.get("subqueries") or []
    generated_sql = trace.get("generated_sql", "")
    sql_results = trace.get("sql_results")
    intermediate = trace.get("intermediate_answers") or []
    sql_events = flatten_sql_events(trace)
    all_sql_text = json.dumps(sql_events, ensure_ascii=False)
    evidence_pool = "\n".join(
        [
            context,
            json.dumps(sql_results, ensure_ascii=False),
            "\n".join(intermediate),
            all_sql_text,
        ]
    )

    pred_has_gold = contains_answer(gold, pred)
    evidence_has_gold = contains_answer(gold, evidence_pool)
    context_has_gold = contains_answer(gold, context)
    sql_has_gold = contains_answer(gold, json.dumps(sql_results, ensure_ascii=False))
    intermediate_has_gold = contains_answer(gold, "\n".join(intermediate))

    sql_error_terms = [
        "syntax error",
        "sql syntax",
        "unknown column",
        "no such column",
        "unknown table",
        "doesn't exist",
        "does not exist",
        "invalid identifier",
        "execution error",
        "error",
    ]
    syntax_terms = ["syntax error", "sql syntax", "parse", "near "]
    schema_terms = ["unknown column", "no such column", "unknown table", "doesn't exist", "does not exist", "invalid identifier"]
    lower_sql = all_sql_text.lower()

    trace_evidence = []
    secondary = ""

    if pred_has_gold:
        trace_evidence.append("Prediction already contains the gold answer.")
        return (
            "Other/Uncertain",
            "Judge false negative / answer appears correct",
            "LLM judge marked 0 although prediction matches gold",
            " ".join(trace_evidence),
            "Prediction includes the gold answer, so no clear pipeline reasoning failure can be established from the trace.",
        )

    if any(term in lower_sql for term in schema_terms):
        trace_evidence.append(compact(all_sql_text))
        return (
            "SQL",
            "Schema/column",
            "",
            " ".join(trace_evidence),
            "The first concrete failure is an invalid table/column/identifier in the SQL event.",
        )

    if any(term in lower_sql for term in syntax_terms):
        trace_evidence.append(compact(all_sql_text))
        return (
            "SQL",
            "Syntax",
            "",
            " ".join(trace_evidence),
            "The SQL event shows a syntax-level execution failure.",
        )

    if any(term in lower_sql for term in sql_error_terms):
        trace_evidence.append(compact(all_sql_text))
        return (
            "SQL",
            "Syntax",
            "",
            " ".join(trace_evidence),
            "The SQL event reports an execution error; the trace does not expose a more specific earlier failure.",
        )

    empty_sql = "[]" == compact(json.dumps(sql_results, ensure_ascii=False), 10) or any(
        result.strip() in {"[]", "", "None"} for _, result, _ in sql_events
    )
    if generated_sql and empty_sql and context_has_gold:
        trace_evidence.append(f"SQL returned empty result: {compact(generated_sql)}")
        secondary = "Final answer was forced from incomplete/empty SQL evidence"
        return (
            "SQL",
            "Value/type mismatch",
            secondary,
            " ".join(trace_evidence),
            "Relevant context was retrieved, but the SQL filter/value did not match the stored table values and returned no rows.",
        )

    if not context_has_gold and not evidence_has_gold:
        top_files = ", ".join(retrieved_files[:3])
        trace_evidence.append(f"Gold answer not found in retrieved context/evidence. Top files: {top_files}")
        return (
            "Retrieval",
            "Missing context",
            "Downstream SQL/synthesis lacked the needed evidence",
            " ".join(trace_evidence),
            "The trace does not contain the gold answer or equivalent evidence before reasoning begins.",
        )

    if retrieved_files:
        bases = [Path(f).stem.lower() for f in retrieved_files[:5]]
        if len(set(bases)) >= 4 and not context_has_gold:
            trace_evidence.append(f"Top-k files are dispersed: {', '.join(retrieved_files[:5])}")
            return (
                "Retrieval",
                "Wrong document/table",
                "Downstream SQL/reasoning operated on the wrong evidence set",
                " ".join(trace_evidence),
                "Retrieval brought mostly inconsistent documents/tables and did not surface the needed answer evidence.",
            )

    qn = norm(question)
    last_subq = norm(subqueries[-1] if subqueries else "")
    gold_n = norm(gold)
    asks_full_date = bool(re.search(r"\b(date|when|what day)\b", qn)) and re.search(r"\d{1,2}\s+[a-zA-Z]+\s+\d{4}", gold or "")
    narrowed_to_year = bool(re.search(r"\byear\b", last_subq)) and re.fullmatch(r"\d{4}", norm(pred).split(" ")[-1] if norm(pred).split(" ") else "")
    if subqueries and len(subqueries) > 1 and (asks_full_date and "date" not in last_subq and "day" not in last_subq or narrowed_to_year):
        trace_evidence.append(f"Subqueries: {compact(' | '.join(subqueries), 360)}")
        return (
            "Decomposition",
            "Condition/answer-type lost",
            "SQL and final answer followed the narrowed subquery",
            " ".join(trace_evidence),
            "The decomposition narrowed or changed the requested answer type before SQL/reasoning.",
        )

    if generated_sql and sql_results not in (None, [], "") and not sql_has_gold and evidence_has_gold:
        trace_evidence.append(f"SQL result: {compact(json.dumps(sql_results, ensure_ascii=False))}; SQL: {compact(generated_sql)}")
        return (
            "SQL",
            "Logical",
            "Final synthesis used wrong SQL output",
            " ".join(trace_evidence),
            "The SQL executed but returned the wrong value due to query logic, filtering, aggregation, or sorting.",
        )

    if evidence_has_gold and not pred_has_gold:
        where = []
        if context_has_gold:
            where.append("retrieved context")
        if sql_has_gold:
            where.append("SQL result")
        if intermediate_has_gold:
            where.append("intermediate answer")
        trace_evidence.append(f"Gold appears in {', '.join(where) or 'trace evidence'} but not in prediction.")
        return (
            "Final synthesis/reasoning",
            "Wrong selection/calculation/conclusion",
            "",
            " ".join(trace_evidence),
            "The needed evidence was present before the final response, but the model selected or synthesized the wrong answer.",
        )

    trace_evidence.append("Evidence is ambiguous under the available trace fields.")
    return (
        "Other/Uncertain",
        "Insufficient evidence",
        "",
        " ".join(trace_evidence),
        "The trace is not sufficient to assign a single first point of failure without guessing.",
    )


def gather():
    details = []
    system_errors = []
    for model, sheet_xml in SHEETS.items():
        for row in read_sheet_rows(sheet_xml):
            if row.get("LLM_Judge") != "0":
                continue
            if row.get("Figure9_Category") != "Reasoning Error":
                if row.get("Status") not in {"SUCCESS", ""}:
                    system_errors.append((model, row["_index"], row.get("Question_ID"), row.get("Status")))
                continue
            trace, trace_path = load_trace(model, row["_index"])
            primary, subtype, secondary, evidence, explanation = classify(row, trace)
            details.append(
                {
                    "model": model,
                    "qa_id": row["Question_ID"],
                    "trace_index": f"q_{row['_index']:06d}",
                    "question": row["Question"],
                    "gold": row["Gold_Answer"],
                    "prediction": row["Prediction"],
                    "Primary_Root_Cause": primary,
                    "Error_Subtype": subtype,
                    "Secondary_Error": secondary,
                    "Trace_Evidence": compact(evidence, 420),
                    "Short_Explanation": explanation,
                    "trace_file": str(trace_path.relative_to(ROOT)),
                }
            )
    return details, system_errors


def write_outputs(details, system_errors):
    OUT_DIR.mkdir(exist_ok=True)
    cols = [
        "model",
        "qa_id",
        "trace_index",
        "question",
        "gold",
        "prediction",
        "Primary_Root_Cause",
        "Error_Subtype",
        "Secondary_Error",
        "Trace_Evidence",
        "Short_Explanation",
        "trace_file",
    ]
    with DETAIL_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(details)

    total = len(details)
    counts = Counter(d["Primary_Root_Cause"] for d in details)
    subtype_counts = Counter((d["Primary_Root_Cause"], d["Error_Subtype"]) for d in details)
    by_model = Counter((d["model"], d["Primary_Root_Cause"]) for d in details)
    examples = defaultdict(list)
    for d in details:
        if len(examples[d["Primary_Root_Cause"]]) < 2:
            examples[d["Primary_Root_Cause"]].append(d)

    lines = []
    lines.append("# Root Cause Analysis for Reasoning Errors\n")
    lines.append(f"Input workbook: `{WORKBOOK.name}`\n")
    lines.append(f"Scope: `LLM_Judge = 0` and `Figure9_Category = Reasoning Error` for Qwen and DeepSeek.\n")
    lines.append(f"Total reasoning errors analyzed: **{total}**\n")
    if system_errors:
        lines.append("System/API-like failures excluded from RCA scope:\n")
        for item in system_errors:
            lines.append(f"- {item[0]} {item[1]} {item[2]} status={item[3]}\n")
    lines.append("\n## Summary\n\n")
    lines.append("| Root Cause | Count | % of Reasoning Errors |\n")
    lines.append("| --- | ---: | ---: |\n")
    for root, count in counts.most_common():
        lines.append(f"| {root} | {count} | {count / total * 100:.1f}% |\n")

    lines.append("\n## Summary by Model\n\n")
    lines.append("| Model | Root Cause | Count |\n")
    lines.append("| --- | --- | ---: |\n")
    for (model, root), count in sorted(by_model.items()):
        lines.append(f"| {model} | {root} | {count} |\n")

    lines.append("\n## Subtypes\n\n")
    lines.append("| Root Cause | Error Subtype | Count |\n")
    lines.append("| --- | --- | ---: |\n")
    for (root, subtype), count in subtype_counts.most_common():
        lines.append(f"| {root} | {subtype} | {count} |\n")

    lines.append("\n## Representative Examples\n\n")
    for root in counts:
        lines.append(f"### {root}\n\n")
        for d in examples[root]:
            lines.append(
                f"- `{d['model']}` `{d['trace_index']}` `{d['qa_id']}`: {compact(d['question'], 150)} "
                f"Gold=`{compact(d['gold'], 80)}` Prediction=`{compact(d['prediction'], 100)}`. "
                f"{d['Short_Explanation']} Evidence: {compact(d['Trace_Evidence'], 180)}\n"
            )
        lines.append("\n")

    lines.append("## Detailed Table\n\n")
    lines.append(
        "| qa_id | question | gold | prediction | Primary_Root_Cause | Error_Subtype | Secondary_Error | Trace_Evidence | Short_Explanation |\n"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for d in details:
        vals = [
            f"{d['model']}:{d['qa_id']}",
            d["question"],
            d["gold"],
            d["prediction"],
            d["Primary_Root_Cause"],
            d["Error_Subtype"],
            d["Secondary_Error"],
            d["Trace_Evidence"],
            d["Short_Explanation"],
        ]
        vals = [compact(v, 220).replace("|", "\\|") for v in vals]
        lines.append("| " + " | ".join(vals) + " |\n")

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    return counts, subtype_counts


if __name__ == "__main__":
    details, system_errors = gather()
    counts, subtype_counts = write_outputs(details, system_errors)
    print(f"details={DETAIL_CSV}")
    print(f"report={REPORT_MD}")
    print(f"total={len(details)}")
    print("root_counts", dict(counts))
    print("subtype_counts", {f"{k[0]}::{k[1]}": v for k, v in subtype_counts.items()})
