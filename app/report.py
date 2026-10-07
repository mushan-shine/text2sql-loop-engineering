"""Analysis report for a console run — built from the run's events, no LLM call.

The loop part of the report (what was observed, diagnosed, repaired) uses only
agent-visible data. Correctness comes from the ``judged`` events, which the
evaluation side adds after each case's loop has finished; the report labels it.

Any number of repair rounds is supported (attempts = repairs + 1). Chart data
(``first_final``, ``attempt_progress``, ``status_grid``) is computed here so the
page only draws it and the numbers are unit-tested.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter, defaultdict
from typing import Any

TYPE_CN = {"TABLE_RETRIEVAL_FAILURE": "选表", "COLUMN_MAPPING_FAILURE": "列映射", "JOIN_KEY_FAILURE": "关联键",
           "DOMAIN_KNOWLEDGE_FAILURE": "领域知识", "QUERY_DECOMPOSITION_FAILURE": "查询拆解",
           "EXECUTION_FAILURE": "执行错误", "UNKNOWN": "未知"}
STRATEGY_CN = {"targeted": "Targeted Loop（诊断 → 定向修复）", "generic": "Generic Retry（通用重试）"}
STATUS = ("报错", "能执行但错", "答对")          # per-attempt status buckets (status grid)
METRICS = ("可执行", "自检通过", "答对")        # per-attempt metrics; 自检 = SelfVerifier (no gold)


def error_class(msg: str | None) -> str | None:
    m = re.search(r"\[([A-Z][A-Z0-9_]+)", msg or "")
    return m.group(1) if m else None


def outcome(first_correct: bool, final_correct: bool, first_status: str, final_status: str) -> str:
    """Same buckets as the console's case trace; 'final' is the attempt the loop chose, not the last one."""
    if first_correct and final_correct:
        return "首次即对"
    if final_correct:
        return "已恢复"
    if first_correct:
        return "被误伤"
    if first_status != "SUCCESS" and final_status == "SUCCESS":
        return "报错 → 可执行（仍错）"
    return "仍报错" if final_status != "SUCCESS" else "可执行但错"


def cases_from_events(events: list[dict]) -> list[dict[str, Any]]:
    """Group events per case into one record per case (order of appearance)."""
    order, by = [], defaultdict(list)
    for e in events:
        cid = e.get("case_id")
        if not cid:
            continue
        if cid not in by:
            order.append(cid)
        by[cid].append(e)
    out = []
    for cid in order:
        ev = by[cid]

        def all_of(step: str) -> list[dict]:
            return [e for e in ev if e["step"] == step]

        start = next(iter(all_of("start")), {})
        execs, judged, final = all_of("execute"), next(iter(all_of("judged")), None), next(iter(all_of("final")), None)
        rec = {
            "case_id": cid, "question": start.get("question", ""),
            "retrieve": next(iter(all_of("retrieve")), None), "generate": next(iter(all_of("generate")), None),
            "executions": execs, "verifies": all_of("verify"), "observes": all_of("observe"),
            "diagnoses": all_of("diagnose"), "routes": all_of("route"), "repairs": all_of("repair"),
            "final": final, "judged": judged, "stop": next(iter(all_of("stop")), None),
        }
        k = (final["final_attempt"] - 1) if final else len(execs) - 1
        if judged and execs:
            rec["outcome"] = outcome(judged["attempt_correct"][0], judged["final_correct"],
                                     execs[0]["execution_status"], execs[min(k, len(execs) - 1)]["execution_status"])
        else:
            rec["outcome"] = "未完成"
        out.append(rec)
    return out


def preview_block(preview: str | None, n_rows: int | None, max_rows: int = 5) -> str:
    """The stored result preview (first rows, as JSON) shown whole row by row — never cut inside a value.
    Display only: judging always uses the complete result rows, not this preview."""
    try:
        rows = json.loads(preview) if preview else []
    except (TypeError, ValueError):
        rows = []
    if not rows:
        return f"- **最终答案结果**：{n_rows if n_rows is not None else 0} 行"
    shown = rows[:max_rows]
    total = n_rows if n_rows is not None else len(rows)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in shown)
    return (f"- **最终答案结果预览**（前 {len(shown)} 行，共 {total} 行；判分使用完整结果）：\n\n"
            f"```json\n{lines}\n```")


def attempt_status(case: dict, i: int) -> str:
    """Status of attempt i (0-based) of a finished case."""
    if case["judged"]["attempt_correct"][i]:
        return "答对"
    return "能执行但错" if case["executions"][i]["execution_status"] == "SUCCESS" else "报错"


# ------------------------------------------------------------------ chart data

def first_final(cases: list[dict]) -> list[dict]:
    """Counts for the first attempt vs the loop's chosen final answer."""
    done = [c for c in cases if c["judged"] and c["executions"]]
    rows = []
    for stage in ("首次", "最终"):
        def idx(c):
            return 0 if stage == "首次" else (c["final"]["final_attempt"] - 1 if c["final"] else len(c["executions"]) - 1)
        rows += [
            {"指标": "可执行", "阶段": stage, "题数": sum(c["executions"][idx(c)]["execution_status"] == "SUCCESS" for c in done)},
            {"指标": "自检通过", "阶段": stage, "题数": sum(bool(c["verifies"][idx(c)]["passed"]) for c in done)},
            {"指标": "答对", "阶段": stage, "题数": sum(bool(c["judged"]["attempt_correct"][idx(c)]) for c in done)},
        ]
    return rows


def attempt_progress(cases: list[dict], max_attempts: int) -> list[dict]:
    """Metric counts 'after attempt k' for k = 1..max_attempts. A case that already stopped (verifier
    PASS or out of budget) keeps its last attempt, so each point answers: what if the loop allowed k attempts?"""
    done = [c for c in cases if c["judged"] and c["executions"]]
    rows = []
    for k in range(1, max_attempts + 1):
        counts = Counter()
        for c in done:
            i = min(k, len(c["executions"])) - 1
            counts["可执行"] += c["executions"][i]["execution_status"] == "SUCCESS"
            counts["自检通过"] += bool(c["verifies"][i]["passed"])
            counts["答对"] += bool(c["judged"]["attempt_correct"][i])
        rows += [{"尝试": k, "指标": m, "题数": counts[m]} for m in METRICS]
    return rows


def status_grid(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        if not (c["judged"] and c["executions"]):
            continue
        fin = (c["final"]["final_attempt"] if c["final"] else len(c["executions"]))
        for i, e in enumerate(c["executions"]):
            rep = next((r for r in c["repairs"] if r["attempt_id"] == i + 1), None)
            rows.append({"题目": c["case_id"].split(":")[-1], "尝试": i + 1, "状态": attempt_status(c, i),
                         "执行": e["execution_status"] + (f" {error_class(e.get('execution_error'))}"
                                                         if error_class(e.get("execution_error")) else ""),
                         "自检": "通过" if c["verifies"][i]["passed"] else "未通过",
                         "由谁生成": rep["skill"] if rep else "首次生成", "最终答案": "是" if i + 1 == fin else ""})
    return rows


# ------------------------------------------------------------------ markdown

def _pct(a: int, b: int) -> str:
    return f"{a}/{b}" + (f"（{100 * a / b:.0f}%）" if b else "")


def build_report(events: list[dict], request: dict, summary: dict | None, run_id: str | None,
                 elapsed_s: float | None = None) -> str:
    cases = cases_from_events(events)
    done = [c for c in cases if c["judged"] and c["executions"]]
    n = len(done)
    max_attempts = int(request.get("max_repairs", 1)) + 1
    L: list[str] = []
    add = L.append

    add("# Loop 运行分析报告\n")
    add(f"- **运行**：`{run_id or '—'}`")
    add(f"- **配置**：模型 `{request.get('model')}` · {STRATEGY_CN.get(request.get('strategy'), request.get('strategy'))}"
        f" · Verifier `{request.get('verifier')}` · 最多修复 {max_attempts - 1} 次（最多 {max_attempts} 次 SQL 尝试）"
        f" · {'开发集' if request.get('split') == 'dev' else '评测集'} · {n} 题")
    if elapsed_s is not None:
        add(f"- **Loop 运行耗时**：{elapsed_s:.0f} 秒（含连接 SQL Warehouse，不含之后发布到 Delta）")
    add("- **说明**：Loop 运行时只使用无 Gold 的信号；“自检通过”只表示没发现报错、空结果等明显问题，不代表答案正确；表中“对 / 错”来自每题 Loop 结束后的 Gold 判分（评测侧），Loop 看不到。\n")

    # 1. overall
    ff = {(r["指标"], r["阶段"]): r["题数"] for r in first_final(cases)}
    recovered = sum(not c["judged"]["attempt_correct"][0] and c["judged"]["final_correct"] for c in done)
    harmed = sum(c["judged"]["attempt_correct"][0] and not c["judged"]["final_correct"] for c in done)
    triggered = [c for c in done if not c["verifies"][0]["passed"]]
    miss = [c for c in done if c["final"] and c["final"]["verifier_decision"] == "PASS" and not c["judged"]["final_correct"]]
    tok = sum(c["judged"]["tokens"] for c in done)
    rep_tok = sum(r.get("tokens", 0) + r.get("diag_tokens", 0) for c in done for r in c["repairs"])
    rounds = sum(len(c["repairs"]) for c in done)
    add("## 1. 总体结果\n")
    add("| 指标 | 首次 | 最终 |\n|---|---|---|")
    for m in METRICS:
        add(f"| {m} | {ff.get((m, '首次'), 0)} / {n} | {ff.get((m, '最终'), 0)} / {n} |")
    add("")
    add("| 其他 | 结果 |\n|---|---|")
    add(f"| 恢复 / 误伤 / 净收益 | {recovered} / {harmed} / {recovered - harmed:+d} |")
    add(f"| 第 1 次自检未通过、触发修复 | {_pct(len(triggered), n)} |")
    add(f"| 实际修复轮数（总计） | {rounds} |")
    add(f"| 自检通过但答案错（Verifier 漏报） | {len(miss)} |")
    add(f"| token（总计 / 其中修复阶段） | {tok:,} / {rep_tok:,} |")
    if summary and summary.get("llm_usage"):
        u = summary["llm_usage"]
        add(f"| LLM 调用（实际 / 缓存重放） | {u.get('calls', 0)} / {u.get('cached_calls', 0)} |")
    add("")

    # 2. by attempt
    add("## 2. 逐次尝试的指标变化\n")
    add("第 k 列表示“如果最多允许 k 次尝试”时的题数；已经停止的题（自检通过或达到最大修复次数）沿用最后一次的状态。\n")
    prog = attempt_progress(cases, max_attempts)
    add("| 指标 | " + " | ".join(f"第 {k} 次" for k in range(1, max_attempts + 1)) + " |")
    add("|---|" + "---|" * max_attempts)
    for m in METRICS:
        add(f"| {m} | " + " | ".join(str(r["题数"]) for r in prog if r["指标"] == m) + " |")
    add("")

    # 3. per case
    add("## 3. 逐题结局\n")
    add("| 题目 | 结局 | 各次尝试 | 诊断（每轮） | 修复（每轮） | 最终答案 | 漏报 |\n|---|---|---|---|---|---|---|")
    for c in cases:
        if not (c["judged"] and c["executions"]):
            add(f"| `{c['case_id'].split(':')[-1]}` | {c['outcome']} | — | — | — | — | — |")
            continue
        chain = " → ".join(attempt_status(c, i) for i in range(len(c["executions"])))
        diag = " → ".join(f"{TYPE_CN.get(d['failure_type'], d['failure_type'])}（{'规则' if d['source'] == 'rule' else d['source']}）"
                          for d in c["diagnoses"]) or "—"
        rep = " → ".join(f"{r['skill']}（{'LLM' if r.get('used_llm') else '确定性'}）" for r in c["repairs"]) or "—"
        fin = c["final"]
        self_ok = bool(fin) and fin["verifier_decision"] == "PASS"
        fin_s = f"第 {fin['final_attempt']} 次 · {'自检通过' if self_ok else '自检未通过'}" if fin else "—"
        missed = "是" if self_ok and not c["judged"]["final_correct"] else ""
        add(f"| `{c['case_id'].split(':')[-1]}` | {c['outcome']} | {chain} | {diag} | {rep} | {fin_s} | {missed} |")
    add("")

    # 4. components
    add("## 4. Loop 各环节表现\n")
    sig = Counter(s for c in done for s in c["verifies"][0]["signals"])
    add(f"- **自检（Verifier）**：第 1 次尝试中 {len(triggered)} 题自检未通过、进入修复"
        + (f"，信号：{'、'.join(f'{k} ×{v}' for k, v in sig.items())}" if sig else "") + "。"
        + (f"另有 {len(miss)} 题自检通过但 Gold 判错（漏报）：{'、'.join(c['case_id'].split(':')[-1] for c in miss)}"
           "（能执行但答案错，SelfVerifier 发现不了）。" if miss else ""))
    diags = [d for c in done for d in c["diagnoses"]]
    if diags:
        dist = Counter(TYPE_CN.get(d["failure_type"], d["failure_type"]) for d in diags)
        src = Counter(d["source"] for d in diags)
        add(f"- **诊断**：{len(diags)} 次，类型分布 {dict(dist)}；来源 规则 {src.get('rule', 0)} / LLM {src.get('llm', 0)}"
            f" / 兜底 {src.get('fallback', 0)}。")
    per = defaultdict(lambda: {"n": 0, "exec": 0, "right": 0, "det": 0})
    for c in done:
        for r in c["repairs"]:
            i = r["attempt_id"] - 1                     # the attempt this repair produced
            if i >= len(c["executions"]):
                continue
            s = per[r["skill"]]
            s["n"] += 1
            s["det"] += not r.get("used_llm")
            s["exec"] += c["executions"][i]["execution_status"] == "SUCCESS"
            s["right"] += bool(c["judged"]["attempt_correct"][i])
    if per:
        add("- **修复技能**（按修复轮次计）：\n")
        add("| 技能 | 轮次 | 其中确定性修复 | 修复后可执行 | 修复后答对 |\n|---|---|---|---|---|")
        for k, s in per.items():
            add(f"| {k} | {s['n']} | {s['det']} | {s['exec']} | {s['right']} |")
    add("")

    # 5. time & cost
    add("## 5. 时间和成本花在哪里\n")
    gen_ms = sum((c["generate"] or {}).get("latency_ms", 0) for c in done)
    rep_ms = sum(r.get("latency_ms", 0) for c in done for r in c["repairs"])
    exe_ms = sum(e.get("exec_latency_ms") or 0 for c in done for e in c["executions"])
    total = gen_ms + rep_ms + exe_ms
    if total:
        add("| 阶段 | 耗时 | 占比 |\n|---|---|---|")
        for name, ms in (("生成 SQL（LLM）", gen_ms), ("修复（LLM）", rep_ms), ("执行 SQL（Warehouse）", exe_ms)):
            add(f"| {name} | {ms / 1000:.1f} 秒 | {100 * ms / total:.0f}% |")
        cached = sum(bool((c["generate"] or {}).get("cached")) for c in done) + \
            sum(bool((r.get("details") or {}).get("cached")) for c in done for r in c["repairs"])
        if cached:
            add(f"\n其中 {cached} 次 LLM 调用是缓存重放（同样的 prompt 之前调用过，直接复用回答，不产生费用）；"
                "表中 LLM 耗时是当初实际调用时记录的耗时，本次实际等待更短。\n")

    # 6. findings
    add("## 6. 发现与建议\n")
    findings = []
    if recovered:
        findings.append(f"**Loop 修好了 {recovered} 题**：{'、'.join(c['case_id'].split(':')[-1] for c in done if not c['judged']['attempt_correct'][0] and c['judged']['final_correct'])}。")
    if harmed:
        findings.append(f"**有 {harmed} 题被修复改坏**（首次对、最终错），需要检查 Verifier 是否误报。")
    same_err = [c for c in done if any(
        c["executions"][i]["execution_status"] != "SUCCESS"
        and error_class(c["executions"][i].get("execution_error"))
        and error_class(c["executions"][i].get("execution_error")) == error_class(c["executions"][i + 1].get("execution_error"))
        for i in range(len(c["executions"]) - 1))]
    if same_err:
        findings.append(f"**{len(same_err)} 题修复后出现同样的报错**（{'、'.join(c['case_id'].split(':')[-1] for c in same_err)}）："
                        "诊断和定向指令已给出，但模型没有按指令修改。可考虑把这类问题做成确定性修复，或换更强的模型。")
    if max_attempts > 2:
        late = [c for c in done if len(c["executions"]) > 2 and c["executions"][1]["execution_status"] != "SUCCESS"
                and any(e["execution_status"] == "SUCCESS" for e in c["executions"][2:])]
        more = [r["题数"] for r in prog if r["指标"] == "可执行"]
        findings.append(f"**多轮修复的效果**：可执行题数随尝试次数变化 {' → '.join(map(str, more))}；"
                        + (f"{len(late)} 题在第 3 次及以后才修到可执行（{'、'.join(c['case_id'].split(':')[-1] for c in late)}），"
                           "说明额外的修复轮次有价值。" if late else "第 2 次之后的修复轮次没有带来新的可执行题，增加轮次主要增加成本。"))
    surf = [c for c in done if c["outcome"] == "报错 → 可执行（仍错）"]
    if surf:
        findings.append(f"**{len(surf)} 题修复后能执行但答案仍错**：修复解决了表层的执行错误，没有解决语义问题；"
                        "这类结果会自检通过，Loop 不再修复。")
    if miss:
        findings.append("**Verifier 是当前短板**：能执行但答案错的结果会自检通过，Loop 就不再修复。建议补强 Verifier，例如检查题干里的过滤条件、"
                        "输出列、分组要求是否都体现在 SQL 里，以及过滤值在库里是否存在。")
    det_ok = [c for c in done if any(not r.get("used_llm") and r["attempt_id"] - 1 < len(c["executions"])
                                     and c["executions"][r["attempt_id"] - 1]["execution_status"] == "SUCCESS"
                                     for r in c["repairs"])]
    if det_ok:
        findings.append(f"**确定性修复有效**：{len(det_ok)} 题没有调用 LLM 就修到可执行，成本为 0。")
    if request.get("model") == "glm-4-flash" and not recovered:
        findings.append("当前模型 glm-4-flash 的修复能力有限（干预实验：给全部 Gold 提示也修不好 0/30）；"
                        "deepseek-flash 在同样条件下能修好 7/27，可对比同一批题的表现。")
    if not findings:
        findings.append("本次运行没有明显问题。")
    L += [f"{i}. {f}" for i, f in enumerate(findings, 1)]
    add("")

    # 7. details
    add("## 7. 逐题详情\n")
    for c in cases:
        add(f"### `{c['case_id'].split(':')[-1]}` · {c['outcome']}\n")
        add(f"> {c['question']}\n")
        for k, r in enumerate(c["repairs"]):
            o = c["observes"][k] if k < len(c["observes"]) else None
            d = c["diagnoses"][k] if k < len(c["diagnoses"]) else None
            add(f"**第 {k + 1} 轮修复**（第 {r['attempt_id'] - 1} 次 → 第 {r['attempt_id']} 次尝试）\n")
            if o:
                parts = [f"错误类别 `{o['error_class']}`" if o.get("error_class") else None,
                         f"找不到的列 `{(o.get('unresolved_qualifier') + '.') if o.get('unresolved_qualifier') else ''}{o['unresolved_column']}`" if o.get("unresolved_column") else None,
                         f"引擎候选 {', '.join(o['suggestions'][:5])}" if o.get("suggestions") else None,
                         f"缺失的表 `{o['missing_table']}`" if o.get("missing_table") else None]
                add("- **观察**：" + ("；".join(p for p in parts if p) or f"执行状态 {o['execution_status']}，无报错信号"))
            if d:
                add(f"- **诊断**：{TYPE_CN.get(d['failure_type'], d['failure_type'])}（置信度 {d['confidence']:.2f}）— {d['reason']}")
            det = r.get("details") or {}
            add(f"- **修复**：{r['skill']} — {r.get('action') or ''}")
            if det.get("deterministic_changes"):
                add(f"  - 确定性修改：{'；'.join(det['deterministic_changes'])}")
            if det.get("unresolved"):
                add(f"  - 规则无法决定、交给 LLM：{'；'.join(det['unresolved'])}")
            if det.get("tables_added"):
                add(f"  - 补充的表：{', '.join(det['tables_added'])}")
            if det.get("join_candidates"):
                add(f"  - 提供的关联键：{'; '.join(det['join_candidates'][:6])}")
            if det.get("instruction"):
                add(f"  - 给 LLM 的指令：{det['instruction'][:500]}")
            diff = "\n".join(difflib.unified_diff((r.get("before_sql") or "").splitlines(), (r.get("sql") or "").splitlines(),
                                                  "修复前", "修复后", lineterm="", n=1))
            if diff:
                add(f"\n```diff\n{diff}\n```")
            i = r["attempt_id"] - 1
            if i < len(c["executions"]):
                e = c["executions"][i]
                add(f"- **修复后执行**：{e['execution_status']}"
                    + (f" `{error_class(e.get('execution_error'))}`" if error_class(e.get("execution_error")) else "")
                    + (f"，{e.get('result_row_count')} 行" if e["execution_status"] == "SUCCESS" else ""))
            elif c["stop"]:
                add(f"- **提前结束**：修复后的 SQL 与第 {c['stop']['repeated_attempt']} 次尝试相同，不再执行")
            add("")
        if c["executions"] and c["final"]:
            last = c["executions"][c["final"]["final_attempt"] - 1]
            if last["execution_status"] == "SUCCESS":
                add(preview_block(last.get("result_preview"), last.get("result_row_count")))
            elif last.get("execution_error"):
                err = last["execution_error"]
                add(f"- **最终答案的报错**：`{err[:400]}{' …（已截断）' if len(err) > 400 else ''}`")
        add("")
    return "\n".join(L)
