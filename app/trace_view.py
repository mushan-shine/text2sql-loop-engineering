"""Step-by-step trace timeline of one question, used by the "Run loop" page.

``render_case(events, active)`` draws one question's loop events (retrieve, generate, execute, verify,
observe, diagnose, route, repair, final, judged) as a Streamlit step timeline.
"""
from __future__ import annotations

import difflib
import json
import re

import pandas as pd
import streamlit as st

TYPE_CN = {"TABLE_RETRIEVAL_FAILURE": "选表", "COLUMN_MAPPING_FAILURE": "列映射", "JOIN_KEY_FAILURE": "关联键",
           "DOMAIN_KNOWLEDGE_FAILURE": "领域知识", "QUERY_DECOMPOSITION_FAILURE": "查询拆解",
           "EXECUTION_FAILURE": "执行错误", "UNKNOWN": "未知"}


def fmt_ms(ms) -> str:
    return f"{(ms or 0) / 1000:.1f} 秒"


def render_case(events: list[dict], active: bool) -> None:
    for e in events:
        s, n, t = e["step"], e.get("attempt_id"), f" · +{e['t']:.0f}s"
        if s == "retrieve":
            box = st.status(f"检索 · BM25 从 schema 中选出 {len(e['tables'])} 张候选表{t}", type="step", state="complete")
            box.caption("按表名 / 列名 / 样例值的词频相关度打分（权重 3 / 2 / 1），取前 k 张，作为生成 SQL 时给模型看的 schema。")
            box.dataframe(pd.DataFrame({"表": e["tables"], "BM25 分数": e.get("scores") or [None] * len(e["tables"])}),
                          hide_index=True, height=220)
        elif s == "generate":
            cached = " · 缓存重放（耗时为当初记录）" if e.get("cached") else ""
            box = st.status(f"生成 SQL（第 1 次）· {e['tokens']:,} token · {fmt_ms(e['latency_ms'])}{cached}{t}",
                            type="step", state="complete")
            box.caption(f"prompt 构成：规则 + {e.get('schema_tables', '?')} 张表的 schema + {e.get('few_shot', '?')} 个 "
                        f"few-shot 示例 + 问题；输入 {e.get('input_tokens', 0):,} token，输出 {e.get('output_tokens', 0):,} token。")
            if "dynfs" in str(e.get("prompt_version") or ""):
                box.markdown(f"**相似题示例**（BEAVER id）：{', '.join(e.get('example_ids') or []) or '—'}；"
                             f"示例用到、检索没选出的表已补进 schema：{', '.join(e.get('added_tables') or []) or '无'}")
            if e.get("knowledge_notes"):
                box.markdown("**数仓使用说明**（外层循环从已解题统计的约定，本题相关部分）")
                box.code(e["knowledge_notes"], language="text", wrap_lines=True)
            box.markdown("**生成的 SQL**")
            box.code(e["sql"] or "(没有生成 SQL)", language="sql", wrap_lines=True)
            if e.get("prompt"):
                box.markdown("**完整 prompt**（可滚动）")
                box.code(e["prompt"], language="text", height=220, wrap_lines=True)
        elif s == "execute":
            ok = e["execution_status"] == "SUCCESS"
            cls = re.search(r"\[([A-Z][A-Z0-9_]+)", e.get("execution_error") or "")
            label = (f"执行（第 {n} 次）· 成功，{e['result_row_count']} 行" if ok
                     else f"执行（第 {n} 次）· {e['execution_status']}" + (f" · {cls.group(1)}" if cls else ""))
            took = f" · {fmt_ms(e['exec_latency_ms'])}" if e.get("exec_latency_ms") else ""
            box = st.status(f"{label}{took}{t}", type="step",
                            state="complete" if ok else "error")
            box.caption("在 Databricks SQL Warehouse 上只读执行（单条语句，超 50 万行判为结果过大）。")
            if ok:
                box.markdown("**结果预览（前 5 行）**")
                box.code(e.get("result_preview") or "[]", language="json", wrap_lines=True)
            elif e.get("execution_error"):
                box.markdown("**引擎报错**")
                box.code(e["execution_error"][:1500], language="text", wrap_lines=True)
        elif s == "verify":
            sig = "、".join(e["signals"])
            who = "自检" if e["mode"] == "self" else "Oracle 校验（上界）"
            label = f"{who}（第 {n} 次）· " + ("通过" if e["passed"] else f"未通过：{sig}")
            if not e["passed"] and e.get("last"):
                label += " · 已达最大修复次数，不再修复"
            box = st.status(label + t, type="step", state="complete" if e["passed"] else "error")
            box.caption("自检（SelfVerifier）只看无 Gold 的信号：没有 SQL、执行报错、结果过大、空结果、整列 NULL；"
                        "能执行时再对照题干做结构检查（关联条件恒为真、JOIN 缺条件、缺少分组、四舍五入），"
                        "并由代码核对结果数值是否自洽（平均值在最小 / 最大值之间、方差 = 标准差²、计数为非负整数等，不用大模型推理）。"
                        "“通过”只表示没发现问题，不代表答案正确——Loop 运行时看不到标准答案。"
                        if e["mode"] == "self" else "Oracle 用 Gold 判断对错，只作上界参考。")
            for f in e.get("findings") or []:
                box.error(f"**{f['signal']}**：{f['message']}", icon=":material/rule:")
            for f in e.get("advisories") or []:
                box.info(f"提示（误报率较高，不触发修复）**{f['signal']}**：{f['message']}", icon=":material/lightbulb:")
        elif s == "observe":
            box = st.status(f"观察 · 从执行结果中提取诊断信号{t}", type="step", state="complete")
            box.caption("Observer 只通过白名单读取这次尝试的字段，Gold 相关字段进不来：" + ", ".join(e.get("fields") or []))
            rows = [("执行状态", e.get("execution_status")), ("错误类别", e.get("error_class")),
                    ("找不到的列", ".".join(x for x in (e.get("unresolved_qualifier"), e.get("unresolved_column")) if x)),
                    ("引擎给出的候选列", ", ".join(e.get("suggestions") or [])), ("找不到的表", e.get("missing_table")),
                    ("结果行数", e.get("result_row_count"))]
            present = [(k, str(v)) for k, v in rows if v not in (None, "")]
            if present:
                box.dataframe(pd.DataFrame(present, columns=["信号", "值"]), hide_index=True)
            else:
                box.write("没有可解析的信号。")
        elif s == "diagnose":
            box = st.status(f"诊断 · {TYPE_CN.get(e['failure_type'], e['failure_type'])} · 置信度 "
                            f"{e['confidence']:.2f} · 来源 {'规则' if e['source'] == 'rule' else e['source']}{t}",
                            type="step", state="complete")
            box.markdown(f"**归因**：{e['reason']}")
            if e.get("repair_hints"):
                box.markdown("**传给修复技能的证据**（repair_hints）")
                box.code(json.dumps(e["repair_hints"], ensure_ascii=False, indent=1), language="json", height=200)
        elif s == "route":
            st.status(f"路由 → {e['skill']}" + ("（兜底）" if e["fallback"] else "") + f" · {e['reason']}{t}",
                      type="step", state="complete")
        elif s == "repair":
            det = e.get("details") or {}
            how = "LLM 改写" if e.get("used_llm") else "确定性修复（不调 LLM）"
            if det.get("cached"):
                how += " · 缓存重放"
            box = st.status(f"修复 · {e['skill']} · {how} · 生成第 {n} 次尝试" +
                            (f" · {e['tokens']:,} token · {fmt_ms(e.get('latency_ms'))}" if e.get("tokens") else "") + t,
                            type="step", state="complete")
            if e.get("action"):
                box.caption(f"技能动作：{e['action']}")
            facts = []
            if det.get("deterministic_changes"):
                facts.append("**确定性修改**（按 schema 查表完成）：" + "；".join(f"`{c}`" for c in det["deterministic_changes"]))
            if det.get("unresolved"):
                facts.append("**规则无法决定、交给 LLM**：" + "；".join(det["unresolved"]))
            if det.get("tables_added"):
                facts.append("**补充的表**：" + ", ".join(f"`{x}`" for x in det["tables_added"]))
            if det.get("candidate_columns"):
                facts.append("**候选列**：" + ", ".join(det["candidate_columns"][:10]))
            if det.get("join_candidates"):
                facts.append("**从 schema 推断的关联键**：" + "; ".join(det["join_candidates"][:8]))
            if det.get("schema_tables"):
                facts.append(f"**提供给 LLM 的 schema**：{len(det['schema_tables'])} 张表")
            for f in facts:
                box.markdown(f"- {f}")
            if det.get("instruction"):
                box.markdown("**给 LLM 的定向指令**")
                box.code(det["instruction"], language="text", wrap_lines=True)
            diff = "\n".join(difflib.unified_diff((e.get("before_sql") or "").splitlines(),
                                                  (e.get("sql") or "").splitlines(), "修复前", "修复后", lineterm=""))
            box.markdown("**SQL 变化**")
            box.code(diff or "(没有变化)", language="diff", wrap_lines=True)
            if det.get("prompt"):
                box.markdown("**完整修复 prompt**（可滚动）")
                box.code(det["prompt"], language="text", height=220, wrap_lines=True)
        elif s == "stop":
            st.status(f"提前结束 · 修复后的 SQL 与第 {e['repeated_attempt']} 次尝试相同，再执行结果不会变{t}",
                      type="step", state="error")
        elif s == "final":
            ok = e["verifier_decision"] == "PASS"
            st.status(f"最终答案 · 取第 {e['final_attempt']} 次尝试（共 {e['attempts']} 次）· "
                      f"{'自检通过' if ok else '自检未通过'} · {e['execution_status']}", type="step",
                      state="complete" if ok else "error")
    if active:
        st.status(":shimmer[等待下一步…]", type="step", state="running")
    judged = [e for e in events if e["step"] == "judged"]
    final = next((e for e in events if e["step"] == "final"), None)
    if judged:
        j = judged[0]
        marks = " → ".join("对" if c else "错" for c in j["attempt_correct"])
        msg = (f"Gold 判分（Loop 不可见）：各次尝试 {marks}；最终 {'对' if j['final_correct'] else '错'} · "
               f"{j['tokens']:,} token · {fmt_ms(j['latency_ms'])}")
        if j["final_correct"]:
            st.success(msg, icon=":material/grading:")
        elif final and final["verifier_decision"] == "PASS":
            st.warning(msg + "\n\n**自检通过，但答案是错的（Verifier 漏报）**：SQL 能执行、结果非空，自检找不到明显问题，"
                       "所以 Loop 停止了修复。减少这类情况要靠补强 Verifier。", icon=":material/report:")
        else:
            st.error(msg, icon=":material/grading:")
