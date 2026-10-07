"""Loop Debug Console (Phase 9).

    streamlit run app/dashboard.py

Reads the Delta tables the pipeline publishes. Phase 7 (evaluation-set arms)
and Phase 8 (ablations) have reserved sections that fill in automatically once
those runs are published with scripts/publish_run.py.
"""
from __future__ import annotations

import difflib
import html
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import data  # noqa: E402

st.set_page_config(page_title="Loop Debug Console", page_icon="🔁", layout="wide")

ACCENT, OK, WARN, BAD, MUTED = "#1f5f7a", "#2f7d4f", "#b26a14", "#b23a33", "#6b7a80"
TYPE_CN = {"TABLE_RETRIEVAL_FAILURE": "选表", "COLUMN_MAPPING_FAILURE": "列映射", "JOIN_KEY_FAILURE": "关联键",
           "DOMAIN_KNOWLEDGE_FAILURE": "领域知识", "QUERY_DECOMPOSITION_FAILURE": "查询拆解",
           "EXECUTION_FAILURE": "执行错误", "UNKNOWN": "未知"}
ARM_CN = {"targeted-self": "Targeted Loop · Self", "generic-self": "Generic Retry · Self",
          "targeted-oracle": "Targeted Loop · Oracle（上界）", "generic-oracle": "Generic Retry · Oracle（上界）"}
PLANNED_P7 = ["targeted-self", "generic-self", "targeted-oracle", "generic-oracle"]
PLANNED_P8 = [  # label, match(meta) -> bool
    ("Full Loop（Targeted · Self）", lambda r: r.strategy == "targeted" and r.policy == "targeted" and not r.disabled),
    ("No Repair Policy（全部走 RepairSQL）", lambda r: r.strategy == "targeted" and r.policy == "generic"),
    ("No SchemaSearch", lambda r: r.disabled == "SchemaSearch"),
    ("No RetrieveAgain", lambda r: r.disabled == "RetrieveAgain"),
    ("No FindJoinPath", lambda r: r.disabled == "FindJoinPath"),
    ("No ReplanQuery", lambda r: r.disabled == "ReplanQuery"),
    ("No Diagnosis（Generic Retry）", lambda r: r.strategy == "generic"),
]

st.markdown(f"""<style>
.block-container{{padding-top:1.6rem; max-width:1280px}}
.pill{{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;margin-right:6px}}
.ok{{background:{OK}22;color:{OK}}} .warn{{background:{WARN}22;color:{WARN}}} .bad{{background:{BAD}22;color:{BAD}}}
.acc{{background:{ACCENT}22;color:{ACCENT}}} .plain{{background:#8883;color:inherit}}
.phase{{display:inline-block;padding:4px 10px;margin:0 6px 6px 0;border-radius:6px;border:1px solid #8884;font-size:13px}}
.phase.done{{border-color:{OK};box-shadow:inset 0 3px 0 {OK}}} .phase.part{{border-color:{WARN};box-shadow:inset 0 3px 0 {WARN}}}
.diff{{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;line-height:1.5;overflow-x:auto;border-radius:6px;
       border:1px solid #8883;padding:6px 0;max-height:520px;overflow-y:auto}}
.diff div{{white-space:pre;padding:0 10px}} .diff .a{{background:{OK}22}} .diff .d{{background:{BAD}22}}
.reserved{{border:1px dashed #8886;border-radius:8px;padding:14px 16px;opacity:.9}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data

@st.cache_resource(show_spinner="连接 Databricks SQL Warehouse…")
def conn():
    return data.connect()


@st.cache_data(ttl=300, show_spinner=False)
def q_runs():
    return data.runs(conn())


@st.cache_data(ttl=300, show_spinner=False)
def q_traces(run_id):
    return data.traces(conn(), run_id)


@st.cache_data(ttl=300, show_spinner=False)
def q_evals(run_id):
    return data.evaluations(conn(), run_id)


@st.cache_data(ttl=300, show_spinner=False)
def q_labels():
    return data.failure_labels(conn())


@st.cache_data(ttl=300, show_spinner=False)
def q_diag_eval():
    return data.diagnosis_eval(conn())


@st.cache_data(ttl=600, show_spinner=False)
def q_bench():
    return data.benchmark_summary(conn())


def nz(x):
    """Database NULLs arrive as NaN (truthy!) -> None."""
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else x


def pct(x):
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.1f}%"


def pill(text, kind="plain"):
    return f'<span class="pill {kind}">{html.escape(str(text))}</span>'


def loop_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        s = r.summary
        vc = s.get("verifier_confusion", {})
        rows.append({
            "实验组": ARM_CN.get(r.arm, r.arm), "题数": s.get("cases"),
            "首次准确率": pct(s.get("first_pass_accuracy")), "最终准确率": pct(s.get("final_accuracy")),
            "恢复": s.get("recovered"), "恢复率": pct(s.get("recovery_rate")),
            "误伤": s.get("harmed"), "误伤率": pct(s.get("harm_rate")), "净收益": s.get("net_gain"),
            "可执行 首→终": f"{s.get('executable_first')} → {s.get('executable_final')}",
            "平均尝试": s.get("avg_attempts"), "额外 token": f"{s.get('extra_tokens_total', 0):,}",
            "每净恢复 token": s.get("extra_tokens_per_net_recovery") or "—",
            "Verifier 命中/漏报/误报": f"{vc.get('triggered_and_wrong(hit)', 0)} / {vc.get('passed_but_wrong(miss)', 0)} / "
                                   f"{vc.get('triggered_but_right(false_alarm)', 0)}",
            "run_id": r.run_id,
        })
    return pd.DataFrame(rows)


def latest(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("created_at").groupby("arm", as_index=False).tail(1)


# ------------------------------------------------------------------ header

try:
    R = q_runs()
except Exception as e:  # connection / auth problems
    st.error(f"无法读取 Delta 表：{e}\n\n本地运行请先执行 `databricks auth login`；在 Databricks Apps 中请配置 SQL warehouse 资源。")
    st.stop()

all_loops = R[R["mode"] == "loop"]  # incl. small runs started from the Run page (case trace only)
loops = all_loops[all_loops["source"] != "console"]
main_loops = loops[(loops["policy"] == "targeted") & (loops["disabled"] == "")]
eval_loops = main_loops[main_loops["split"] == "eval"]
ablations = loops[(loops["split"] == "eval") & ((loops["policy"] != "targeted") | (loops["disabled"] != ""))]
base_eval = R[(R["mode"] == "baseline") & (R["split"] == "eval")].tail(1)

st.markdown("##### BEAVER dw · Databricks · Loop Engineering")
st.title("Loop Debug Console")
phases = [("P0 数据准备", "done"), ("P1 Baseline", "done"), ("P2 Trace", "done"), ("P3 失败分类", "done"),
          ("P4 诊断", "done"), ("P5 修复技能", "done"), ("P6 Loop", "done"),
          ("P7 对照实验", "done" if len(eval_loops) else ""), ("P8 消融", "done" if len(ablations) else ""),
          ("P9 Console", "part")]
st.markdown("".join(f'<span class="phase {c}">{html.escape(n)}</span>' for n, c in phases), unsafe_allow_html=True)
# the model of the loop experiments, not of the latest run (model probes also publish runs)
loop_models = main_loops.sort_values("created_at")["model"].dropna()
model = loop_models.iloc[-1] if len(loop_models) else (R["model"].dropna().iloc[-1] if R["model"].notna().any() else "—")
probed = sorted(set(R["model"].dropna()) - {model})
st.caption(f"Loop 实验模型 `{model}`" + (f" · 另有探测模型 {', '.join(f'`{m}`' for m in probed)}（见总览·模型对比）"
                                         if probed else "")
           + f" · 数据来自 Unity Catalog `{data.CATALOG}` · 缓存 5 分钟，右上角菜单 Rerun 可刷新")

tab_over, tab_exp, tab_abl, tab_trace, tab_diag = st.tabs(
    ["总览", "对照实验（P6 / P7）", "消融实验（P8）", "逐题追踪", "失败与诊断"])

# ------------------------------------------------------------------ overview

with tab_over:
    b = q_bench()
    g = b["gold"]
    prim = int(g.loc[g["e"] == "PRIMARY", "n"].sum())
    orig = int(g.loc[(g["e"] == "PRIMARY") & (g["s"] == "databricks_original_gold_sql"), "n"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("评测集可用题（PRIMARY）", f"{prim} / {b['cases']}", help=f"原始 Gold SQL {orig}，规则改写 {prim - orig}")
    rep = b["replication"]
    c2.metric("复制到 Databricks 的表", f"{int(rep['tables'])}", help=f"{int(rep['rows']):,} 行，逐列核对一致 {int(rep['ok'])} 张")
    if len(base_eval):
        s = base_eval.iloc[0]
        c3.metric("Baseline 首次准确率（评测集）", f"{int(s['correct'])} / {int(s['cases'])}",
                  help=f"可执行率 {pct(s['executable_rate'])}")
    main = latest(main_loops[(main_loops["arm"] == "targeted-self")])
    if len(main):
        s = main.iloc[-1]["summary"]
        c4.metric(f"Targeted Loop 可执行（{main.iloc[-1]['split']}）", f"{s['executable_first']} → {s['executable_final']}",
                  help=f"恢复 {s['recovered']} 题，净收益 {s['net_gain']}")

    st.subheader("目前的判断")
    st.markdown(
        "- **流程已跑通**：生成 → 验证 → 诊断 → 修复 → 再验证，四个实验组和上界分析都能自动运行和发布。\n"
        "- **瓶颈在模型能力**：干预实验中，5 类 Gold 提示全部给出后，开发集仍是 0/30；各组恢复都是 0（决策 D4）。\n"
        "- **工程层面的差异**：相近的 token 成本下，Targeted Loop 让更多 SQL 变得可执行（开发集 9 vs 6）。\n"
        "- **换模型探测**：deepseek-flash 给全部 Gold 提示能修好 7/27（glm-4-flash 为 0/30），说明换用更强模型后 Loop 有发挥空间；"
        "失败形态随之变为“能执行但答错”，Verifier 成为主要短板。目前仍用 glm-4-flash 梳理流程（决策 D6）。")

    st.subheader("模型对比（开发集探测）")
    st.caption("同一套配置（prompt baseline-v2、BM25 k=20、3 个 few-shot）只换模型。"
               "“全部 Gold 提示后修好”是离线上界分析：生成时给了 Gold 提示，Loop 运行时看不到这些信息。")
    devb = R[(R["mode"] == "dev")].sort_values("created_at")
    interv = R[(R["mode"] == "intervention")].sort_values("created_at")
    mrows = []
    for m in sorted(set(devb["model"].dropna()) | set(interv["model"].dropna())):
        b = devb[devb["model"] == m].tail(1)
        iv = interv[interv["model"] == m].tail(1)
        bs = b.iloc[0]["summary"] if len(b) else {}
        ex = (bs.get("execution_status") or {}).get("SUCCESS")
        mrows.append({
            "模型": m, "Loop 实验使用": "是" if m == model else "探测",
            "首次答对": f"{int(b.iloc[0]['correct'])} / {int(b.iloc[0]['cases'])}" if len(b) else "—",
            "可执行": f"{ex} / {bs.get('cases')}" if ex is not None else "—",
            "全部 Gold 提示后修好": f"{int(iv.iloc[0]['correct'])} / {int(iv.iloc[0]['cases'])}" if len(iv) else "—",
            "平均 token/题": f"{bs['tokens_mean']:,}" if bs.get("tokens_mean") else "—",
            "LLM 响应中位耗时": f"{bs['llm_latency_ms_median'] / 1000:.1f} 秒" if bs.get("llm_latency_ms_median") else "—",
            "基线 run_id": b.iloc[0]["run_id"] if len(b) else "—",
        })
    if mrows:
        st.dataframe(pd.DataFrame(mrows), hide_index=True, width="stretch")
    else:
        st.info("还没有开发集基线运行。")

    st.subheader("所有运行")
    view = R[["created_at", "run_id", "mode", "split", "arm", "model", "prompt_version", "cases", "correct",
              "first_pass_accuracy", "tokens_total", "mlflow_run_id"]].sort_values("created_at", ascending=False)
    st.dataframe(view, hide_index=True, width="stretch")

# ------------------------------------------------------------------ experiments

with tab_exp:
    st.subheader("Phase 6 · 开发集（30 题）")
    st.caption("所有实验组的第 1 次尝试相同；差别只在第 2 次尝试。Oracle 组用 Gold 判断对错，只作为上界。")
    dev = latest(main_loops[main_loops["split"] == "dev"])
    if len(dev):
        dev = dev.set_index("arm").reindex([a for a in PLANNED_P7 if a in set(dev["arm"])]).reset_index()
        st.dataframe(loop_table(dev), hide_index=True, width="stretch")
        long = pd.DataFrame([{"实验组": ARM_CN.get(r.arm, r.arm), "阶段": k, "题数": r.summary[f"executable_{v}"]}
                             for r in dev.itertuples() for k, v in (("第 1 次", "first"), ("最终", "final"))])
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("题数:Q", title="可执行的题数（开发集 30 题）"),
            y=alt.Y("实验组:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
            yOffset=alt.YOffset("阶段:N", sort=["第 1 次", "最终"]),
            color=alt.Color("阶段:N", scale=alt.Scale(domain=["第 1 次", "最终"], range=[MUTED, ACCENT]), title=None),
            tooltip=["实验组", "阶段", "题数"]).properties(height=90 * len(dev))
        st.altair_chart(chart, width="stretch")
    else:
        st.info("还没有开发集上的 Loop 运行。")

    st.subheader("Phase 7 · 评测集（89 题）")
    ev = latest(eval_loops)
    if len(ev):
        ev = ev.set_index("arm").reindex([a for a in PLANNED_P7 if a in set(ev["arm"])]).reset_index()
        st.dataframe(loop_table(ev), hide_index=True, width="stretch")
        missing = [ARM_CN[a] for a in PLANNED_P7 if a not in set(ev["arm"])]
        if missing:
            st.caption("尚未运行：" + "、".join(missing))
    else:
        st.markdown(
            '<div class="reserved"><b>预留：Phase 7 尚未运行。</b><br>'
            "Loop 配置冻结后，在 89 道评测题上各跑一次以下实验组，结果发布后会自动出现在这里：<br>"
            + "".join(pill(ARM_CN[a], "acc") for a in PLANNED_P7) +
            "<br><br>将展示：首次 / 最终准确率、恢复率、<b>误伤率</b>、<b>净收益</b>、Verifier 混淆矩阵、每净恢复一题的额外 token，"
            "以及 Oracle 与 Self 的差距（瓶颈在“发现不了错”还是“修不好”）。<br>"
            "运行命令：<code>python scripts/phase6.py --strategy targeted --verifier self --split eval --eval</code>"
            "，然后 <code>python scripts/publish_run.py runs/phase6/&lt;run_id&gt; --experiment-id targeted_loop</code></div>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ ablations

with tab_abl:
    st.subheader("Phase 8 · 消融实验（评测集）")
    st.caption("逐个去掉 Loop 的部件，看净收益下降多少，从而判断哪个部件真正起作用。")
    rows = []
    for label, match in PLANNED_P8:
        hit = [r for r in pd.concat([ablations, eval_loops]).itertuples() if r.verifier == "self" and match(r)]
        if hit:
            s = sorted(hit, key=lambda r: r.created_at)[-1].summary
            rows.append({"消融组": label, "状态": "已运行", "最终准确率": pct(s.get("final_accuracy")),
                         "恢复": s.get("recovered"), "误伤": s.get("harmed"), "净收益": s.get("net_gain"),
                         "可执行 首→终": f"{s.get('executable_first')} → {s.get('executable_final')}",
                         "额外 token": f"{s.get('extra_tokens_total', 0):,}"})
        else:
            rows.append({"消融组": label, "状态": "未运行", "最终准确率": "—", "恢复": "—", "误伤": "—",
                         "净收益": "—", "可执行 首→终": "—", "额外 token": "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not len(ablations):
        st.markdown(
            '<div class="reserved"><b>预留：Phase 8 尚未运行。</b><br>'
            "Policy 与技能开关已经支持消融，例如 <code>--policy generic</code>（所有失败走 RepairSQL）、"
            "<code>--disable SchemaSearch</code>。结果发布后，上表的“状态”会自动变为“已运行”，"
            "并按净收益相对 Full Loop 的下降幅度排序。</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ case trace

with tab_trace:
    if not len(all_loops):
        st.info("还没有 Loop 运行。")
    else:
        opts = all_loops.sort_values("created_at", ascending=False)
        labels = {r.run_id: (f"[运行页] {ARM_CN.get(r.arm.removeprefix('console-'), r.arm)} · {r.model} · {r.split}"
                             f" · {r.run_id[-22:]}" if r.source == "console" else
                             f"{ARM_CN.get(r.arm, r.arm)} · {r.split} · {r.run_id[-22:]}") for r in opts.itertuples()}
        default = next((r.run_id for r in opts.itertuples() if r.arm == "targeted-self"), opts.iloc[0]["run_id"])
        run_id = st.selectbox("运行", list(labels), index=list(labels).index(default), format_func=labels.get)
        T, E = q_traces(run_id), q_evals(run_id)
        corr = {(r.case_id, int(r.attempt_id)): bool(r.correct) for r in E.itertuples()}

        def outcome(g: pd.DataFrame) -> tuple[str, str]:
            final = g[g["final_status"] == "FINAL"].iloc[0]
            first_ok, final_ok = corr.get((final.case_id, 1), False), corr.get((final.case_id, int(final.attempt_id)), False)
            if first_ok and final_ok:
                return "首次即对", "ok"
            if final_ok:
                return "已恢复", "ok"
            if first_ok:
                return "被误伤", "bad"
            s0, s1 = g.iloc[0]["execution_status"], final["execution_status"]
            if s0 != "SUCCESS" and s1 == "SUCCESS":
                return "报错 → 可执行（仍错）", "warn"
            return ("仍报错" if s1 != "SUCCESS" else "可执行但错"), "bad" if s1 != "SUCCESS" else "warn"

        groups = {cid: g.sort_values("attempt_id") for cid, g in T.groupby("case_id")}
        outs = {cid: outcome(g) for cid, g in groups.items()}
        f = st.radio("筛选", ["全部", "已恢复", "报错 → 可执行（仍错）", "仍报错", "可执行但错", "被误伤"], horizontal=True)
        ids = [c for c in groups if f == "全部" or outs[c][0] == f]
        if not ids:
            st.caption("没有符合条件的题。")
        else:
            cid = st.selectbox("题目", ids, format_func=lambda c: f"{c} · {outs[c][0]} · {len(groups[c])} 次尝试")
            g = groups[cid]
            first = g.iloc[0]
            st.markdown(pill(outs[cid][0], outs[cid][1]) + pill(f"{len(g)} 次尝试", "plain"), unsafe_allow_html=True)
            st.markdown(f"**问题**：{first['question']}")
            with st.expander("检索到的表"):
                st.write(", ".join(json.loads(first["retrieved_tables"])) if first["retrieved_tables"] else "—")
            for i, a in enumerate(g.itertuples()):
                ok = corr.get((cid, int(a.attempt_id)))
                sql, err, ftype = nz(a.generated_sql), nz(a.execution_error), nz(a.failure_type)
                signals, skill = nz(a.verifier_signals), nz(a.repair_skill)
                head = (pill(f"第 {a.attempt_id} 次", "acc") + pill(a.execution_status, "ok" if a.execution_status == "SUCCESS" else "bad")
                        + pill(("自检通过" if a.verifier_decision == "PASS" else f"自检未通过 {signals or ''}")
                               if a.verifier_mode == "self" else f"Oracle {a.verifier_decision}",
                               "ok" if a.verifier_decision == "PASS" else "warn")
                        + pill("最终答案" if a.final_status == "FINAL" else "被替代", "acc" if a.final_status == "FINAL" else "plain")
                        + pill(f"Gold 判分：{'对' if ok else '错'}（Loop 不可见）", "ok" if ok else "plain"))
                st.markdown(head, unsafe_allow_html=True)
                st.code(sql or "(没有生成 SQL)", language="sql")
                if err:
                    st.error(err[:1500])
                if ftype:
                    conf = nz(a.diagnosis_confidence)
                    st.markdown(f"**诊断**：{pill(TYPE_CN.get(ftype, ftype), 'acc')}"
                                f" 置信度 {conf if conf is None else f'{conf:.2f}'} — {html.escape(nz(a.diagnosis_reason) or '')}",
                                unsafe_allow_html=True)
                if i + 1 < len(g):
                    nxt = g.iloc[i + 1]
                    st.markdown(f"**修复**：{pill(skill or '—', 'acc')} → 生成第 {nxt['attempt_id']} 次尝试",
                                unsafe_allow_html=True)
                    diff = difflib.ndiff((sql or "").splitlines(), (nz(nxt["generated_sql"]) or "").splitlines())
                    lines = [f'<div class="{"a" if d[0] == "+" else "d" if d[0] == "-" else ""}">{html.escape(d)}</div>'
                             for d in diff if not d.startswith("?")]
                    with st.expander(f"第 {a.attempt_id} 次 → 第 {nxt['attempt_id']} 次的 SQL 变化", expanded=True):
                        st.markdown(f'<div class="diff">{"".join(lines)}</div>', unsafe_allow_html=True)
                st.divider()

# ------------------------------------------------------------------ failures & diagnosis

with tab_diag:
    L, DE = q_labels(), q_diag_eval()
    runs_with = sorted(set(L["run_id"]), reverse=True)
    if not runs_with:
        st.info("还没有失败标签。")
    else:
        rid = st.selectbox("运行（Phase 3 标签 / Phase 4 诊断）", runs_with,
                           format_func=lambda r: f"{r} · {R.set_index('run_id').loc[r, 'split'] if r in set(R['run_id']) else ''}")
        l = L[L["run_id"] == rid]
        d = DE[DE["run_id"] == rid]
        c1, c2, c3 = st.columns(3)
        c1.metric("失败题数", len(l))
        if len(d):
            c2.metric("诊断准确率（严格）", pct(d["diagnosis_correct"].mean()), help="诊断结果 = 标注器主因")
            c3.metric("诊断准确率（宽松）", pct(d["diagnosis_correct_lenient"].mean()), help="诊断结果在所有未通过的检查里")
        prim = l["actual_failure_type"].value_counts().rename_axis("t").reset_index(name="n")
        prim["来源"] = "标注器主因"
        parts = [prim]
        if len(d):
            pr = d["predicted_failure_type"].value_counts().rename_axis("t").reset_index(name="n")
            pr["来源"] = "运行时诊断"
            parts.append(pr)
        dist = pd.concat(parts)
        dist["类型"] = dist["t"].map(lambda x: TYPE_CN.get(x, x))
        st.altair_chart(alt.Chart(dist).mark_bar().encode(
            x=alt.X("n:Q", title="题数"), y=alt.Y("类型:N", title=None), yOffset="来源:N",
            color=alt.Color("来源:N", scale=alt.Scale(domain=["标注器主因", "运行时诊断"], range=[ACCENT, WARN]), title=None),
            tooltip=["类型", "来源", "n"]).properties(height=280), width="stretch")
        if len(d):
            st.markdown("**诊断来源的准确率**")
            by = d.groupby("diagnosis_source").agg(题数=("case_id", "count"), 严格=("diagnosis_correct", "mean"),
                                                   宽松=("diagnosis_correct_lenient", "mean")).reset_index()
            by["严格"], by["宽松"] = by["严格"].map(pct), by["宽松"].map(pct)
            st.dataframe(by.rename(columns={"diagnosis_source": "来源"}), hide_index=True)
            st.markdown("**标注器主因 → 运行时诊断**")
            cm = pd.crosstab(d["actual_failure_type"].map(TYPE_CN), d["predicted_failure_type"].map(TYPE_CN))
            st.dataframe(cm, width="stretch")
        st.caption("标注器使用 Gold 标注，只用于评测；主因按上下游顺序取第一项，是约定而非因果证明。")
