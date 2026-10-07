"""Run the loop from the browser and watch its trace grow step by step."""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app import data, report, runner  # noqa: E402

TYPE_CN = {"TABLE_RETRIEVAL_FAILURE": "选表", "COLUMN_MAPPING_FAILURE": "列映射", "JOIN_KEY_FAILURE": "关联键",
           "DOMAIN_KNOWLEDGE_FAILURE": "领域知识", "QUERY_DECOMPOSITION_FAILURE": "查询拆解",
           "EXECUTION_FAILURE": "执行错误", "UNKNOWN": "未知"}
STRATEGY = {"targeted": "Targeted Loop（诊断 → 定向修复）", "generic": "Generic Retry（通用重试）"}
VERIFIER = {"self": "Self（无 Gold）", "oracle": "Oracle（上界）"}

runner.load_dotenv()


@st.cache_resource(show_spinner="加载 schema、few-shot 示例和开发集…")
def bundle():
    return runner.load_bundle()


@st.cache_data(ttl=600, show_spinner=False)
def eval_options():
    conn = data.connect()
    try:
        return runner.eval_choices(conn)
    finally:
        conn.close()


st.title("运行 Loop", anchor=False)
st.caption("选题目、模型和策略后点执行。Loop 每走一步，下面的 trace 就追加一步；Gold 判分在每题的 Loop 结束后才出现，Loop 本身看不到。")

try:
    B = bundle()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

live: runner.LiveRun | None = st.session_state.get("live")
running = live is not None and live.status == "running"

# ------------------------------------------------------------------ form

split = st.segmented_control("数据集", ["dev", "eval"], default="dev", key="split",
                             format_func=lambda s: "开发集（30 题）" if s == "dev" else
                             ("评测集（89 题）" if runner.eval_allowed() else "评测集（已锁定）"))
if split == "eval" and not runner.eval_allowed():
    st.caption("评测集已锁定（决策 D2：配置冻结后才在评测集上跑）。部署时设置环境变量 SHT_ALLOW_EVAL=1 即可解锁。")
    options = []
elif split == "eval":
    options = eval_options()
else:
    options = runner.dev_choices(B)
by_id = {o["case_id"]: o for o in options}

with st.form("run_form", border=True):
    limit = runner.MAX_CASES or None
    picked = st.multiselect("题目（可多选）" + (f"，每次最多 {limit} 题" if limit else ""), list(by_id),
                            max_selections=limit, default=[c for c in ["dw:dw_5478"] if c in by_id],
                            format_func=lambda c: f"{c.split(':')[-1]} · {by_id[c]['question'][:110]}")
    pick_all = st.checkbox(f"选择全部 {len(by_id)} 题（忽略上面的选择）", value=False, disabled=bool(limit))
    if pick_all:
        picked = list(by_id)
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.1, 1.3])
    model = c1.segmented_control("模型", list(runner.MODELS), default="glm-4-flash", key="model")
    strategy = c2.segmented_control("策略", list(STRATEGY), default="targeted", key="strategy",
                                    format_func=lambda s: STRATEGY[s].split("（")[0])
    verifier = c3.segmented_control("Verifier", list(VERIFIER), default="self", key="verifier",
                                    format_func=VERIFIER.get)
    max_repairs = c4.segmented_control("最大修复次数", list(range(1, runner.MAX_REPAIRS + 1)), default=1,
                                       key="max_repairs", format_func=lambda k: f"{k} 次",
                                       help="每道题最多修复几轮；SQL 尝试次数 = 修复次数 + 1。自检通过就提前停止。")
    few_shot = st.segmented_control(
        "Few-shot 示例", ["static", "dynamic"], default="static", key="few_shot",
        format_func=lambda m: "固定 3 例（baseline-v2）" if m == "static" else "相似题检索（baseline-v3）",
        help="相似题检索：每道题从已解题库（排除评测集和开发集）里找最相似的 4 道题作为示例，并把它们用到的表加进 schema，"
             "让模型看到这个数仓里同类问题用哪些表、怎么关联。")
    knowledge = st.toggle(
        "数仓使用说明（外层循环知识）", value=False, key="knowledge",
        help="从已解题库（排除评测集和开发集）统计出的约定：问题里的概念通常用哪些表、相似表之间怎么选、"
             "表与表常用的关联键和 INNER / LEFT JOIN。每题只放与它相关的几条。")
    publish = st.checkbox("跑完发布到 Delta（之后可在 Console 的「逐题追踪」里查看）", value=True)
    st.caption("glm-4-flash 免费；deepseek-flash 按量计费（每题每轮约几分钱人民币）。每题约 30–60 秒，题目按顺序执行；"
               "修复次数越多，最坏情况下的耗时和 token 越多（每轮最多多 1–2 次 LLM 调用）。Oracle 用 Gold 判断要不要修，只作上界参考。")
    go = st.form_submit_button("执行", icon=":material/play_arrow:", type="primary", disabled=running)

if go:
    problem = None
    if split == "eval" and not runner.eval_allowed():
        problem = "评测集已锁定（决策 D2）。"
    elif not split:
        problem = "请选择数据集。"
    elif not picked:
        problem = "请至少选择一道题。"
    elif not model or not strategy or not verifier or not max_repairs:
        problem = "请选择模型、策略、Verifier 和最大修复次数。"
    elif not runner.model_available(model):
        problem = f"没有配置 {model} 的 API key。"
    if problem:
        st.warning(problem)
    else:
        req = runner.RunRequest(split=split, case_ids=picked, model=model, strategy=strategy, verifier=verifier,
                                max_repairs=int(max_repairs), few_shot=few_shot or "static",
                                knowledge=bool(knowledge), publish=publish)
        st.session_state["live"] = runner.start_run(req, B, data.connect)
        st.rerun()


# ------------------------------------------------------------------ live trace

from app.trace_view import render_case  # noqa: E402


def palette() -> dict:
    """Validated categorical slots 1-3 (light / dark steps), neutral baseline and fixed status colors."""
    dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    return {"series": ["#3987e5", "#d95926", "#199e70"] if dark else ["#2a78d6", "#eb6834", "#1baf7a"],
            "baseline": "#898781", "surface": "#1a1a19" if dark else "#fcfcfb",
            "status": {"报错": "#d03b3b", "能执行但错": "#fab219", "答对": "#0ca30c"},
            "status_ink": {"报错": "#ffffff", "能执行但错": "#0b0b0b", "答对": "#ffffff"}}


def render_charts(lv: runner.LiveRun) -> None:
    cases = report.cases_from_events(lv.snapshot())
    done = [c for c in cases if c["judged"] and c["executions"]]
    if not done:
        return
    P, n, k_max = palette(), len(done), lv.request.max_repairs + 1
    y_scale = alt.Scale(domain=[0, max(n, 1)], nice=False)
    y_axis = alt.Axis(tickMinStep=1, format="d", title="题数", grid=True)
    c1, c2 = st.columns(2)

    with c1.container(border=True):
        st.markdown("**修复前后对比**（首次尝试 vs Loop 选出的最终答案；自检通过 ≠ 答对）")
        df = pd.DataFrame(report.first_final(cases))
        base = alt.Chart(df).encode(
            x=alt.X("指标:N", sort=list(report.METRICS), title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("阶段:N", sort=["首次", "最终"]),
            y=alt.Y("题数:Q", scale=y_scale, axis=y_axis),
            color=alt.Color("阶段:N", sort=["首次", "最终"], title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["首次", "最终"], range=[P["baseline"], P["series"][0]])),
            tooltip=["指标", "阶段", "题数"])
        bars = base.mark_bar(size=26, cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                             stroke=P["surface"], strokeWidth=2)
        labels = base.mark_text(dy=-8, fontSize=12).encode(text="题数:Q", color=alt.value("#898781"))
        st.altair_chart((bars + labels).properties(height=260), width="stretch")

    with c2.container(border=True):
        st.markdown(f"**逐次尝试的指标变化**（最多 {k_max} 次尝试）")
        df = pd.DataFrame(report.attempt_progress(cases, k_max))
        df["尝试"] = df["尝试"].map(lambda k: f"第 {k} 次")
        order = [f"第 {k} 次" for k in range(1, k_max + 1)]
        metrics = list(report.METRICS)
        color = alt.Color("指标:N", sort=metrics, title=None, legend=alt.Legend(orient="top"),
                          scale=alt.Scale(domain=metrics, range=P["series"]))
        # secondary encodings so equal values stay distinguishable: point shape + a dashed middle series
        shape = alt.Shape("指标:N", sort=metrics, legend=None,
                          scale=alt.Scale(domain=metrics, range=["circle", "square", "triangle-up"]))
        dash = alt.StrokeDash("指标:N", sort=metrics, legend=None,
                              scale=alt.Scale(domain=metrics, range=[[1, 0], [6, 4], [1, 0]]))
        x = alt.X("尝试:O", sort=order, title=None, axis=alt.Axis(labelAngle=0),
                  scale=alt.Scale(paddingOuter=0.35))
        base = alt.Chart(df).encode(x=x, y=alt.Y("题数:Q", scale=y_scale, axis=y_axis), color=color,
                                    tooltip=["尝试", "指标", "题数"])
        lines = base.mark_line(strokeWidth=2).encode(strokeDash=dash) + \
            base.mark_point(size=80, filled=True, stroke=P["surface"], strokeWidth=1.5, opacity=1).encode(shape=shape)
        last = df[df["尝试"] == order[-1]]
        ends = alt.layer(*[  # direct labels at the line ends, fixed offsets so equal values don't collide
            alt.Chart(last[last["指标"] == m]).mark_text(align="left", dx=10, dy=dy, fontSize=12).encode(
                x=x, y=alt.Y("题数:Q", scale=y_scale), text="指标:N", color=color)
            for m, dy in zip(metrics, (-12, 0, 12))])
        st.altair_chart((lines + ends).properties(height=260), width="stretch")
        st.caption("第 k 次 = 如果最多允许 k 次尝试时的题数；已停止的题沿用最后一次的状态。")

    with st.container(border=True):
        st.markdown("**逐题 × 逐次尝试的状态**（★ = Loop 选出的最终答案；悬停看详情）")
        df = pd.DataFrame(report.status_grid(cases))
        df["尝试"] = df["尝试"].map(lambda k: f"第 {k} 次")
        df["标签"] = df["状态"] + df["最终答案"].map(lambda v: " ★" if v else "")
        order = [f"第 {k} 次" for k in range(1, k_max + 1)]
        enc = dict(x=alt.X("尝试:O", sort=order, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                   y=alt.Y("题目:N", title=None, sort=None, scale=alt.Scale(paddingInner=0.12)))
        cells = alt.Chart(df).mark_rect(cornerRadius=4, stroke=P["surface"], strokeWidth=2).encode(
            **enc, color=alt.Color("状态:N", title=None, legend=alt.Legend(orient="bottom"),
                                   scale=alt.Scale(domain=list(report.STATUS),
                                                   range=[P["status"][s] for s in report.STATUS])),
            tooltip=["题目", "尝试", "状态", "执行", "自检", "由谁生成", "最终答案"])
        text = alt.Chart(df).mark_text(fontSize=12, fontWeight="bold").encode(
            **enc, text="标签:N", color=alt.Color("状态:N", legend=None, scale=alt.Scale(
                domain=list(report.STATUS), range=[P["status_ink"][s] for s in report.STATUS])))
        # independent color scales: otherwise the text layer reuses the fill colors and disappears into its cell
        grid = (cells + text).resolve_scale(color="independent").properties(height=alt.Step(48))  # 48px per row
        st.altair_chart(grid, width="stretch")


def case_block(lv: runner.LiveRun, ev: list[dict], sev: dict, active: bool) -> None:
    cid = sev["case_id"]
    with st.container(border=True):
        st.markdown(f"**第 {sev['index']} / {sev['total']} 题 · `{cid.split(':')[-1]}`**")
        st.caption(sev["question"])
        render_case([e for e in ev if e["case_id"] == cid and e["step"] != "start"], active)


def render_cases(lv: runner.LiveRun, ev: list[dict], starts: list[dict]) -> None:
    """Finished questions -> one summary table + a picker for the full timeline; the running one in full.
    Keeps each 1-second refresh small even when all 30 questions are selected."""
    recs = {c["case_id"]: c for c in report.cases_from_events(ev)}
    finished = [s for s in starts if recs.get(s["case_id"], {}).get("judged")]
    running_now = [s for s in starts if s not in finished]
    if finished:
        rows = []
        for s in finished:
            c = recs[s["case_id"]]
            j, fin = c["judged"], c["final"]
            self_ok = bool(fin) and fin["verifier_decision"] == "PASS"
            rows.append({"题目": s["case_id"].split(":")[-1], "结局": c["outcome"],
                         "各次尝试": " → ".join(report.attempt_status(c, i) for i in range(len(c["executions"]))),
                         "最终自检": "通过" if self_ok else "未通过", "Gold 判分": "对" if j["final_correct"] else "错",
                         "漏报": "是" if self_ok and not j["final_correct"] else "",
                         "修复": " → ".join(r["skill"] for r in c["repairs"]) or "—",
                         "token": j["tokens"], "耗时（秒）": round(j["latency_ms"] / 1000, 1)})
        st.markdown(f"**已完成 {len(finished)} 题**（“漏报” = 自检通过但 Gold 判错）")
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(38 * len(rows) + 40, 420))
        pick = st.selectbox("查看某题的完整执行过程", [s["case_id"] for s in finished], index=None,
                            key="trace_case", placeholder="选择题目…",
                            format_func=lambda cid: f"{cid.split(':')[-1]} · {recs[cid]['outcome']}")
        if pick:
            case_block(lv, ev, next(s for s in finished if s["case_id"] == pick), False)
    for s in running_now:
        case_block(lv, ev, s, lv.status == "running")


def render_live(lv: runner.LiveRun) -> None:
    ev = lv.snapshot()
    req = lv.request
    starts = [e for e in ev if e["step"] == "start"]
    done = sum(e["step"] == "judged" for e in ev)
    with st.container(horizontal=True):
        st.metric("状态", {"running": "运行中", "done": "完成", "error": "出错"}[lv.status], border=True)
        st.metric("进度", f"{done} / {len(req.case_ids)} 题", border=True)
        st.metric("已用时间", f"{lv.elapsed:.0f} 秒", border=True)
        if lv.summary:
            s = lv.summary
            st.metric("答对 首次 → 最终", f"{s['first_correct']} → {s['final_correct']}", border=True)
            st.metric("可执行 首次 → 最终", f"{s['executable_first']} → {s['executable_final']}", border=True)
    st.caption(f"{req.model} · {STRATEGY[req.strategy]} · Verifier {VERIFIER[req.verifier]} · "
               f"最多修复 {req.max_repairs} 次 · 示例 {'相似题检索' if req.few_shot == 'dynamic' else '固定 3 例'} · "
               f"使用说明 {'开' if req.knowledge else '关'} · "
               f"{'开发集' if req.split == 'dev' else '评测集'}")
    for e in ev:
        if e["step"] == "error":
            st.error(e["message"])
            with st.expander("错误详情"):
                st.code(e.get("trace", ""), language="text")
    if lv.report:
        tab_rep, tab_trace = st.tabs([":material/summarize: 分析报告", ":material/timeline: 执行 Trace"])
        with tab_rep:
            render_charts(lv)
            st.download_button("下载报告（Markdown）", lv.report, file_name=f"{lv.run_id}-report.md",
                               mime="text/markdown", icon=":material/download:", on_click="ignore")
            with st.container(border=True):
                st.markdown(lv.report)
        with tab_trace:
            render_cases(lv, ev, starts)
    else:
        render_cases(lv, ev, starts)
    if lv.status == "running" and not starts:
        st.status(":shimmer[连接 SQL Warehouse、准备模型…]", type="step", state="running")
    pub = [e for e in ev if e["step"] == "publish"]
    if pub and pub[-1]["state"] == "done":
        st.success(f"已发布到 Delta：`{pub[-1]['run_id']}`。可在 Console 的「逐题追踪」里选择这次运行查看"
                   "（Console 数据缓存 5 分钟，可用右上角菜单 Rerun 刷新）。", icon=":material/cloud_done:")
    elif pub and lv.status == "running":
        st.status(":shimmer[发布到 Delta…]", type="step", state="running")


def live_panel() -> None:
    lv: runner.LiveRun | None = st.session_state.get("live")
    if lv is None:
        st.info("还没有运行。选择题目后点「执行」。", icon=":material/info:")
        return
    render_live(lv)
    if lv.status != "running" and st.session_state.get("rendered_final") != id(lv):
        st.session_state["rendered_final"] = id(lv)
        st.rerun(scope="app")  # re-enable the form and stop polling


st.subheader("执行过程与分析报告", anchor=False)
st.fragment(live_panel, run_every=1 if running else None)()
