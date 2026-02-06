# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="v0.7 客诉分析看板", layout="wide")

# =========================
# 工具函数
# =========================
def _try_parse_datetime(s):
    if s is None:
        return s
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s.loc[mask], errors="coerce")
    return out


def _starts_with_v07(s):
    return s.astype(str).str.strip().str.lower().str.startswith("v0.7")


def _make_beautiful_pie(df, name_col, value_col, title, max_categories=10):
    tmp = df[[name_col, value_col]].copy()
    tmp[name_col] = tmp[name_col].fillna("未填写")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0)

    g = (
        tmp.groupby(name_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )

    total = g[value_col].sum()
    if total <= 0:
        return None

    if len(g) > max_categories:
        top = g.iloc[:max_categories - 1]
        others = g.iloc[max_categories - 1:]
        g = pd.concat([
            top,
            pd.DataFrame({name_col: ["其他"], value_col: [others[value_col].sum()]})
        ])

    fig = go.Figure(go.Pie(
        labels=g[name_col],
        values=g[value_col],
        hole=0.4,
        textinfo="percent+label",
        textposition="inside",
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color="white", width=2)
        ),
        hovertemplate="%{label}<br>问题数: %{value:,.0f}<br>占比: %{percent:.1%}<extra></extra>",
        sort=False
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=[dict(
            text=f"总计<br>{total:,.0f}",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )],
        margin=dict(t=50, b=20, l=20, r=150)
    )
    return fig


# =========================
# 页面标题
# =========================
st.title("v0.7 款式客诉分析看板")

# =========================
# Sidebar：文件上传
# =========================
with st.sidebar:
    st.header("① 上传主数据")
    main_file = st.file_uploader("主数据 Excel", type=["xlsx", "xls"])

    st.divider()
    st.header("② 上传其他表（原样展示）")
    extra_file = st.file_uploader("改进方案 / 其他数据", type=["xlsx", "xls"], key="extra")

# =========================
# 额外表展示（保留）
# =========================
if extra_file is not None:
    extra_df = pd.read_excel(extra_file)
    st.subheader("📎 其他数据表（原样展示）")
    st.dataframe(extra_df, use_container_width=True, height=500)
    st.divider()

# =========================
# 主数据
# =========================
if main_file is None:
    st.warning("请先上传主数据")
    st.stop()

df = pd.read_excel(main_file)

df = df[_starts_with_v07(df["erpsku款式名称"])].copy()
df["_order_time"] = _try_parse_datetime(df["平台订单时间(day)"])
df["问题数"] = pd.to_numeric(df["问题数"], errors="coerce").fillna(0)

# =========================
# KPI
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("订单数", df["订单参考号"].nunique())
c2.metric("ERP SKU 数", df["erp sku"].nunique())
c3.metric("问题数", int(df["问题数"].sum()))
c4.metric("款式数", df["erpsku款式名称"].nunique())

st.divider()

# =========================
# ⭐ 款式风险识别表（你说缺的那张）
# =========================
st.subheader("🚨 款式客诉统计表")

tmp = df.copy()
tmp["_pair"] = tmp["订单参考号"].astype(str) + "||" + tmp["erp sku"].astype(str)

style_risk = (
    tmp.groupby("erpsku款式名称", as_index=False)
    .agg(
        数量=("_pair", "nunique"),
        问题数=("问题数", "sum")
    )
)

style_risk["客诉率(%)"] = (
    style_risk["问题数"] / style_risk["数量"] * 100
).round(2)

style_risk = style_risk.sort_values("客诉率(%)", ascending=False)

st.dataframe(style_risk, use_container_width=True, height=420)

st.divider()

# =========================
# 一级 → 二级 联动分析
# =========================
st.subheader("一级 → 二级问题联动分析")

level1_opts = ["全部"] + sorted(df["一级问题名称"].dropna().unique())
selected_l1 = st.selectbox(
    "选择一级问题（驱动下方二级分析）",
    level1_opts
)

if selected_l1 == "全部":
    df_l1 = df.copy()
else:
    df_l1 = df[df["一级问题名称"] == selected_l1]

# =========================
# 饼图联动
# =========================
col1, col2 = st.columns(2)

with col1:
    fig1 = _make_beautiful_pie(
        df, "一级问题名称", "问题数", "一级问题分布（全量）"
    )
    if fig1:
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    title = "二级问题分布"
    if selected_l1 != "全部":
        title += f"（一级问题：{selected_l1}）"

    fig2 = _make_beautiful_pie(
        df_l1, "二级问题名称", "问题数", title
    )
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# 排行表联动
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 一级问题 Top10（全量）")
    st.dataframe(
        df.groupby("一级问题名称", as_index=False)
        .agg(问题数=("问题数", "sum"))
        .sort_values("问题数", ascending=False)
        .head(10),
        use_container_width=True
    )

with col2:
    subtitle = "全部一级问题" if selected_l1 == "全部" else f"一级问题：{selected_l1}"
    st.markdown(f"#### 二级问题 Top10（{subtitle}）")
    st.dataframe(
        df_l1.groupby("二级问题名称", as_index=False)
        .agg(问题数=("问题数", "sum"))
        .sort_values("问题数", ascending=False)
        .head(10),
        use_container_width=True
    )

st.divider()

# =========================
# 明细联动
# =========================
with st.expander("📋 查看明细（随一级问题联动）"):
    st.dataframe(df_l1, use_container_width=True, height=500)
