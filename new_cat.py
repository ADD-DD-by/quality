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
def _try_parse_datetime(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(
            s.loc[mask], errors="coerce", infer_datetime_format=True
        )
    return out


def _read_excel(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)


def _safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("")


def _starts_with_v07(s: pd.Series) -> pd.Series:
    t = _safe_str_series(s).str.strip().str.lower()
    return t.str.startswith("v0.7")


def _make_beautiful_pie(df, name_col, value_col, title, max_categories=10):
    tmp = df[[name_col, value_col]].copy()
    tmp[name_col] = tmp[name_col].fillna("未填写")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0)

    grouped = (
        tmp.groupby(name_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )

    total = grouped[value_col].sum()
    if total <= 0:
        return None

    if len(grouped) > max_categories:
        top = grouped.iloc[:max_categories-1]
        others = grouped.iloc[max_categories-1:]
        grouped = pd.concat([
            top,
            pd.DataFrame({
                name_col: ["其他"],
                value_col: [others[value_col].sum()]
            })
        ])

    fig = go.Figure(go.Pie(
        labels=grouped[name_col],
        values=grouped[value_col],
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
        margin=dict(t=50, b=20, l=20, r=150),
        showlegend=True,
        legend=dict(y=0.5),
        annotations=[dict(
            text=f"总计<br>{total:,.0f}",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )]
    )
    return fig


# =========================
# 页面
# =========================
st.title("v0.7 款式客诉分析")

with st.sidebar:
    st.header("① 上传主数据")
    main_file = st.file_uploader("上传 Excel", type=["xlsx", "xls"])

if main_file is None:
    st.stop()

df = _read_excel(main_file)

required_cols = [
    "订单参考号", "平台订单时间(day)", "站点",
    "erpsku款式名称", "erp sku", "问题数",
    "一级问题名称", "二级问题名称"
]
for c in required_cols:
    if c not in df.columns:
        st.error(f"缺少字段：{c}")
        st.stop()

# v0.7
df = df[_starts_with_v07(df["erpsku款式名称"])].copy()
df["_order_time"] = _try_parse_datetime(df["平台订单时间(day)"])
df["问题数"] = pd.to_numeric(df["问题数"], errors="coerce").fillna(0)

# =========================
# 全局筛选
# =========================
with st.sidebar:
    st.header("② 全局筛选")

    if df["_order_time"].notna().any():
        tmin, tmax = df["_order_time"].min(), df["_order_time"].max()
        date_range = st.date_input(
            "时间范围",
            value=(tmin.date(), tmax.date())
        )
    else:
        date_range = None

    sites = st.multiselect(
        "站点",
        sorted(df["站点"].dropna().unique()),
        default=sorted(df["站点"].dropna().unique())
    )

filtered = df.copy()
if date_range and df["_order_time"].notna().any():
    start, end = date_range
    filtered = filtered[
        (filtered["_order_time"] >= pd.to_datetime(start)) &
        (filtered["_order_time"] <= pd.to_datetime(end) + pd.Timedelta(days=1))
    ]

if sites:
    filtered = filtered[filtered["站点"].isin(sites)]

# =========================
# KPI
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("订单数", filtered["订单参考号"].nunique())
c2.metric("ERP SKU 数", filtered["erp sku"].nunique())
c3.metric("问题数", int(filtered["问题数"].sum()))
c4.metric("款式数", filtered["erpsku款式名称"].nunique())

st.divider()

# =========================
# 统计表（客诉率）
# =========================
tmp = filtered.copy()
tmp["_pair"] = tmp["订单参考号"].astype(str) + tmp["erp sku"].astype(str)

summary = (
    tmp.groupby("erpsku款式名称", as_index=False)
    .agg(
        销售数量=("_pair", "nunique"),
        问题数=("问题数", "sum")
    )
)
summary["客诉率(%)"] = (summary["问题数"] / summary["销售数量"] * 100).round(2)

st.subheader("📊 款式客诉率")
st.dataframe(summary.sort_values("客诉率(%)", ascending=False),
             use_container_width=True)

st.divider()

# =========================
# 一级问题联动选择
# =========================
st.subheader("🎯 一级 → 二级问题联动分析")

level1_options = ["全部"] + sorted(filtered["一级问题名称"].dropna().unique())
selected_l1 = st.selectbox("选择一级问题", level1_options)

if selected_l1 == "全部":
    filtered_l1 = filtered.copy()
else:
    filtered_l1 = filtered[filtered["一级问题名称"] == selected_l1]

# =========================
# 饼图联动
# =========================
col1, col2 = st.columns(2)

with col1:
    fig1 = _make_beautiful_pie(
        filtered,
        "一级问题名称",
        "问题数",
        "一级问题分布（全量）"
    )
    if fig1:
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    title = "二级问题分布"
    if selected_l1 != "全部":
        title += f"（一级：{selected_l1}）"

    fig2 = _make_beautiful_pie(
        filtered_l1,
        "二级问题名称",
        "问题数",
        title
    )
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================
# 问题排行（联动）
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 一级问题 Top10")
    st.dataframe(
        filtered.groupby("一级问题名称", as_index=False)
        .agg(问题数=("问题数", "sum"))
        .sort_values("问题数", ascending=False)
        .head(10),
        use_container_width=True
    )

with col2:
    st.markdown("#### 二级问题 Top10（联动）")
    st.dataframe(
        filtered_l1.groupby("二级问题名称", as_index=False)
        .agg(问题数=("问题数", "sum"))
        .sort_values("问题数", ascending=False)
        .head(10),
        use_container_width=True
    )

st.divider()

# =========================
# 明细
# =========================
with st.expander("📋 查看明细"):
    st.dataframe(filtered_l1, use_container_width=True)

