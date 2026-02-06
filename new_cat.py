# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="v0.7 客诉分析看板", layout="wide")

# =========================
# 工具函数
# =========================
def _try_parse_datetime(s: pd.Series) -> pd.Series:
    """解析时间字段，优先支持 YYYYMMDD（如 20260102），失败再用自动解析兜底"""
    if s is None:
        return s
    
    # 先按 YYYYMMDD 强制解析
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    
    # 未解析成功的再自动兜底
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
    # 以 v0.7 开头（忽略大小写、前后空格）
    t = _safe_str_series(s).str.strip().str.lower()
    return t.str.startswith("v0.7")

def _make_beautiful_pie(df: pd.DataFrame, name_col: str, value_col: str, title: str, max_categories=10):
    """
    绘制美观的饼图
    """
    tmp = df[[name_col, value_col]].copy()
    tmp[name_col] = tmp[name_col].fillna("未填写")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0)
    
    # 分组汇总
    grouped = (
        tmp.groupby(name_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )
    
    total_value = grouped[value_col].sum()
    
    if total_value <= 0:
        st.info(f"{title}：当前筛选下 {value_col} 全为 0 / 空，无法绘图。")
        return None
    
    # 如果类别太多，合并小类别为"其他"
    if len(grouped) > max_categories:
        top_n = grouped.iloc[:max_categories-1]
        others = grouped.iloc[max_categories-1:]
        others_sum = others[value_col].sum()
        
        if others_sum > 0:
            others_row = pd.DataFrame({
                name_col: ["其他"],
                value_col: [others_sum]
            })
            grouped = pd.concat([top_n, others_row], ignore_index=True)
        else:
            grouped = top_n
    
    # 使用专业的配色方案（Plotly 默认的 Set3 色系，适合分类数据）
    colors = px.colors.qualitative.Set3
    
    # 创建饼图
    fig = go.Figure()
    
    # 使用自定义的标签格式
    labels = grouped[name_col].tolist()
    values = grouped[value_col].tolist()
    
    # 计算百分比
    percentages = [(v / total_value * 100) for v in values]
    
    # 创建悬浮文本
    hover_texts = []
    for label, value, pct in zip(labels, values, percentages):
        hover_texts.append(f"{label}<br>问题数: {value:,.0f}<br>占比: {pct:.1f}%")
    
    # 添加饼图
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hoverinfo="text",
        text=labels,  # 在饼图上显示标签
        textinfo="percent+label",  # 显示百分比和标签
        textposition="inside",  # 文字在内部
        insidetextorientation="radial",  # 文字径向排列
        hole=0.4,  # 甜甜圈图，中间留空
        marker=dict(
            colors=colors[:len(labels)],  # 使用配色
            line=dict(color='white', width=2)  # 白色边框
        ),
        hovertemplate="%{text}<br>问题数: %{value:,.0f}<br>占比: %{percent:.1%}<extra></extra>",
        sort=False  # 保持排序顺序
    ))
    
    # 美化布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.5,
            xanchor="center"
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=50, b=20, l=20, r=150),  # 右侧留空间给图例
        paper_bgcolor="rgba(0,0,0,0)",  # 透明背景
        plot_bgcolor="rgba(0,0,0,0)",  # 透明背景
        annotations=[
            dict(
                text=f"总计: {total_value:,.0f}",
                showarrow=False,
                x=0.5,
                y=0.5,
                font=dict(size=14, color="gray")
            )
        ]
    )
    
    # 更新饼图的文字字体
    fig.update_traces(
        textfont=dict(
            size=11,
            family="Arial, sans-serif"
        ),
        outsidetextfont=dict(size=10)
    )
    
    return fig

def _create_problem_hierarchy_chart(filtered_df):
    """
    创建问题层级桑基图（一级问题 -> 二级问题）
    """
    if len(filtered_df) == 0:
        return None
    
    # 准备数据
    tmp = filtered_df.copy()
    tmp["一级问题名称"] = tmp["一级问题名称"].fillna("未填写")
    tmp["二级问题名称"] = tmp["二级问题名称"].fillna("未填写")
    tmp["问题数"] = pd.to_numeric(tmp["问题数"], errors="coerce").fillna(0)
    
    # 汇总一级到二级的问题数
    hierarchy_df = (
        tmp.groupby(["一级问题名称", "二级问题名称"])
        .agg(问题数=("问题数", "sum"))
        .reset_index()
    )
    
    # 过滤掉问题数为0的行
    hierarchy_df = hierarchy_df[hierarchy_df["问题数"] > 0]
    
    if len(hierarchy_df) == 0:
        return None
    
    # 创建唯一的节点名称
    level1_nodes = hierarchy_df["一级问题名称"].unique().tolist()
    level2_nodes = hierarchy_df["二级问题名称"].unique().tolist()
    
    all_nodes = level1_nodes + level2_nodes
    
    # 创建节点映射
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    # 创建桑基图数据
    source = [node_indices[row["一级问题名称"]] for _, row in hierarchy_df.iterrows()]
    target = [node_indices[row["二级问题名称"]] for _, row in hierarchy_df.iterrows()]
    value = [row["问题数"] for _, row in hierarchy_df.iterrows()]
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=px.colors.qualitative.Set3[:len(all_nodes)]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate="%{source.label} → %{target.label}<br>问题数: %{value:,.0f}<extra></extra>"
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="问题层级流向图（一级问题 → 二级问题）",
            font=dict(size=16),
            x=0.5,
            xanchor="center"
        ),
        font=dict(size=10),
        margin=dict(t=50, b=20, l=20, r=20),
        height=500
    )
    
    return fig

# =========================
# 页面
# =========================
st.title("📌 v0.7 款式客诉分析")
st.markdown("""
- **全局筛选**：时间范围（按 平台订单时间(day)）、站点、erpsku款式名称（多选）
- **图表交互**：所有图表均可悬停查看详情，点击图例可筛选
""")

with st.sidebar:
    st.header("① 上传主数据（Excel）")
    main_file = st.file_uploader("上传 Excel（主数据）", type=["xlsx", "xls"])
    
    st.divider()
    st.header("② 改进情况数据上传")
    extra_file = st.file_uploader("上传 Excel（额外展示用）", type=["xlsx", "xls"], key="extra")

# =========================
# 额外表展示
# =========================
if extra_file is not None:
    try:
        extra_df = _read_excel(extra_file)
        st.subheader("📎 改进方案")
        st.dataframe(extra_df, use_container_width=True, height=520)
    except Exception as e:
        st.error(f"额外表读取失败：{e}")
    st.divider()

# =========================
# 主数据分析
# =========================
if main_file is None:
    st.warning("请先在左侧上传主数据 Excel。")
    st.stop()

try:
    df = _read_excel(main_file)
except Exception as e:
    st.error(f"主数据读取失败：{e}")
    st.stop()

# 必要字段检查
required_cols = [
    "订单参考号",
    "平台订单时间(day)",
    "站点",
    "erpsku款式名称",
    "erp sku",
    "问题数",
    "一级问题名称",
    "二级问题名称",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"主数据缺少必要字段：{missing}")
    st.stop()

# =========================
# 1) 强制筛选 v0.7
# =========================
df = df.copy()
df = df[_starts_with_v07(df["erpsku款式名称"])].copy()
st.caption(f"✅ 已筛选：erpsku款式名称 以 v0.7 开头（当前 {len(df):,} 行）")

# =========================
# 2) 时间字段处理（YYYYMMDD）
# =========================
df["_order_time"] = _try_parse_datetime(df["平台订单时间(day)"])
time_parse_ok = df["_order_time"].notna().sum()
if time_parse_ok == 0:
    st.warning("⚠️ 平台订单时间(day) 无法解析为日期，时间筛选不可用。")

# =========================
# 3) 全局筛选
# =========================
with st.sidebar:
    st.divider()
    st.header("③ 全局筛选")
    
    # 时间范围
    if time_parse_ok > 0:
        tmin = df["_order_time"].min()
        tmax = df["_order_time"].max()
        date_range = st.date_input(
            "时间范围（平台订单时间）",
            value=(tmin.date(), tmax.date()),
            min_value=tmin.date(),
            max_value=tmax.date(),
        )
    else:
        date_range = None
        st.info("时间列不可解析，已跳过时间筛选")
    
    # 站点
    site_options = sorted(df["站点"].dropna().unique().tolist())
    selected_sites = st.multiselect("站点（多选）", site_options, default=site_options)
    
    # 款式
    style_options = sorted(df["erpsku款式名称"].dropna().unique().tolist())
    selected_styles = st.multiselect("erpsku款式名称（多选）", style_options, default=style_options)

# =========================
# 4) 应用筛选
# =========================
filtered = df.copy()

if date_range is not None and time_parse_ok > 0:
    start_date, end_date = date_range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[
        (filtered["_order_time"] >= start_dt) & (filtered["_order_time"] <= end_dt)
    ]

if selected_sites:
    filtered = filtered[filtered["站点"].isin(selected_sites)]
else:
    filtered = filtered.iloc[0:0]

if selected_styles:
    filtered = filtered[filtered["erpsku款式名称"].isin(selected_styles)]
else:
    filtered = filtered.iloc[0:0]

# =========================
# KPI
# =========================
filtered["问题数"] = pd.to_numeric(filtered["问题数"], errors="coerce").fillna(0)

st.subheader("📈 关键指标概览")
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric(
        label="筛选后行数", 
        value=f"{len(filtered):,}",
        delta=None
    )

with kpi_cols[1]:
    st.metric(
        label="订单数", 
        value=f"{filtered['订单参考号'].nunique():,}",
        delta=None
    )

with kpi_cols[2]:
    st.metric(
        label="ERP SKU 数", 
        value=f"{filtered['erp sku'].nunique():,}",
        delta=None
    )

with kpi_cols[3]:
    total_problems = filtered["问题数"].sum()
    st.metric(
        label="总问题数", 
        value=f"{total_problems:,.0f}",
        delta=None
    )

st.divider()

# =========================
# 统计表（含 erpsku客诉率）
# =========================
st.subheader("统计表（按 erpsku款式名称）")
tmp = filtered.copy()
tmp["_pair"] = tmp["订单参考号"].astype(str) + "||" + tmp["erp sku"].astype(str)

summary = (
    tmp.groupby("erpsku款式名称", as_index=False)
    .agg(
        销售数量=("_pair", pd.Series.nunique),
        问题数=("问题数", "sum"),
    )
)

# 客诉率
summary["erpsku客诉率"] = np.where(
    summary["销售数量"] > 0,
    summary["问题数"] / summary["销售数量"],
    0
)
summary["erpsku客诉率"] = summary["erpsku客诉率"].round(4)

# 添加百分比列用于显示
summary["客诉率(%)"] = (summary["erpsku客诉率"] * 100).round(2)

summary = summary.sort_values(
    ["erpsku客诉率", "问题数", "销售数量"],
    ascending=[False, False, True]
)

# 格式化显示
display_summary = summary.copy()
display_summary["销售数量"] = display_summary["销售数量"].apply(lambda x: f"{x:,}")
display_summary["问题数"] = display_summary["问题数"].apply(lambda x: f"{x:,}")
display_summary["客诉率(%)"] = display_summary["客诉率(%)"].apply(lambda x: f"{x:.2f}%")

st.dataframe(
    display_summary[["erpsku款式名称", "销售数量", "问题数", "客诉率(%)"]],
    use_container_width=True,
    height=420,
    column_config={
        "erpsku款式名称": "款式名称",
        "销售数量": "销售数量",
        "问题数": "问题数",
        "客诉率(%)": "客诉率"
    }
)

st.divider()

# =========================
# 问题分析图表
# =========================
st.subheader("📊 问题分析")

# 第一行：饼图
col1, col2 = st.columns(2)

with col1:
    fig1 = _make_beautiful_pie(
        filtered, 
        name_col="一级问题名称", 
        value_col="问题数", 
        title="一级问题分布"
    )
    if fig1:
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("暂无一级问题数据")

with col2:
    fig2 = _make_beautiful_pie(
        filtered, 
        name_col="二级问题名称", 
        value_col="问题数", 
        title="二级问题分布"
    )
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("暂无二级问题数据")

# 第二行：桑基图
st.subheader("🔗 问题层级关系")
sankey_fig = _create_problem_hierarchy_chart(filtered)
if sankey_fig:
    st.plotly_chart(sankey_fig, use_container_width=True)
else:
    st.info("暂无层级关系数据")

st.divider()

# =========================
# 明细展示
# =========================
with st.expander("📋 查看筛选后的明细数据", expanded=False):
    st.dataframe(
        filtered.drop(columns=["_order_time", "_pair"], errors="ignore"),
        use_container_width=True,
        height=520,
    )

# =========================
# 样式优化
# =========================
st.markdown("""
<style>
    /* 美化metric卡片 */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666;
        font-size: 14px;
    }
    
    [data-testid="stMetricValue"] {
        color: #1e88e5;
        font-size: 24px;
        font-weight: bold;
    }
    
    /* 美化展开器 */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    /* 美化分隔线 */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #ddd, transparent);
    }
</style>
""", unsafe_allow_html=True)
