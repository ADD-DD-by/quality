import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# 基础函数
# =========================
def logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

# =========================
# 初始训练数据（你的真实数据）
# =========================
data = [
    ["100*70*2.5", 107.5, 75, 10.5, 278.5, 21.9, 0.0581, 0.0181],
    ["120*70*2.5", 130, 77, 12, 308, 25.5, 0.0666, 0.0186],
    ["150*70*2.5", 157, 75, 11, 329, 31.7, 0.1439, 0.0492],
    ["120*75", 126.6, 79.6, 6, 297.8, 18.15, 0.0208, 0.0078],
    ["140*75", 146.5, 79.7, 6, 317.9, 21.55, 0.0383, 0.0163],
    ["150*70*2.5_v2", 157.5, 75.2, 7, 321.9, 21.95, 0.0371, 0.0159],
    ["180*75*2.5", 187.5, 79.7, 7, 360.9, 28, 0.0523, 0.0295],
    ["200*75*2.5", 208, 79.8, 6, 379.6, 30.55, 0.1148, 0.0405],
]

cols = [
    "desk_size", "pkg_len", "pkg_wid", "pkg_hei",
    "girth", "weight", "complaint_rate", "loss_rate"
]

# =========================
# Session State：训练数据可动态扩展
# =========================
if "train_df" not in st.session_state:
    df = pd.DataFrame(data, columns=cols)
    df["len_ratio"] = df["pkg_len"] / df["girth"]
    st.session_state.train_df = df.copy()

# =========================
# 模型训练
# =========================
def train_loss_model(df):
    X = df[["weight", "girth", "len_ratio"]]
    y = logit(df["loss_rate"])
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()

def train_complaint_model(df):
    X = df[["weight", "girth", "len_ratio"]]
    y = logit(df["complaint_rate"])
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()

loss_model = train_loss_model(st.session_state.train_df)
complaint_model = train_complaint_model(st.session_state.train_df)

# =========================
# 页面配置
# =========================
st.set_page_config(page_title="包装运损风险评估工具", layout="centered")
st.title("📦 包装运损风险评估工具")

st.caption(
    f"当前训练样本数量：**{len(st.session_state.train_df)} 条** ｜ "
    "基于历史包装方案构建"
)

st.divider()

# =========================
# 包装方案评估区
# =========================
st.subheader("📖 输入待评估的包装方案")

col1, col2 = st.columns(2)
with col1:
    pkg_len = st.number_input("包装长 (cm)", value=160.0)
    pkg_wid = st.number_input("包装宽 (cm)", value=75.0)
with col2:
    pkg_hei = st.number_input("包装高 (cm)", value=7.0)
    weight = st.number_input("包装重量 (kg)", value=27.0)

if st.button("🔍 评估运损风险", use_container_width=True):
    girth = pkg_len + 2 * (pkg_wid + pkg_hei)
    len_ratio = pkg_len / girth

    X_new = pd.DataFrame({
        "const": [1],
        "weight": [weight],
        "girth": [girth],
        "len_ratio": [len_ratio]
    })

    pred_loss = inv_logit(loss_model.predict(X_new)[0])
    pred_complaint = inv_logit(complaint_model.predict(X_new)[0])

    if pred_loss < 0.015:
        level = "🟢 低风险"
    elif pred_loss < 0.03:
        level = "🟡 中风险"
    else:
        level = "🔴 高风险"

    st.subheader("✨评估结果")
    st.metric("预测运损资损率", f"{pred_loss*100:.2f}%")
    st.metric("预测运损客诉率（辅助）", f"{pred_complaint*100:.2f}%")
    st.markdown(f"**风险等级：{level}**")

    st.info(
        "风险判断依据："
        + (" 围长偏大；" if girth >= 330 else "")
        + (" 重量偏高；" if weight >= 25 else "")
        + (" 结构偏细长" if len_ratio >= 0.45 else " 结构整体可控")
    )

# =========================
# 模型解释面板
# =========================
with st.expander("模型系数解释（资损率模型）"):
    coef = loss_model.params
    st.write("**模型使用特征：重量、围长、长度占比**")
    st.markdown(
        f"""
- **重量系数：{coef['weight']:.3f}**  
  → 包装越重，一旦发生运损，实际资损越严重  

- **围长系数：{coef['girth']:.3f}**  
  → 包装外形越大，越容易进入运损风险区  

- **长度占比系数：{coef['len_ratio']:.3f}**  
  → 包装越细长，结构性运损风险越高
        """
    )

# =========================
# 新增训练数据接口
# =========================
st.divider()
st.subheader("➕ 新增一条训练数据（用于模型更新）")

with st.form("add_train_data"):
    desk = st.text_input("桌板尺寸标识")
    t_len = st.number_input("包装长(cm)", value=150.0)
    t_wid = st.number_input("包装宽(cm)", value=75.0)
    t_hei = st.number_input("包装高(cm)", value=7.0)
    t_weight = st.number_input("包装重量(kg)", value=25.0)
    t_complaint = st.number_input("运损客诉率(0-1)", value=0.05)
    t_loss = st.number_input("运损资损率(0-1)", value=0.02)

    submitted = st.form_submit_button("📥 添加并重新训练模型")

    if submitted:
        t_girth = t_len + 2 * (t_wid + t_hei)
        t_len_ratio = t_len / t_girth

        new_row = {
            "desk_size": desk,
            "pkg_len": t_len,
            "pkg_wid": t_wid,
            "pkg_hei": t_hei,
            "girth": t_girth,
            "weight": t_weight,
            "complaint_rate": t_complaint,
            "loss_rate": t_loss,
            "len_ratio": t_len_ratio
        }

        st.session_state.train_df = pd.concat(
            [st.session_state.train_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

        st.success(
            f"已添加新样本，当前训练样本数：{len(st.session_state.train_df)} 条"
        )
        st.experimental_rerun()

# =========================
# 风险声明
# =========================
st.divider()
st.caption(
    "⚠️ 本工具用于评估包装方案的相对运损风险水平，"
    "预测结果为区间性判断，不用于精确预测单一订单的实际资损结果。"
)
