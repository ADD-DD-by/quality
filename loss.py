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
# 初始训练数据（体积 + 重量）
# =========================
data = [
    [100*70*10.5, 21.9, 0.0181, 0.0581],
    [120*75*6,   18.15, 0.0078, 0.0208],
    [120*70*12,  25.5,  0.0186, 0.0666],
    [140*75*6,   21.55, 0.0163, 0.0383],
    [150*70*7,   21.95, 0.0159, 0.0371],
    [150*70*11,  31.75, 0.0492, 0.1439],
    [180*75*7,   28.0,  0.0295, 0.0523],
    [200*75*6,   30.55, 0.0405, 0.1148],
]

cols = [
    "V",              # 体积
    "weight",         # 重量
    "loss_rate",      # 资损率
    "complaint_rate", # 客诉率
]

FEATURE_COLS = ["V", "weight"]


# =========================
# Session State
# =========================
if "train_df" not in st.session_state:
    df = pd.DataFrame(data, columns=cols)
    st.session_state.train_df = df.copy()


# =========================
# 模型训练
# =========================
def train_loss_model(df):
    X = sm.add_constant(df[FEATURE_COLS])
    y = logit(df["loss_rate"])
    return sm.OLS(y, X).fit()

def train_complaint_model(df):
    X = sm.add_constant(df[FEATURE_COLS])
    y = logit(df["complaint_rate"])
    return sm.OLS(y, X).fit()


loss_model = train_loss_model(st.session_state.train_df)
complaint_model = train_complaint_model(st.session_state.train_df)


# =========================
# 页面
# =========================
st.set_page_config(page_title="包装运损风险评估工具", layout="centered")
st.title("📦 包装运损风险评估工具")

st.caption(
    f"当前训练样本数量：**{len(st.session_state.train_df)} 条** ｜ "
    "模型特征：体积 + 重量（Logit-OLS）"
)

st.divider()


# =========================
# 输入区
# =========================
st.subheader("📖 输入待评估的包装方案")

col1, col2 = st.columns(2)
with col1:
    L = st.number_input("长 (cm)", value=160.0)
    W = st.number_input("宽 (cm)", value=75.0)
with col2:
    H = st.number_input("高 (cm)", value=7.0)
    weight = st.number_input("重量 (kg)", value=27.0)


if st.button("🔍 评估运损风险", use_container_width=True):

    V = L * W * H

    X_new = pd.DataFrame([{
        "const": 1,
        "V": V,
        "weight": weight
    }])

    pred_loss = inv_logit(loss_model.predict(X_new)[0])
    pred_complaint = inv_logit(complaint_model.predict(X_new)[0])

    # 业务规则（非模型）
    girth = L + 2 * (W + H)
    len_ratio = L / girth

    if pred_loss < 0.015:
        level = "🟢 低风险"
    elif pred_loss < 0.03:
        level = "🟡 中风险"
    else:
        level = "🔴 高风险"

    st.subheader("✨ 评估结果")
    st.metric("预测运损资损率", f"{pred_loss*100:.2f}%")
    st.metric("预测运损客诉率（辅助）", f"{pred_complaint*100:.2f}%")
    st.markdown(f"**风险等级：{level}**")

    st.info(
        "结构风险提示："
        + (" 围长偏大；" if girth >= 330 else "")
        + (" 重量偏高；" if weight >= 25 else "")
        + (" 结构偏细长" if len_ratio >= 0.45 else " 结构整体可控")
    )


# =========================
# 模型解释
# =========================
with st.expander("📊 模型系数解释（资损率模型）"):
    coef = loss_model.params

    st.caption("注：模型在 logit 空间训练，系数表示对风险对数几率的影响")

    st.markdown(
        f"""
- **体积系数：{coef['V']:.6f}**  
  → 包装越大，运输过程中的系统性风险越高  

- **重量系数：{coef['weight']:.3f}**  
  → 包装越重，一旦发生破损，资损程度越高
        """
    )


# =========================
# 新增训练数据
# =========================
st.divider()
st.subheader("➕ 新增训练样本")

with st.form("add_train"):
    t_L = st.number_input("长(cm)", value=150.0)
    t_W = st.number_input("宽(cm)", value=75.0)
    t_H = st.number_input("高(cm)", value=7.0)
    t_weight = st.number_input("重量(kg)", value=25.0)
    t_loss = st.number_input("资损率(0-1)", value=0.02)
    t_complaint = st.number_input("客诉率(0-1)", value=0.05)

    if st.form_submit_button("📥 添加并重训"):
        t_V = t_L * t_W * t_H

        new_row = {
            "V": t_V,
            "weight": t_weight,
            "loss_rate": t_loss,
            "complaint_rate": t_complaint,
        }

        st.session_state.train_df = pd.concat(
            [st.session_state.train_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

        st.success(f"样本已添加，总数：{len(st.session_state.train_df)}")
        st.experimental_rerun()


# =========================
# 声明
# =========================
st.divider()
st.caption(
    "⚠️ 本工具用于评估包装方案在不同体积与重量条件下的**相对资损/运损风险**，"
    "不用于精确预测单一订单的实际损失。"
)
