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
# 初始训练数据（真实历史数据）
# =========================
data = [
    [100, 70, 10.5, 21.9, 0.0181, 0.0581, 73500.0, 261.0],
    [120, 75, 6,   18.15, 0.0078, 0.0208, 54000.0, 282.0],
    [120, 70, 12,  25.5,  0.0186, 0.0666, 100800.0, 284.0],
    [140, 75, 6,   21.55, 0.0163, 0.0383, 63000.0, 302.0],
    [150, 70, 7,   21.95, 0.0159, 0.0371, 73500.0, 304.0],
    [150, 70, 11,  31.75, 0.0492, 0.1439, 115500.0, 312.0],
    [180, 75, 7,   28.0,  0.0295, 0.0523, 94500.0, 344.0],
    [200, 75, 6,   30.55, 0.0405, 0.1148, 90000.0, 362.0],
]

cols = [
    "L",              # 长
    "W",              # 宽
    "H",              # 厚
    "weight",         # 重量(kg)
    "loss_rate",      # 资损率
    "complaint_rate", # 运损率
    "V",              # 体积
    "girth",          # 围长
]

FEATURE_COLS = ["weight", "L", "W", "H"]


# =========================
# Session State：初始化训练数据
# =========================
if "train_df" not in st.session_state:
    df = pd.DataFrame(data, columns=cols)
    df["len_ratio"] = df["L"] / df["girth"]
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
# 页面配置
# =========================
st.set_page_config(page_title="包装运损风险评估工具", layout="centered")
st.title("📦 包装运损风险评估工具")

st.caption(
    f"当前训练样本数量：**{len(st.session_state.train_df)} 条** ｜ "
    "基于历史包装方案构建（结构性风险模型）"
)

if len(st.session_state.train_df) < 5:
    st.warning("⚠️ 当前训练样本较少，模型稳定性有限")

st.divider()


# =========================
# 包装方案评估区
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
    girth = L + 2 * (W + H)
    len_ratio = L / girth
    V = L * W * H

    X_new = pd.DataFrame([{
        "const": 1,
        "weight": weight,
        "girth": girth,
        "len_ratio": len_ratio,
        "V": V
    }])

    pred_loss = inv_logit(loss_model.predict(X_new)[0])
    pred_complaint = inv_logit(complaint_model.predict(X_new)[0])

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
        "风险判断依据："
        + (" 围长偏大；" if girth >= 330 else "")
        + (" 重量偏高；" if weight >= 25 else "")
        + (" 结构偏细长" if len_ratio >= 0.45 else " 结构整体可控")
    )


# =========================
# 模型解释面板
# =========================
with st.expander("📊 模型系数解释（资损率模型）"):
    coef = loss_model.params
    st.write("**模型特征：重量 / 围长 / 长度占比 / 体积**")
    st.caption("注：模型在 logit 空间训练，系数表示对风险“对数几率”的影响")

    st.markdown(
        f"""
- **重量系数：{coef['weight']:.3f}**  
  → 包装越重，发生运损后的资损风险越高  

- **围长系数：{coef['girth']:.3f}**  
  → 外形越大，进入高风险运输区间的概率越高  

- **长度占比系数：{coef['len_ratio']:.3f}**  
  → 结构越细长，结构性运损风险越明显  

- **体积系数：{coef['V']:.6f}**  
  → 体积对风险有系统性影响
        """
    )


# =========================
# 新增训练数据接口
# =========================
st.divider()
st.subheader("➕ 新增一条训练数据（用于模型更新）")

with st.form("add_train_data"):
    t_len = st.number_input("长(cm)", value=150.0)
    t_wid = st.number_input("宽(cm)", value=75.0)
    t_hei = st.number_input("高(cm)", value=7.0)
    t_weight = st.number_input("重量(kg)", value=25.0)
    t_loss = st.number_input("运损资损率(0-1)", value=0.02)
    t_complaint = st.number_input("运损客诉率(0-1)", value=0.05)

    submitted = st.form_submit_button("📥 添加并重新训练模型")

    if submitted:
        t_girth = t_len + 2 * (t_wid + t_hei)
        t_len_ratio = t_len / t_girth
        t_V = t_len * t_wid * t_hei

        new_row = {
            "L": t_len,
            "W": t_wid,
            "H": t_hei,
            "weight": t_weight,
            "loss_rate": t_loss,
            "complaint_rate": t_complaint,
            "V": t_V,
            "girth": t_girth,
            "len_ratio": t_len_ratio,
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
    "⚠️ 本工具用于评估不同包装结构方案的**相对运损风险水平**，"
    "预测结果为区间性判断，不用于精确预测单一订单的实际资损结果。"
)
