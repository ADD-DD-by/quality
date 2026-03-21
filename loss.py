import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="包装风险评估工具", layout="wide")

st.title("📦 包装风险评估工具（客诉率 & 资损率）")

# =========================
# 1️⃣ 读取模型（你需要先训练一次保存coef）
# =========================

@st.cache_data
def load_model():
    df = pd.read_excel("运损对比.xlsx")

    df = df.rename(columns={
        "包装长": "L",
        "包装宽": "W",
        "包装高": "H",
        "包装重": "weight",
        "围长": "girth",
        "包装系数": "pack_coef",
        "运损客诉率": "claim_rate",
        "运损资损率": "loss_rate"
    })

    def parse_percent(x):
        if isinstance(x, str):
            return float(x.replace("%","")) / 100
        elif pd.notnull(x):
            return x/100 if x > 1 else x
        return np.nan

    df["claim_rate"] = df["claim_rate"].apply(parse_percent)
    df["loss_rate"] = df["loss_rate"].apply(parse_percent)

    # 特征
    df["log_girth"] = np.log(df["girth"])
    df["log_weight"] = np.log(df["weight"])
    df["log_H"] = np.log(df["H"])

    df["girth_excess_260"] = np.maximum(0, df["girth"] - 260)
    df["girth_excess_300"] = np.maximum(0, df["girth"] - 300)

    df["weight_excess_30"] = np.maximum(0, df["weight"] - 30)
    df["weight_excess_40"] = np.maximum(0, df["weight"] - 40)

    df["girth_pack_penalty"] = df["girth_excess_300"] * df["pack_coef"]
    # =========================
# ⭐ 新增补偿特征（核心）
# =========================

    df["girth_comp"] = np.log1p(np.maximum(0, df["girth"] - 350)) * df["pack_coef"]
    FEATURES = [
        "log_girth","log_weight","log_H","pack_coef",
        "girth_excess_260","girth_excess_300",
        "weight_excess_30","weight_excess_40",
        "girth_pack_penalty","girth_comp"
    ]

    X = sm.add_constant(df[FEATURES])

    model_claim = sm.OLS(df["claim_rate"], X).fit()
    model_loss = sm.OLS(df["loss_rate"], X).fit()

    return model_claim.params, model_loss.params


coef_claim, coef_loss = load_model()

# =========================
# 2️⃣ 核心预测函数
# =========================
def predict_row(row, coef):
    girth = row["girth"]
    weight = row["weight"]
    H = row["H"]
    pack = row["pack_coef"]

    # =========================
    # 原模型（完全不动！！）
    # =========================
    val = (
        coef['const']
        + coef['log_girth'] * np.log(girth)
        + coef['log_weight'] * np.log(weight)
        + coef['log_H'] * np.log(H)
        + coef['pack_coef'] * pack
        + coef['girth_excess_260'] * max(0, girth-260)
        + coef['girth_excess_300'] * max(0, girth-300)
        + coef['weight_excess_30'] * max(0, weight-30)
        + coef['weight_excess_40'] * max(0, weight-40)
        + coef['girth_pack_penalty'] * max(0, girth-300) * pack
        + coef['girth_comp'] * np.log1p(max(0, girth-400)) * pack
    )

    # ======================

    # =========================
    # 最终结果
    # =========================
    pred = val + compensation

    # 保留你原来的clip（保证训练数据一致）
    return np.clip(pred, 0, 0.5)

# =========================
# 3️⃣ 单条预测
# =========================
st.subheader("🔹 单条预测")

col1, col2, col3, col4 = st.columns(4)

L = col1.number_input("包装长", value=120.0)
W = col2.number_input("包装宽", value=80.0)
H = col3.number_input("包装高", value=10.0)
weight = col4.number_input("包装重", value=35.0)

girth = st.number_input("围长", value=280.0)
pack_coef = st.number_input("包装系数", value=1.0)

if st.button("计算风险"):
    row = {
        "L": L,
        "W": W,
        "H": H,
        "weight": weight,
        "girth": girth,
        "pack_coef": pack_coef
    }

    claim = predict_row(row, coef_claim)
    loss = predict_row(row, coef_loss)

    st.success(f"📊 客诉率：{claim:.2%}")
    st.success(f"💰 资损率：{loss:.2%}")

    # 风险等级
    if claim < 0.01:
        level = "低风险"
    elif claim < 0.03:
        level = "中风险"
    else:
        level = "高风险"

    st.info(f"🚦 风险等级：{level}")

# =========================
# 4️⃣ 批量预测
# =========================
st.subheader("🔹 批量预测（上传Excel,数据表列名：包装长、包装宽、包装高、包装重、围长、包装系数）")

file = st.file_uploader("上传数据", type=["xlsx"])

if file:
    df_input = pd.read_excel(file)

    df_input = df_input.rename(columns={
        "包装长": "L",
        "包装宽": "W",
        "包装高": "H",
        "包装重": "weight",
        "围长": "girth",
        "包装系数": "pack_coef"
    })

    df_input["pred_claim"] = df_input.apply(lambda r: predict_row(r, coef_claim), axis=1)
    df_input["pred_loss"] = df_input.apply(lambda r: predict_row(r, coef_loss), axis=1)

    st.dataframe(df_input)

    st.download_button(
        "下载结果",
        df_input.to_csv(index=False).encode("utf-8"),
        "预测结果.csv"
    )
