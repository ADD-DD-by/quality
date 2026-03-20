# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="运损风险模型（验证版）", layout="wide")
st.title("📦 运损风险模型（训练 + 验证 + 预测）")

# =========================
# 全局配置
# =========================
FEATURES = [
    "girth","dim_weight","density",
    "len_ratio","max_edge","min_edge",
    "is_large","is_heavy",
    "L","W","H","weight",
    "LW","LH","WH","L2"
]

SEG_ORDER = ["small","medium","large"]

# =========================
# 初始化
# =========================
if "models" not in st.session_state:
    st.session_state.models = {}

# =========================
# 工具函数
# =========================
def parse(x):
    if isinstance(x, str) and "%" in x:
        return float(x.replace("%","")) / 100
    return float(x)

def segment(x):
    if x <= 120:
        return "small"
    elif x <= 160:
        return "medium"
    else:
        return "large"

def base_rule(row):
    score = 0.0035
    if row["L"] > 150: score += 0.01
    if row["girth"] > 300: score += 0.01
    if row["weight"] > 30: score += 0.01
    return score

def add_features(df):
    df["V"] = df["L"] * df["W"] * df["H"]
    df["dim_weight"] = df["V"] / 5000
    df["density"] = df["weight"] / (df["V"] + 1e-6)
    df["len_ratio"] = df["L"] / (df["W"] + df["H"] + 1e-6)

    df["max_edge"] = df[["L","W","H"]].max(axis=1)
    df["min_edge"] = df[["L","W","H"]].min(axis=1)

    df["is_large"] = (df["L"] > 150).astype(int)
    df["is_heavy"] = (df["weight"] > 25).astype(int)

    df["LW"] = df["L"] * df["W"]
    df["LH"] = df["L"] * df["H"]
    df["WH"] = df["W"] * df["H"]
    df["L2"] = df["L"]**2

    return df

# =========================
# 训练模型
# =========================
def train_model(df):

    df = df.sort_values(["L","W","H","weight"]).reset_index(drop=True)
    df = add_features(df)
    df["segment"] = df["L"].apply(segment)

    df["rule_c"] = df.apply(base_rule, axis=1)
    df["rule_l"] = df["rule_c"] + (df["weight"] > 30) * 0.01

    models = {}

    for seg in SEG_ORDER:

        sub = df[df["segment"] == seg].copy()
        if len(sub) < 5:
            continue

        X = sub[FEATURES].copy()
        X["const"] = 1
        X = X[["const"] + FEATURES]

        sub["ratio_c"] = sub["complaint_rate"] / (sub["rule_c"] + 1e-6)
        sub["ratio_l"] = sub["loss_rate"] / (sub["rule_l"] + 1e-6)

        sub["ratio_c"] = np.clip(sub["ratio_c"], 0.1, 5)
        sub["ratio_l"] = np.clip(sub["ratio_l"], 0.1, 5)

        model_c = sm.OLS(sub["ratio_c"], X).fit()
        model_l = sm.OLS(sub["ratio_l"], X).fit()

        models[seg] = (model_c, model_l)

    return models, df

# =========================
# 预测函数
# =========================
def predict(df, models):

    df = df.copy()
    df = add_features(df)
    df["segment"] = df["L"].apply(segment)

    df["rule_c"] = df.apply(base_rule, axis=1)
    df["rule_l"] = df["rule_c"] + (df["weight"] > 30) * 0.01

    preds = []

    for seg in SEG_ORDER:

        sub = df[df["segment"] == seg].copy()
        if len(sub) == 0:
            continue

        if seg not in models:
            sub["pred_c"] = sub["rule_c"]
            sub["pred_l"] = sub["rule_l"]
        else:
            model_c, model_l = models[seg]

            X = sub[FEATURES].copy()
            X["const"] = 1
            X = X[["const"] + FEATURES]

            ratio_c = np.clip(model_c.predict(X),0.1,3)
            ratio_l = np.clip(model_l.predict(X),0.1,3)

            sub["pred_c"] = sub["rule_c"] * ratio_c
            sub["pred_l"] = sub["rule_l"] * ratio_l

        preds.append(sub)

    return pd.concat(preds)

# =========================
# 📥 上传训练数据
# =========================
st.sidebar.header("📥 上传训练数据")

train_file = st.sidebar.file_uploader("上传训练Excel", type=["xlsx"])

if train_file:

    df = pd.read_excel(train_file)

    df = df.rename(columns={
        "产品-长": "L",
        "产品-宽": "W",
        "包装-高": "H",
        "包装-重": "weight",
        "围长": "girth",
        "运损客诉率": "complaint_rate",
        "运损资损率": "loss_rate"
    })

    df["complaint_rate"] = df["complaint_rate"].apply(parse)
    df["loss_rate"] = df["loss_rate"].apply(parse)

    if st.button("🚀 训练模型"):

        models, df_train = train_model(df)
        st.session_state.models = models

        st.success("模型训练完成")

        # =========================
        # 🔥 拟合验证
        # =========================
        df_pred = predict(df_train, models)

        df_pred["error_c"] = df_pred["pred_c"] - df_pred["complaint_rate"]
        df_pred["error_l"] = df_pred["pred_l"] - df_pred["loss_rate"]

        st.subheader("📊 模型评估")

        col1, col2 = st.columns(2)

        col1.metric("客诉 MAE", round(np.mean(np.abs(df_pred["error_c"])),4))
        col2.metric("资损 MAE", round(np.mean(np.abs(df_pred["error_l"])),4))

        # 图
        fig, ax = plt.subplots()
        ax.scatter(df_pred["complaint_rate"], df_pred["pred_c"], label="Complaint")
        ax.scatter(df_pred["loss_rate"], df_pred["pred_l"], label="Loss")
        ax.plot([0,0.2],[0,0.2],'r--')
        ax.legend()
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")

        st.pyplot(fig)

        # Top误差
        st.subheader("⚠️ 最大误差样本")
        st.dataframe(
            df_pred.sort_values("error_c", key=np.abs, ascending=False).head(10)
        )

# =========================
# 📤 上传预测数据
# =========================
st.subheader("📤 方案预测")

pred_file = st.file_uploader("上传预测Excel", type=["xlsx"])

if pred_file and st.session_state.models:

    pred_df = pd.read_excel(pred_file)

    pred_df = pred_df.rename(columns={
        "产品-长": "L",
        "产品-宽": "W",
        "包装-高": "H",
        "包装-重": "weight",
        "围长": "girth"
    })

    if "评估方案" not in pred_df.columns:
        pred_df["评估方案"] = "方案"

    result = predict(pred_df, st.session_state.models)

    output = result[[
        "评估方案","L","W","H",
        "pred_c","pred_l"
    ]]

    st.dataframe(output)

    # 推荐方案
    output["score"] = 0.5*output["pred_c"] + 0.5*output["pred_l"]
    best = output.sort_values("score").iloc[0]

    st.success(f"🏆 推荐方案：{best['评估方案']}")

    st.download_button(
        "下载预测结果",
        output.to_csv(index=False),
        file_name="预测结果.csv"
    )
