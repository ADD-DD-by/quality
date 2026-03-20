# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="运损风险预测工具", layout="wide")
st.title("📦 运损风险预测工具（乘法残差模型）")

# =========================
# 全局配置
# =========================
FEATURES = [
    "girth","dim_weight","density",
    "len_ratio","max_edge","min_edge",
    "is_large","is_heavy"
]

MODEL_COLUMNS = ["const"] + FEATURES
SEG_ORDER = ["small","medium","large"]

# =========================
# 初始化
# =========================
if "models_dict" not in st.session_state:
    st.session_state.models_dict = {}

# =========================
# 工具函数
# =========================
def parse(x):
    if isinstance(x, str) and "%" in x:
        return float(x.replace("%","")) / 100
    return float(x)

def build_X(df):
    df = df.copy()
    df["const"] = 1.0

    # 防止缺列
    for col in MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df[MODEL_COLUMNS]

def feature_engineering(df):
    df["V"] = df["L"] * df["W"] * df["H"]
    df["dim_weight"] = df["V"] / 5000
    df["density"] = df["weight"] / (df["V"] + 1e-6)
    df["len_ratio"] = df["L"] / (df["W"] + df["H"] + 1e-6)

    df["max_edge"] = df[["L","W","H"]].max(axis=1)
    df["min_edge"] = df[["L","W","H"]].min(axis=1)

    df["is_large"] = (df["L"] > 150).astype(int)
    df["is_heavy"] = (df["weight"] > 25).astype(int)
    return df

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

def preprocess_pred_df(df):
    df = df.rename(columns={
        "产品-长": "L",
        "产品-宽": "W",
        "包装-高": "H",
        "包装-重": "weight",
        "围长": "girth"
    })

    if "评估方案" not in df.columns:
        df["评估方案"] = "默认方案"

    numeric_cols = ["L","W","H","weight","girth"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    return df

# =========================
# 训练模型（乘法🔥）
# =========================
def train_model(df):

    df = df.sort_values(["L","W","H","weight"]).reset_index(drop=True)

    df = feature_engineering(df)
    df["segment"] = df["L"].apply(segment)

    df["rule_c"] = df.apply(base_rule, axis=1)
    df["rule_l"] = df["rule_c"] + (df["weight"] > 30) * 0.01

    models_dict = {}

    for seg in SEG_ORDER:

        sub = df[df["segment"] == seg].copy()
        if len(sub) < 5:
            continue

        X = build_X(sub)

        sub["ratio_c"] = sub["complaint_rate"] / (sub["rule_c"] + 1e-6)
        sub["ratio_l"] = sub["loss_rate"] / (sub["rule_l"] + 1e-6)

        sub["ratio_c"] = np.clip(sub["ratio_c"], 0.1, 5)
        sub["ratio_l"] = np.clip(sub["ratio_l"], 0.1, 5)

        model_c = sm.OLS(sub["ratio_c"], X).fit()
        model_l = sm.OLS(sub["ratio_l"], X).fit()

        models_dict[seg] = (model_c, model_l)

    return models_dict

# =========================
# 预测
# =========================
def predict(df, models_dict):

    df = df.sort_values(["L","W","H","weight"]).reset_index(drop=True)

    df = feature_engineering(df)
    df["segment"] = df["L"].apply(segment)

    df["rule_c"] = df.apply(base_rule, axis=1)
    df["rule_l"] = df["rule_c"] + (df["weight"] > 30) * 0.01

    results = []

    for seg in SEG_ORDER:

        sub = df[df["segment"] == seg].copy()
        if len(sub) == 0:
            continue

        if seg not in models_dict:
            sub["complaint_pred"] = sub["rule_c"]
            sub["loss_pred"] = sub["rule_l"]
        else:
            model_c, model_l = models_dict[seg]

            X = build_X(sub)

            ratio_c = model_c.predict(X)
            ratio_l = model_l.predict(X)

            ratio_c = np.clip(ratio_c, 0.1, 3)
            ratio_l = np.clip(ratio_l, 0.1, 3)

            sub["complaint_pred"] = sub["rule_c"] * ratio_c
            sub["loss_pred"] = sub["rule_l"] * ratio_l

        results.append(sub)

    return pd.concat(results)

# =========================
# UI - 训练
# =========================
st.sidebar.header("📥 训练模型")

train_file = st.sidebar.file_uploader("上传训练数据", type=["xlsx"])

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
        st.session_state.models_dict = train_model(df)
        st.success("模型训练完成")

# =========================
# UI - 预测
# =========================
st.subheader("📤 上传预测数据")

pred_file = st.file_uploader("上传预测Excel", type=["xlsx"])

if pred_file and st.session_state.models_dict:

    pred_df = pd.read_excel(pred_file)
    pred_df = preprocess_pred_df(pred_df)

    result = predict(pred_df, st.session_state.models_dict)

    output = result[[
        "评估方案",
        "L","W","H","weight","girth",
        "complaint_pred","loss_pred"
    ]]

    st.dataframe(output)

    # ✅ 最优方案
    output["score"] = 0.5*output["complaint_pred"] + 0.5*output["loss_pred"]
    best = output.sort_values("score").iloc[0]

    st.success(f"🏆 推荐方案：{best['评估方案']}")

    st.download_button(
        "下载结果",
        output.to_csv(index=False),
        file_name="预测结果.csv"
    )
