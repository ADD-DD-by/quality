import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="包装风险模型工具", layout="wide")
st.title("📦 运损风险预测工具")

# =========================
# 初始化 session
# =========================
if "models_dict" not in st.session_state:
    st.session_state.models_dict = {}

if "model_versions" not in st.session_state:
    st.session_state.model_versions = []

# =========================
# 工具函数
# =========================
def parse(x):
    if isinstance(x, str) and "%" in x:
        return float(x.replace("%","")) / 100
    return float(x)

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

# =========================
# 模型训练
# =========================
def train_model(df):

    df = feature_engineering(df)
    df["segment"] = df["L"].apply(segment)

    df["rule_c"] = df.apply(base_rule, axis=1)
    df["rule_l"] = df["rule_c"] + (df["weight"] > 30) * 0.01

    models_dict = {}
    model_info = []

    for seg in df["segment"].unique():

        sub = df[df["segment"] == seg].copy()
        if len(sub) < 5:
            continue

        features = [
            "girth","dim_weight","density",
            "len_ratio","max_edge","min_edge",
            "is_large","is_heavy"
        ]

        X = sm.add_constant(sub[features])

        sub["residual_c"] = sub["complaint_rate"] - sub["rule_c"]
        sub["residual_l"] = sub["loss_rate"] - sub["rule_l"]

        model_c = sm.OLS(sub["residual_c"], X).fit()
        model_l = sm.OLS(sub["residual_l"], X).fit()

        sub["model_c"] = model_c.predict(X)
        sub["model_l"] = model_l.predict(X)

        fusion_input_c = np.vstack([sub["rule_c"], sub["model_c"]]).T
        fusion_input_l = np.vstack([sub["rule_l"], sub["model_l"]]).T

        fusion_model_c = LinearRegression().fit(fusion_input_c, sub["complaint_rate"])
        fusion_model_l = LinearRegression().fit(fusion_input_l, sub["loss_rate"])

        models_dict[seg] = (model_c, model_l, fusion_model_c, fusion_model_l)

        model_info.append({
            "segment": seg,
            "客诉_rule权重": fusion_model_c.coef_[0],
            "客诉_model权重": fusion_model_c.coef_[1],
            "资损_rule权重": fusion_model_l.coef_[0],
            "资损_model权重": fusion_model_l.coef_[1],
        })

    return models_dict, pd.DataFrame(model_info)

# =========================
# 预测函数
# =========================
def predict(df, models_dict):

    df = feature_engineering(df)
    df["segment"] = df["L"].apply(segment)

    df["rule_c"] = df.apply(base_rule, axis=1)
    df["rule_l"] = df["rule_c"] + (df["weight"] > 30) * 0.01

    features = [
        "girth","dim_weight","density",
        "len_ratio","max_edge","min_edge",
        "is_large","is_heavy"
    ]

    results = []

    for seg in df["segment"].unique():

        sub = df[df["segment"] == seg].copy()

        if seg not in models_dict:
            sub["complaint_pred"] = sub["rule_c"]
            sub["loss_pred"] = sub["rule_l"]
        else:
            model_c, model_l, fusion_c, fusion_l = models_dict[seg]

            X = sm.add_constant(sub[features])

            sub["model_c"] = model_c.predict(X)
            sub["model_l"] = model_l.predict(X)

            fusion_input_c = np.vstack([sub["rule_c"], sub["model_c"]]).T
            fusion_input_l = np.vstack([sub["rule_l"], sub["model_l"]]).T

            sub["complaint_pred"] = fusion_c.predict(fusion_input_c)
            sub["loss_pred"] = fusion_l.predict(fusion_input_l)

        results.append(sub)

    return pd.concat(results)

# =========================
# 📥 训练数据上传
# =========================
st.sidebar.header("📥 上传训练数据")
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

        models_dict, model_info = train_model(df)

        st.session_state.models_dict = models_dict
        st.session_state.model_versions.append(models_dict)

        st.success("模型训练完成")

        st.subheader("📊 模型参数说明")
        st.dataframe(model_info)

# =========================
# 🔄 回滚
# =========================
st.sidebar.subheader("🔄 模型回滚")

if st.sidebar.button("回滚上一个版本"):
    if len(st.session_state.model_versions) > 1:
        st.session_state.model_versions.pop()
        st.session_state.models_dict = st.session_state.model_versions[-1]
        st.success("已回滚")
    else:
        st.warning("没有可回滚版本")

# =========================
# 📤 预测
# =========================
st.subheader("📤 上传预测数据")
pred_file = st.file_uploader("上传需要预测的Excel", type=["xlsx"])

if pred_file and st.session_state.models_dict:

    pred_df = pd.read_excel(pred_file)

    pred_df = pred_df.rename(columns={
        "产品-长": "L",
        "产品-宽": "W",
        "包装-高": "H",
        "包装-重": "weight",
        "围长": "girth"
    })

    result = predict(pred_df, st.session_state.models_dict)

    st.dataframe(result)

    st.download_button(
        "下载预测结果",
        result.to_csv(index=False),
        file_name="预测结果.csv"
    )
