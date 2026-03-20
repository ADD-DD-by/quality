import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from datetime import datetime
from io import BytesIO

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
# 初始数据
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

cols = ["V", "weight", "loss_rate", "complaint_rate"]
FEATURE_COLS = ["V", "weight"]

# =========================
# 版本管理路径
# =========================
DATA_DIR = "data"
HISTORY_DIR = os.path.join(DATA_DIR, "history")

os.makedirs(HISTORY_DIR, exist_ok=True)

# =========================
# 数据读写
# =========================
def load_data():
    path = os.path.join(DATA_DIR, "current.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        df = pd.DataFrame(data, columns=cols)
        df["V"] = df["V"] / 1000
        return df

def save_data(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存历史版本
    history_path = os.path.join(HISTORY_DIR, f"train_{timestamp}.csv")
    df.to_csv(history_path, index=False)

    # 保存当前版本
    current_path = os.path.join(DATA_DIR, "current.csv")
    df.to_csv(current_path, index=False)


# =========================
# Session 初始化
# =========================
if "train_df" not in st.session_state:
    st.session_state.train_df = load_data()


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


# =========================
# 页面
# =========================
st.set_page_config(page_title="包装运损风险评估工具", layout="centered")
st.title("📦 包装运损风险评估工具")

st.caption(f"当前训练样本：{len(st.session_state.train_df)} 条")

# =========================
# 模型训练 & 系数展示
# =========================
loss_model = train_loss_model(st.session_state.train_df)
complaint_model = train_complaint_model(st.session_state.train_df)

st.subheader("📊 当前模型系数")

coef_df = pd.DataFrame({
    "变量": loss_model.params.index,
    "资损率模型系数": loss_model.params.values,
    "客诉率模型系数": complaint_model.params.values
})

st.dataframe(coef_df)


# =========================
# 单条预测
# =========================
st.divider()
st.subheader("📖 单条评估")

col1, col2 = st.columns(2)
with col1:
    L = st.number_input("长", value=160.0)
    W = st.number_input("宽", value=75.0)
with col2:
    H = st.number_input("高", value=7.0)
    weight = st.number_input("重量", value=27.0)

if st.button("🔍 评估"):

    V = (L * W * H) / 1000

    X = pd.DataFrame([{"const":1,"V":V,"weight":weight}])

    loss = inv_logit(loss_model.predict(X)[0])
    complaint = inv_logit(complaint_model.predict(X)[0])

    st.metric("资损率", f"{loss*100:.2f}%")
    st.metric("客诉率", f"{complaint*100:.2f}%")


# =========================
# 上传训练数据
# =========================
st.divider()
st.subheader("📤 上传训练数据")

file = st.file_uploader("上传训练数据", type=["xlsx"])

if file and st.button("导入训练数据"):

    df = pd.read_excel(file)

    df = df.rename(columns={
        "产品-长":"L","产品-宽":"W","包装-高":"H",
        "包装-重":"weight",
        "运损资损率":"loss_rate",
        "运损客诉率":"complaint_rate"
    })

    def parse(x):
        if isinstance(x,str) and "%" in x:
            return float(x.replace("%",""))/100
        return float(x)

    df["loss_rate"] = df["loss_rate"].apply(parse)
    df["complaint_rate"] = df["complaint_rate"].apply(parse)

    df["V"] = (df["L"]*df["W"]*df["H"])/1000
    df = df[["V","weight","loss_rate","complaint_rate"]]

    st.session_state.train_df = pd.concat(
        [st.session_state.train_df, df], ignore_index=True
    )

    save_data(st.session_state.train_df)

    st.success("导入成功")


# =========================
# 批量预测
# =========================
st.divider()
st.subheader("📊 批量预测")

file2 = st.file_uploader("上传预测数据", type=["xlsx"], key="pred")

if file2 and st.button("开始预测"):

    df = pd.read_excel(file2)

    df = df.rename(columns={
        "长":"L","宽":"W","高":"H","重量":"weight"
    })

    df["V"] = (df["L"]*df["W"]*df["H"])/1000

    X = sm.add_constant(df[["V","weight"]])

    df["预测资损率"] = inv_logit(loss_model.predict(X))
    df["预测客诉率"] = inv_logit(complaint_model.predict(X))

    st.dataframe(df)

    buffer = BytesIO()
    df.to_excel(buffer,index=False)

    st.download_button("下载结果", buffer.getvalue(), "预测结果.xlsx")


# =========================
# 版本管理
# =========================
st.divider()
st.subheader("🕘 历史版本")

files = sorted(os.listdir(HISTORY_DIR), reverse=True)

if files:
    f = st.selectbox("选择版本", files)

    if st.button("回滚"):
        df = pd.read_csv(os.path.join(HISTORY_DIR, f))
        st.session_state.train_df = df
        save_data(df)
        st.success("已回滚")


# =========================
# 下载当前训练数据
# =========================
buffer = BytesIO()
st.session_state.train_df.to_excel(buffer,index=False)

st.download_button("📥 下载当前训练数据", buffer.getvalue(), "train_data.xlsx")


# =========================
# 声明
# =========================
st.caption("⚠️ 模型用于相对风险评估")
