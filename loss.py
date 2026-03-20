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
# 路径
# =========================
DATA_DIR = "data"
HISTORY_DIR = os.path.join(DATA_DIR, "history")

os.makedirs(HISTORY_DIR, exist_ok=True)


# =========================
# 初始数据（8条）
# =========================
init_data = [
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


# =========================
# 数据读写
# =========================
def load_data():
    path = os.path.join(DATA_DIR, "current.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        df = pd.DataFrame(init_data, columns=cols)
        df["V"] = df["V"] / 1000
        return df


def save_data(df, tag="normal"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{timestamp}_{tag}.csv"

    # 历史版本
    df.to_csv(os.path.join(HISTORY_DIR, filename), index=False)

    # 当前版本
    df.to_csv(os.path.join(DATA_DIR, "current.csv"), index=False)


# =========================
# 初始化（关键修复🔥）
# =========================
if "train_df" not in st.session_state:

    df = load_data()
    st.session_state.train_df = df

    # ⭐ 如果历史为空 → 写入初始版本
    if len(os.listdir(HISTORY_DIR)) == 0:
        save_data(df, tag="init")


# =========================
# 模型
# =========================
def train_model(df, target):
    X = sm.add_constant(df[["V", "weight"]])
    y = logit(df[target])
    return sm.OLS(y, X).fit()


loss_model = train_model(st.session_state.train_df, "loss_rate")
complaint_model = train_model(st.session_state.train_df, "complaint_rate")


# =========================
# 页面
# =========================
st.title("📦 包装运损风险评估工具")
st.caption(f"当前样本数：{len(st.session_state.train_df)}")


# =========================
# 🔥 一键重置（新增）
# =========================
if st.button("🧱 重置为初始8条数据"):
    df_init = pd.DataFrame(init_data, columns=cols)
    df_init["V"] = df_init["V"] / 1000

    st.session_state.train_df = df_init
    save_data(df_init, tag="reset")

    st.success("已恢复到初始版本（8条）")


# =========================
# 模型系数
# =========================
st.subheader("📊 模型系数")

coef_df = pd.DataFrame({
    "变量": loss_model.params.index,
    "资损率": loss_model.params.values,
    "客诉率": complaint_model.params.values
})
st.dataframe(coef_df)


# =========================
# 模型性能
# =========================
st.subheader("📈 模型性能")

df_eval = st.session_state.train_df
X = sm.add_constant(df_eval[["V", "weight"]])
y_true = df_eval["loss_rate"]
y_pred = inv_logit(loss_model.predict(X))

# 修复MAPE
mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None)))
mae = np.mean(np.abs(y_true - y_pred))

col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{loss_model.rsquared:.3f}")
col2.metric("MAE", f"{mae:.4f}")
col3.metric("MAPE", f"{mape*100:.2f}%")


# =========================
# 上传训练数据
# =========================
st.divider()
st.subheader("📤 上传训练数据")

file = st.file_uploader("上传Excel", type=["xlsx"])

if file and st.button("导入"):

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

    save_data(st.session_state.train_df, tag="upload")

    st.success("导入成功")


# =========================
# 🔄 回滚（升级版）
# =========================
st.divider()
st.subheader("🕘 历史版本回滚")

files = sorted(os.listdir(HISTORY_DIR), reverse=True)

if files:
    selected = st.selectbox("选择版本", files)

    if st.button("回滚到该版本"):

        df_old = pd.read_csv(os.path.join(HISTORY_DIR, selected))

        st.session_state.train_df = df_old
        save_data(df_old, tag="rollback")

        st.success(f"已回滚到：{selected}")


# =========================
# 下载
# =========================
buffer = BytesIO()
st.session_state.train_df.to_excel(buffer, index=False)

st.download_button("📥 下载当前训练数据", buffer.getvalue(), "train.xlsx")


# =========================
# 声明
# =========================
st.caption("⚠️ 模型用于相对风险评估")
