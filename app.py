import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기 & 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측 및 선형회귀 모델 평가")

# 데이터 불러오기 및 전처리 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 데이터 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 집계
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일이 300일 이상인 연도 추출
    filtered_df = yearly_stats[yearly_stats["관측일수"] >= 300].copy()
    
    # 1908년 기준 경과 연수를 독립변수(X)로 설정
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    return filtered_df

# Evaluation metric calculator
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

try:
    data = load_data()

    # ----------------------------------------------------
    # 데이터셋 분할 (훈련 세트 2종류 vs 테스트 세트 1종류)
    # ----------------------------------------------------
    # 공통 테스트 데이터: 최근 20년 (2006 ~ 2025년)
    test_df = data[(data["연도"] >= 2006) & (data["연도"] <= 2025)].copy()
    X_test = test_df["지난연수"].values
    y_test = test_df["연평균기온"].values

    # 모델 1: 전체 데이터 학습
    X_all = data["지난연수"].values
    y_all = data["연평균기온"].values
    slope_all, intercept_all = np.polyfit(X_all, y_all, 1)

    # 모델 2: 최근 100년 학습 (1906 ~ 2005년)
    train_100y_df = data[(data["연도"] >= 1906) & (data["연도"] <= 2005)].copy()
    X_train_100 = train_100y_df["지난연수"].values
    y_train_100 = train_100y_df["연평균기온"].values
    slope_100, intercept_100 = np.polyfit(X_train_100, y_train_100, 1)

    # 모델 3: 최근 50년 학습 (1956 ~ 2005년)
    train_50y_df = data[(data["연도"] >= 1956) & (data["연도"] <= 2005)].copy()
    X_train_50 = train_50y_df["지난연수"].values
    y_train_50 = train_50y_df["연평균기온"].values
    slope_50, intercept_50 = np.polyfit(X_train_50, y_train_50, 1)

    # ----------------------------------------------------
    # 공통 테스트 데이터(2006~2025)에 대한 예측 수행 및 평가
    # ----------------------------------------------------
    pred_test_all = slope_all * X_test + intercept_all
    pred_test_100 = slope_100 * X_test + intercept_100
    pred_test_50 = slope_50 * X_test + intercept_50

    mae_all, mse_all, r2_all = evaluate_model(y_test, pred_test_all)
    mae_100, mse_100, r2_100 = evaluate_model(y_test, pred_test_100)
    mae_50, mse_50, r2_50 = evaluate_model(y_test, pred_test_50)

    # ----------------------------------------------------
    # 1. 상단 기울기(100년당 기온 상승 폭) 메트릭 표시
    # ----------------------------------------------------
    st.subheader("🔥 모델별 100년당 기온 상승 폭 비교")
    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="전체 데이터 모델",
        value=f"+{slope_all * 100:.2f} °C / 100년"
    )
    col2.metric(
        label="최근 100년 학습 (1906~2005년)",
        value=f"+{slope_100 * 100:.2f} °C / 100년"
    )
    col3.metric(
        label="최근 50년 학습 (1956~2005년)",
        value=f"+{slope_50 * 100:.2f} °C / 100년",
        delta=f"+{(slope_50 - slope_100) * 100:.2f} °C (100년 학습 대비 가속)",
        delta_color="normal"
    )

    st.markdown("---")

    # ----------------------------------------------------
    # 2. 테스트 데이터(2006~2025년) 예측 성능 평가 결과표
    # ----------------------------------------------------
    st.subheader("🧪 테스트 데이터(2006~2025년) 예측 성능 평가 (MAE, MSE, R²)")
    
    eval_summary = pd.DataFrame({
        "학습 모델": ["전체 데이터", "최근 100년 (1906~2005)", "최근 50년 (1956~2005)"],
        "기울기 (100년당)": [f"+{slope_all*100:.2f} °C", f"+{slope_100*100:.2f} °C", f"+{slope_50*100:.2f} °C"],
        "MAE (평균 절대 오차)": [f"{mae_all:.4f}", f"{mae_100:.4f}", f"{mae_50:.4f}"],
        "MSE (평균 제곱 오차)": [f"{mse_all:.4f}", f"{mse_100:.4f}", f"{mse_50:.4f}"],
        "R² (결정 계수)": [f"{r2_all:.4f}", f"{r2_100:.4f}", f"{r2_50:.4f}"]
    })
    
    st.table(eval_summary)

    st.markdown("---")

    # ----------------------------------------------------
    # 3. Plotly 회귀선 시각화 비교
    # ----------------------------------------------------
    st.subheader("📈 모델별 회귀선 및 테스트 데이터 추세 시각화")

    line_years = np.arange(1900, 2101)
    line_X = line_years - 1908

    fig = go.Figure()

    # 과거 훈련 데이터
    past_df = data[data["연도"] < 2006]
    fig.add_trace(go.Scatter(
        x=past_df["연도"], y=past_df["연평균기온"],
        mode="markers", name="과거 관측 데이터 (<2006)",
        marker=dict(color="gray", size=6, opacity=0.5)
    ))

    # 테스트 데이터
    fig.add_trace(go.Scatter(
        x=test_df["연도"], y=test_df["연평균기온"],
        mode="markers", name="테스트 데이터 (2006~2025)",
        marker=dict(color="red", size=9, symbol="diamond")
    ))

    # 회귀선
    fig.add_trace(go.Scatter(
        x=line_years, y=slope_all * line_X + intercept_all,
        mode="lines", name="전체 데이터 회귀선",
        line=dict(color="blue", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=line_years, y=slope_100 * line_X + intercept_100,
        mode="lines", name="최근 100년(1906~2005) 회귀선",
        line=dict(color="green", width=2, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=line_years, y=slope_50 * line_X + intercept_50,
        mode="lines", name="최근 50년(1956~2005) 회귀선",
        line=dict(color="orange", width=2, dash="dot")
    ))

    fig.update_layout(
        title="서울 연평균 기온 회귀 모델 비교",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # 4. 연도 선택 및 예상 기온 비교 슬라이더
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 연도별 예측 기온 비교")
    
    selected_year = st.slider("예측 연도 선택 (1900~2100년)", 1900, 2100, 2025, step=1)
    sel_X = selected_year - 1908

    p_all = slope_all * sel_X + intercept_all
    p_100 = slope_100 * sel_X + intercept_100
    p_50 = slope_50 * sel_X + intercept_50

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("전체 모델 예측값", f"{p_all:.2f} °C")
    p_col2.metric("100년 모델 예측값", f"{p_100:.2f} °C")
    p_col3.metric("50년 모델 예측값", f"{p_50:.2f} °C")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
