import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 불러오기 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    # CSV 로드
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 기준 조건: 2025년 이하 데이터만 사용
    df = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일이 300일 이상인 연도만 필터링
    filtered_df = yearly_stats[yearly_stats["관측일수"] >= 300].copy()
    
    # 1908년부터 지난 연수를 독립변수(X)로 설정
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

try:
    data = load_data()

    # 기본 데이터 정보
    count_years = len(data)
    start_year = int(data["연도"].min())
    end_year = int(data["연도"].max())

    # 1. 전체 기간 회귀 분석 (1908년부터 지난 연수 기준)
    X_all = data["지난연수"].values
    Y_all = data["연평균기온"].values
    slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)
    corr_all = np.corrcoef(X_all, Y_all)[0, 1]

    # 2. 최근 20년 데이터 회귀 분석
    data_20y = data[data["연도"] >= (end_year - 19)].copy()
    X_20y = data_20y["지난연수"].values
    Y_20y = data_20y["연평균기온"].values
    slope_20y, intercept_20y = np.polyfit(X_20y, Y_20y, 1)
    corr_20y = np.corrcoef(X_20y, Y_20y)[0, 1]

    # 100년당 상승 기온 계산
    slope_100y_all = slope_all * 100
    slope_100y_20y = slope_20y * 100

    # ----------------------------------------------------
    # 상단 메인 지표: 100년당 기온 상승 폭 비교 (크게 표시)
    # ----------------------------------------------------
    st.subheader("🔥 100년당 기온 상승 폭 비교")
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.metric(
            label=f"전체 기간 ({start_year}~{end_year}년)",
            value=f"100년당 +{slope_100y_all:.2f} °C",
            help="전체 데이터로 구한 회귀선의 기울기"
        )

    with col_metric2:
        # 전체 기간 대비 최근 20년의 가속도 계산
        diff_100y = slope_100y_20y - slope_100y_all
        st.metric(
            label=f"최근 20년 ({end_year-19}~{end_year}년)",
            value=f"100년당 +{slope_100y_20y:.2f} °C",
            delta=f"{diff_100y:+.2f} °C (전체 대비 빠른 정도)",
            delta_color="normal"
        )

    st.markdown("---")

    # ----------------------------------------------------
    # 인터랙티브 연도 선택 및 예측
    # ----------------------------------------------------
    st.subheader("📊 연도별 기온 예측")
    
    selected_year = st.slider(
        "예측하고 싶은 연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1
    )

    # 선택한 연도의 예상 기온 (전체 기간 모델 기준)
    predicted_temp = slope_all * (selected_year - 1908) + intercept_all

    st.metric(
        label=f"{selected_year}년 예상 연평균 기온 (전체 기간 모델 기준)",
        value=f"{predicted_temp:.2f} °C"
    )

    # ----------------------------------------------------
    # Plotly 시각화 (전체 회귀선 vs 최근 20년 회귀선)
    # ----------------------------------------------------
    fig = go.Figure()

    # 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=data["연도"],
        y=data["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(color="blue", size=7, opacity=0.6)
    ))

    # 전체 기간 회귀 직선 (1900년 ~ 2100년)
    line_years = np.arange(1900, 2101)
    line_X = line_years - 1908
    line_Y_all = slope_all * line_X + intercept_all

    fig.add_trace(go.Scatter(
        x=line_years,
        y=line_Y_all,
        mode="lines",
        name=f"전체 기간 회귀선 (+{slope_100y_all:.2f}°C/100년)",
        line=dict(color="red", width=2)
    ))

    # 최근 20년 회귀 직선
    line_Y_20y = slope_20y * line_X + intercept_20y
    fig.add_trace(go.Scatter(
        x=line_years,
        y=line_Y_20y,
        mode="lines",
        name=f"최근 20년 회귀선 (+{slope_100y_20y:.2f}°C/100년)",
        line=dict(color="orange", width=2, dash="dash")
    ))

    # 선택된 연도 강조 점
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택한 연도 ({selected_year}년)",
        marker=dict(color="green", size=13, symbol="star")
    ))

    fig.update_layout(
        title="서울 연평균 기온 추세선 비교 (전체 기간 vs 최근 20년)",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # 상세 통계 비교
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("ℹ️️ 상세 분석 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석에 사용된 해의 개수", f"{count_years} 개")
    col2.metric("시작 연도 / 끝 연도", f"{start_year}년 / {end_year}년")
    col3.metric("전체 기간 상관계수", f"{corr_all:.4f}")
    col4.metric("최근 20년 상관계수", f"{corr_20y:.4f}")

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
