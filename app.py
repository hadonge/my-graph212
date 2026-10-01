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
    
    # 기준 조건 정결화: 2025년 이하 데이터만 사용
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

    # 데이터 통계 정보 계산
    count_years = len(data)
    start_year = int(data["연도"].min())
    end_year = int(data["연도"].max())

    # 선형 회귀 분석 (1908년부터 지난 연수 기준)
    X = data["지난연수"].values
    Y = data["연평균기온"].values
    
    # 1차 다항식 회귀 (기울기, 절편)
    slope, intercept = np.polyfit(X, Y, 1)
    
    # 상관계수 계산
    corr = np.corrcoef(X, Y)[0, 1]

    # 메인 레이아웃 구성
    st.subheader("📊 연도별 기온 예측")
    
    # 연도 선택 슬라이더 (1900년 ~ 2100년)
    selected_year = st.slider(
        "예측하고 싶은 연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1
    )

    # 선택한 연도의 예상 기온 계산
    predicted_temp = slope * (selected_year - 1908) + intercept

    # 예상 기온 강조 표시
    st.metric(
        label=f"{selected_year}년 예상 연평균 기온",
        value=f"{predicted_temp:.2f} °C"
    )

    # Plotly 시각화
    fig = go.Figure()

    # 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=data["연도"],
        y=data["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(color="blue", size=7)
    ))

    # 회귀 직선 (1900년부터 2100년 전체 구간)
    line_years = np.arange(1900, 2101)
    line_X = line_years - 1908
    line_Y = slope * line_X + intercept

    fig.add_trace(go.Scatter(
        x=line_years,
        y=line_Y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="red", width=2)
    ))

    # 선택된 연도 강조 점 추가
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택한 연도 ({selected_year}년)",
        marker=dict(color="green", size=12, symbol="star")
    ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title="서울 연평균 기온 변화 및 회귀 추세선",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 데이터 통계 및 상세 정보 출력
    st.markdown("---")
    st.subheader("ℹ️ 모델 및 데이터 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석에 사용된 해의 개수", f"{count_years} 개")
    col2.metric("시작 연도", f"{start_year} 년")
    col3.metric("끝 연도", f"{end_year} 년")
    col4.metric("상관계수", f"{corr:.4f}")

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
