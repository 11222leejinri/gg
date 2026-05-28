import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# -------------------------------
# 페이지 설정
# -------------------------------
st.set_page_config(
    page_title="폐 건강 군집 분석 시스템",
    page_icon="🫁",
    layout="wide"
)

# -------------------------------
# CSS 디자인
# -------------------------------
st.markdown("""
<style>

/* 배경 */
.stApp {
    background:
    linear-gradient(
        rgba(240,248,255,0.93),
        rgba(240,248,255,0.93)
    ),
    url("https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?auto=format&fit=crop&w=1920&q=80");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* 제목 */
.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #1565C0;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
    font-size: 18px;
}

/* 카드 */
.card {
    background: rgba(255,255,255,0.88);
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

/* 버튼 */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 18px;
    border: none;
    background: linear-gradient(
        90deg,
        #42A5F5,
        #1976D2
    );
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Metric */
[data-testid="stMetricValue"] {
    font-size: 45px;
    color: #1565C0;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# 모델 불러오기
# -------------------------------
model = joblib.load("g_model.pkl")

# scaler 있으면 자동 사용
scaler = None
if os.path.exists("g_scaler.pkl"):
    scaler = joblib.load("g_scaler.pkl")

# -------------------------------
# 데이터 불러오기
# -------------------------------
df = pd.read_csv("lung.csv")

# cluster 컬럼 없으면 자동 생성
if "cluster" not in df.columns:

    feature_cols = ['나이', '담배여부', '알코올']

    X = df[feature_cols]

    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    df["cluster"] = model.predict(X_scaled)

# -------------------------------
# 제목
# -------------------------------
st.markdown(
    '<div class="main-title">🫁 폐 건강 군집 분석 시스템</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">나이 · 흡연 · 음주 데이터를 기반으로 환자 유형을 분석합니다.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

# -------------------------------
# 입력 UI
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "🧑 나이",
        0,
        100,
        30
    )

with col2:
    smoking = st.toggle(
        "🚬 흡연 여부"
    )

    alcohol = st.toggle(
        "🍺 음주 여부"
    )

# bool → 숫자
smoking_num = 1 if smoking else 0
alcohol_num = 1 if alcohol else 0

# -------------------------------
# 예측 버튼
# -------------------------------
if st.button("🔍 환자 유형 분석하기"):

    new_patient = pd.DataFrame(
        [[
            age,
            smoking_num,
            alcohol_num
        ]],
        columns=[
            '나이',
            '담배여부',
            '알코올'
        ]
    )

    try:
        # 스케일링
        if scaler:
            new_patient_input = scaler.transform(
                new_patient
            )
        else:
            new_patient_input = new_patient

        # 군집 예측
        pred_cluster = model.predict(
            new_patient_input
        )

        cluster_num = int(
            pred_cluster[0]
        )

        st.markdown("---")

        col_result, col_graph = st.columns([1, 2])

        # -------------------------------
        # 결과 카드
        # -------------------------------
        with col_result:

            st.subheader("📊 분석 결과")

            st.metric(
                "예측 군집",
                f"{cluster_num}번"
            )

            if cluster_num == 0:
                st.success(
                    "✅ 건강 관리형 환자군"
                )

            elif cluster_num == 1:
                st.warning(
                    "⚠️ 생활 습관 개선 필요군"
                )

            elif cluster_num == 2:
                st.error(
                    "🚨 고위험 생활 습관군"
                )

            else:
                st.info(
                    "ℹ️ 기타 환자군"
                )

        # -------------------------------
        # 군집 시각화
        # -------------------------------
        with col_graph:

            st.subheader("📈 군집 위치 시각화")

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            scatter = ax.scatter(
                df['나이'],
                df['담배여부'],
                c=df['cluster'],
                alpha=0.5
            )

            # 현재 환자 표시
            ax.scatter(
                age,
                smoking_num,
                c='black',
                s=350,
                marker='X',
                label='현재 환자'
            )

            ax.set_xlabel("나이")
            ax.set_ylabel("담배 여부")
            ax.set_title("환자 군집 분포")

            ax.legend()

            st.pyplot(fig)

    except Exception as e:
        st.error(
            f"오류 발생: {e}"
        )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)