import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Colony Counter",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background-color: #0d1117;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

.big-count {
    font-family: 'Space Mono', monospace;
    font-size: 96px;
    font-weight: 700;
    color: #39d353;
    text-align: center;
    line-height: 1;
    text-shadow: 0 0 40px rgba(57,211,83,0.4);
}

.count-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #8b949e;
    text-align: center;
    margin-bottom: 32px;
}

.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
}

.stSlider > div > div > div > div {
    background-color: #39d353 !important;
}

div[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 20px;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #39d353;
}

.stButton > button {
    background-color: #238636;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    padding: 12px 28px;
    width: 100%;
    transition: all 0.2s;
}

.stButton > button:hover {
    background-color: #2ea043;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(57,211,83,0.3);
}

hr {
    border-color: #30363d;
}

.tip-text {
    color: #8b949e;
    font-size: 13px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────
st.markdown("# 🔬 Colony Counter")
st.markdown("<p class='tip-text'>페트리 디시 사진을 업로드하면 콜로니를 자동으로 감지하고 개수를 세어드립니다.</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Upload ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "이미지 업로드 (JPG / PNG / TIFF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# ── Parameters ──────────────────────────────────────────
with st.expander("⚙️ 감지 파라미터 조정", expanded=False):
    st.markdown("<p class='tip-text'>결과가 잘 안 맞을 때 값을 조절해 보세요.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        min_radius = st.slider("최소 콜로니 크기 (px)", 3, 30, 8, help="너무 작은 점은 무시합니다")
        threshold_val = st.slider("이진화 임계값", 50, 200, 120, help="낮을수록 어두운 콜로니도 감지")
    with col2:
        max_radius = st.slider("최대 콜로니 크기 (px)", 20, 200, 80, help="너무 큰 영역은 무시합니다")
        blur_size = st.slider("블러 강도", 1, 11, 3, step=2, help="노이즈 제거 강도")

# ── Processing ──────────────────────────────────────────
def detect_colonies(image_array, min_r, max_r, thresh, blur):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    k = blur if blur % 2 == 1 else blur + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    # Adaptive threshold + binary
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image_array.copy()
    count = 0
    min_area = np.pi * (min_r ** 2)
    max_area = np.pi * (max_r ** 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            # Circularity check (optional loose filter)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.2:  # too non-circular → skip
                continue

            count += 1
            # Draw circle overlay
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(x), int(y), int(radius)
            cv2.circle(result, (cx, cy), r, (57, 211, 83), 2)
            cv2.circle(result, (cx, cy), 2, (57, 211, 83), -1)

    return result, count

# ── Main flow ────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown("**원본**")
        st.image(image, use_container_width=True)

    with st.spinner("콜로니 감지 중..."):
        result_array, colony_count = detect_colonies(
            img_array, min_radius, max_radius, threshold_val, blur_size
        )

    with col_result:
        st.markdown("**감지 결과**")
        st.image(result_array, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<div class='big-count'>{colony_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='count-label'>colonies detected</div>", unsafe_allow_html=True)

    # Stats
    h, w = img_array.shape[:2]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("감지된 콜로니", f"{colony_count}개")
    col_b.metric("이미지 크기", f"{w}×{h}")
    col_c.metric("밀도", f"{colony_count / (w*h) * 1e6:.1f} / Mpx")

    # Download
    st.markdown("---")
    result_pil = Image.fromarray(result_array)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 결과 이미지 다운로드",
        data=buf.getvalue(),
        file_name=f"colony_result_{colony_count}colonies.png",
        mime="image/png"
    )

else:
    st.markdown("""
    <div class='card'>
    <p class='tip-text'>
    📌 <strong>사용 방법</strong><br><br>
    1. 위의 업로더에 페트리 디시 사진을 드래그하거나 클릭해서 업로드<br>
    2. 결과가 잘 안 맞으면 <strong>감지 파라미터</strong>를 조절<br>
    3. 결과 이미지를 다운로드해서 저장<br><br>
    📌 <strong>잘 작동하는 조건</strong><br><br>
    • 밝고 균일한 조명<br>
    • 콜로니와 배지 색 대비가 뚜렷할수록 정확<br>
    • JPEG보다 PNG/TIFF 권장
    </p>
    </div>
    """, unsafe_allow_html=True)
