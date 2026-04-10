import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Colony Counter", page_icon="🔬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.big-count { font-family: 'Space Mono', monospace; font-size: 96px; font-weight: 700; color: #39d353; text-align: center; line-height: 1; text-shadow: 0 0 40px rgba(57,211,83,0.4); }
.count-label { font-family: 'DM Sans', sans-serif; font-size: 14px; letter-spacing: 3px; text-transform: uppercase; color: #8b949e; text-align: center; margin-bottom: 32px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin: 16px 0; }
div[data-testid="stFileUploader"] { background: #161b22; border: 2px dashed #30363d; border-radius: 12px; padding: 20px; }
div[data-testid="stFileUploader"]:hover { border-color: #39d353; }
.stButton > button { background-color: #238636; color: white; border: none; border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 14px; padding: 12px 28px; width: 100%; transition: all 0.2s; }
.stButton > button:hover { background-color: #2ea043; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(57,211,83,0.3); }
hr { border-color: #30363d; }
.tip-text { color: #8b949e; font-size: 13px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔬 Colony Counter")
st.markdown("<p class='tip-text'>페트리 디시 사진을 업로드하면 콜로니를 자동으로 감지하고 개수를 세어드립니다.<br>밝은 배지 위의 흰색/불투명 콜로니 + 플라스틱 반사 제거에 최적화되어 있습니다.</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader("이미지 업로드 (JPG / PNG / TIFF)", type=["jpg","jpeg","png","tif","tiff"])

with st.expander("⚙️ 감지 파라미터 조정", expanded=False):
    st.markdown("<p class='tip-text'>결과가 잘 안 맞을 때 값을 조절해 보세요.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        tophat_thresh = st.slider("감지 민감도 (Top-hat 임계값)", 5, 40, 18,
            help="낮을수록 더 많이 감지. 노이즈 증가 시 높이세요.")
        min_area = st.slider("최소 콜로니 면적 (px²)", 50, 500, 250,
            help="이보다 작은 점은 무시합니다. 작은 콜로니가 안 잡히면 낮추세요.")
    with col2:
        max_area = st.slider("최대 콜로니 면적 (px²)", 1000, 30000, 15000,
            help="이보다 큰 영역은 무시합니다.")
        min_circ = st.slider("최소 원형도", 0.1, 0.8, 0.35, step=0.05,
            help="1.0이 완전한 원. 낮을수록 불규칙한 모양도 감지.")
        dish_margin = st.slider("테두리 반사 제거 비율", 0.75, 0.97, 0.86, step=0.01,
            help="낮출수록 테두리를 더 많이 제외합니다. 반사가 심하면 낮추세요.")

def detect_colonies(image_array, tophat_thresh, min_area, max_area, min_circ, dish_margin):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. 페트리 디시 원 감지
    blurred_hough = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred_hough, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                                param1=50, param2=30,
                                minRadius=min(h,w)//4, maxRadius=min(h,w)//2)
    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        cx, cy, cr = np.round(circles[0][0]).astype(int)
        cv2.circle(mask, (cx, cy), int(cr * dish_margin), 255, -1)
    else:
        mask[:] = 255

    # 2. Top-hat 변환 (주변보다 밝은 콜로니 추출)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

    # 3. 이진화 + 형태학적 정리
    _, binary = cv2.threshold(tophat, tophat_thresh, 255, cv2.THRESH_BINARY)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
    masked = cv2.bitwise_and(cleaned, mask)

    # 4. 컨투어 검출 + 필터링
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image_array.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circ:
            continue
        count += 1
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(result, (int(x), int(y)), int(r) + 3, (57, 211, 83), 3)
        cv2.circle(result, (int(x), int(y)), 4, (57, 211, 83), -1)
    return result, count

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown("**원본**")
        st.image(image, use_container_width=True)

    with st.spinner("콜로니 감지 중..."):
        result_array, colony_count = detect_colonies(
            img_array, tophat_thresh, min_area, max_area, min_circ, dish_margin)

    with col_result:
        st.markdown("**감지 결과**")
        st.image(result_array, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<div class='big-count'>{colony_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='count-label'>colonies detected</div>", unsafe_allow_html=True)

    h, w = img_array.shape[:2]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("감지된 콜로니", f"{colony_count}개")
    col_b.metric("이미지 크기", f"{w}×{h}")
    col_c.metric("밀도", f"{colony_count / (w*h) * 1e6:.1f} / Mpx")

    st.markdown("---")
    result_pil = Image.fromarray(result_array)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(label="📥 결과 이미지 다운로드", data=buf.getvalue(),
                       file_name=f"colony_result_{colony_count}colonies.png", mime="image/png")
else:
    st.markdown("""
    <div class='card'><p class='tip-text'>
    📌 <strong>사용 방법</strong><br><br>
    1. 위의 업로더에 페트리 디시 사진을 드래그하거나 클릭해서 업로드<br>
    2. 결과가 잘 안 맞으면 <strong>감지 파라미터</strong>를 조절<br>
    3. 결과 이미지를 다운로드해서 저장<br><br>
    📌 <strong>파라미터 가이드</strong><br><br>
    • 콜로니가 덜 잡힐 때 → 감지 민감도 낮추기, 최소 면적 낮추기<br>
    • 노이즈/반사가 많이 잡힐 때 → 감지 민감도 높이기, 테두리 반사 제거 비율 낮추기<br>
    • 불규칙한 모양 콜로니 → 원형도 낮추기
    </p></div>
    """, unsafe_allow_html=True)
