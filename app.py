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
.big-count {
    font-family: 'Space Mono', monospace; font-size: 96px; font-weight: 700;
    color: #39d353; text-align: center; line-height: 1;
    text-shadow: 0 0 40px rgba(57,211,83,0.4);
}
.count-label {
    font-family: 'DM Sans', sans-serif; font-size: 14px; letter-spacing: 3px;
    text-transform: uppercase; color: #8b949e; text-align: center; margin-bottom: 32px;
}
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin: 16px 0; }
div[data-testid="stFileUploader"] {
    background: #161b22; border: 2px dashed #30363d; border-radius: 12px; padding: 20px;
}
div[data-testid="stFileUploader"]:hover { border-color: #39d353; }
.stButton > button {
    background-color: #238636; color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-size: 14px;
    padding: 12px 28px; width: 100%; transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #2ea043; transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(57,211,83,0.3);
}
hr { border-color: #30363d; }
.tip-text { color: #8b949e; font-size: 13px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔬 Colony Counter")
st.markdown("---")

uploaded = st.file_uploader("이미지 업로드 (JPG / PNG / TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

with st.expander("⚙️ 감지 파라미터 조정", expanded=True):
    st.markdown("<p class='tip-text'>결과가 잘 안 맞을 때 값을 조절해 보세요.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        dish_margin = st.slider(
            "테두리 감지 범위", 0.80, 0.98, 0.93, step=0.01,
            help="높일수록 가장자리 콜로니까지 더 감지합니다.")
        bg_blur = st.slider(
            "배경 블러 크기", 21, 101, 51, step=10,
            help="크게 할수록 큰 배경 변화를 잘 제거합니다. 기본값 51 권장.")
        sensitivity = st.slider(
            "감지 민감도 (낮을수록 더 많이)", 3, 25, 10,
            help="낮을수록 흐릿한 콜로니도 감지합니다. 노이즈 증가 가능.")
        clahe_clip = st.slider(
            "대비 향상 강도", 1.0, 6.0, 3.0, step=0.5,
            help="높일수록 저대비 콜로니가 더 잘 보입니다.")
    with col2:
        dist_thresh = st.slider(
            "콜로니 분리 민감도", 0.1, 0.6, 0.3, step=0.05,
            help="낮을수록 붙어있는 콜로니를 더 잘 분리합니다.")
        min_area = st.slider(
            "최소 콜로니 면적 (px²)", 5, 200, 40,
            help="이보다 작은 점은 무시합니다. 작은 콜로니가 누락되면 낮추세요.")
        max_area = st.slider(
            "최대 콜로니 면적 (px²)", 500, 10000, 3000,
            help="이보다 큰 영역은 무시합니다.")
        min_circ = st.slider(
            "최소 원형도", 0.1, 0.7, 0.3, step=0.05,
            help="낮출수록 불규칙한 모양도 감지합니다.")


# ── 페트리 디시 마스크 ────────────────────────────────
def get_dish_mask(gray, margin):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    circles = None
    for param2 in [40, 30, 20, 15, 10]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=param2,
            minRadius=int(min(h, w) * 0.30),
            maxRadius=int(min(h, w) * 0.56)
        )
        if circles is not None:
            break
    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        cx, cy, cr = np.round(circles[0][0]).astype(int)
        r_applied = min(int(cr * margin), min(h, w) // 2 - 5)
        cv2.circle(mask, (cx, cy), r_applied, 255, -1)
        return mask, (cx, cy, cr, r_applied)
    else:
        cx, cy = w // 2, h // 2
        r_applied = int(min(h, w) * 0.44)
        cv2.circle(mask, (cx, cy), r_applied, 255, -1)
        return mask, (cx, cy, r_applied, r_applied)


# ── 핵심 감지 알고리즘 ────────────────────────────────
def detect_colonies(image_array, sensitivity, clahe_clip, bg_blur,
                    dist_thresh, min_area, max_area, min_circ, dish_margin):

    # 1) 1500px로 리사이즈 (처리 속도 + 파라미터 일관성)
    h_o, w_o = image_array.shape[:2]
    TARGET = 1500
    scale = TARGET / max(h_o, w_o)
    if scale < 1.0:
        new_w, new_h = int(w_o * scale), int(h_o * scale)
        img = np.array(Image.fromarray(image_array).resize((new_w, new_h), Image.LANCZOS))
    else:
        img = image_array.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 2) 디시 마스크
    dish_mask, dish_info = get_dish_mask(gray, dish_margin)

    # 3) CLAHE 대비 향상
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4) 배경 제거 (가우시안 블러로 배경 모델링 후 차감)
    bg_ksize = bg_blur if bg_blur % 2 == 1 else bg_blur + 1
    background = cv2.GaussianBlur(enhanced, (bg_ksize, bg_ksize), 0)
    diff = cv2.subtract(enhanced, background)
    diff[dish_mask == 0] = 0

    # 5) 임계값 적용
    _, binary = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)

    # 6) 모폴로지 — 노이즈 제거 및 채우기
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)

    # 7) Distance Transform — 붙어있는 콜로니 분리
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 8) 윤곽선 탐지 및 필터링
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
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
        cx_int, cy_int = int(x), int(y)
        r_draw = max(int(r) + 4, 6)

        # 크기별 색상
        if area < 100:
            color = (255, 180, 50)   # 🟡 작은 콜로니
        elif area < 600:
            color = (57, 211, 83)    # 🟢 중간 콜로니
        else:
            color = (100, 180, 255)  # 🔵 큰 콜로니

        cv2.circle(result, (cx_int, cy_int), r_draw, color, 3)
        cv2.circle(result, (cx_int, cy_int), 5, color, -1)

    # 디시 경계 표시
    cx_d, cy_d, _, r_d = dish_info
    cv2.circle(result, (cx_d, cy_d), r_d, (150, 150, 255), 2)

    return result, count, binary, sure_fg, dish_info


# ── 메인 ─────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)
    h_orig, w_orig = img_array.shape[:2]

    with st.spinner("🔬 콜로니 감지 중..."):
        result_array, colony_count, debug_binary, debug_dist, dish_info = detect_colonies(
            img_array, sensitivity, clahe_clip, bg_blur,
            dist_thresh, min_area, max_area, min_circ, dish_margin
        )

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown("**원본**")
        st.image(img_array, use_container_width=True)
    with col_result:
        st.markdown("**감지 결과** 🟡소형 🟢중형 🔵대형")
        st.image(result_array, use_container_width=True)

    # 디버그 탭
    with st.expander("🔍 디버그: 중간 처리 결과", expanded=False):
        st.markdown("<p class='tip-text'>파라미터 조정 참고용입니다.</p>", unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**배경제거 후 이진화**")
            st.image(debug_binary, use_container_width=True,
                     caption="흰 부분 = 콜로니 후보 영역")
        with dc2:
            st.markdown("**Distance Transform 후 (콜로니 중심)**")
            st.image(debug_dist, use_container_width=True,
                     caption="흰 점 = 최종 감지된 콜로니 중심")

    st.markdown("---")
    st.markdown(f"<div class='big-count'>{colony_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='count-label'>colonies detected · Background Subtraction + Distance Transform</div>",
                unsafe_allow_html=True)

    h_r, w_r = result_array.shape[:2]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("감지된 콜로니", f"{colony_count}개")
    col_b.metric("원본 크기", f"{w_orig}×{h_orig}")
    col_c.metric("밀도", f"{colony_count / (w_r * h_r) * 1e6:.1f} / Mpx")

    st.markdown("---")

    # 파라미터 조정 가이드 (결과 기반)
    st.markdown("""
    <div class='card'><p class='tip-text'>
    📌 <strong>결과 조정 가이드</strong><br><br>
    • 콜로니가 <strong>덜 잡힐 때</strong> → 감지 민감도 낮추기 / 최소 면적 낮추기<br>
    • 노이즈가 <strong>너무 많이 잡힐 때</strong> → 감지 민감도 높이기 / 최소 면적 높이기 / 최소 원형도 높이기<br>
    • <strong>가장자리 콜로니 누락</strong> → 테두리 감지 범위 높이기 (0.95 이상)<br>
    • <strong>붙어있는 콜로니가 하나로 잡힐 때</strong> → 콜로니 분리 민감도 낮추기<br>
    • <strong>디버그 탭</strong>의 이진화 결과에서 콜로니 위치에 흰 영역이 있는지 확인하세요
    </p></div>
    """, unsafe_allow_html=True)

    result_pil = Image.fromarray(result_array)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 결과 이미지 다운로드",
        data=buf.getvalue(),
        file_name=f"colony_result_{colony_count}.png",
        mime="image/png"
    )

else:
    st.markdown("""
    <div class='card'><p class='tip-text'>
    📌 <strong>파라미터 조정 가이드</strong><br><br>
    • 콜로니가 <strong>덜 잡힐 때</strong> → 감지 민감도 낮추기 / 대비 향상 강도 높이기<br>
    • 콜로니가 <strong>너무 많이 잡힐 때</strong> → 감지 민감도 높이기 / 최소 면적 높이기<br>
    • <strong>가장자리 콜로니 누락</strong> → 테두리 감지 범위 높이기 (0.95~0.97)<br>
    • <strong>붙어있는 콜로니</strong> → 콜로니 분리 민감도 낮추기<br>
    • <strong>디버그 탭</strong>에서 이진화 결과를 보면 어디가 문제인지 바로 알 수 있습니다
    </p></div>
    """, unsafe_allow_html=True)
