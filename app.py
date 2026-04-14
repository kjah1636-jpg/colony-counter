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
            "테두리 감지 범위", 0.80, 0.98, 0.92, step=0.01,
            help="높일수록 가장자리 콜로니까지 더 감지합니다.")
        sensitivity = st.slider(
            "감지 민감도", 1, 30, 8,
            help="낮을수록 더 많이 감지합니다. 노이즈 증가 가능.")
        clahe_clip = st.slider(
            "대비 향상 강도 (CLAHE)", 1.0, 8.0, 3.0, step=0.5,
            help="높일수록 흐릿한 콜로니가 더 잘 보입니다.")
    with col2:
        min_area = st.slider(
            "최소 콜로니 면적 (px²)", 10, 300, 40,
            help="이보다 작은 점은 무시합니다.")
        max_area = st.slider(
            "최대 콜로니 면적 (px²)", 1000, 60000, 25000,
            help="이보다 큰 영역은 무시합니다.")
        min_circ = st.slider(
            "최소 원형도", 0.1, 0.8, 0.25, step=0.05,
            help="1.0이 완전한 원. 낮출수록 불규칙한 모양도 감지합니다.")


# ── 페트리 디시 마스크 ────────────────────────────────
def get_dish_mask(gray, margin):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    circles = None
    for param2 in [40, 30, 20, 15, 10]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
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


# ── 핵심: 저대비 콜로니 감지 ─────────────────────────
def detect_colonies(image_array, sensitivity, clahe_clip, min_area, max_area, min_circ, dish_margin):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 1) 디시 마스크
    dish_mask, dish_info = get_dish_mask(gray, dish_margin)

    # 2) CLAHE — 대비 향상
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # 3) DoG — 콜로니 blob 강조 (두 스케일)
    g1_small = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    g2_small = cv2.GaussianBlur(enhanced, (15, 15), 5.0)
    dog_small = cv2.subtract(g1_small, g2_small)

    g1_large = cv2.GaussianBlur(enhanced, (11, 11), 3.0)
    g2_large = cv2.GaussianBlur(enhanced, (51, 51), 15.0)
    dog_large = cv2.subtract(g1_large, g2_large)

    dog_combined = cv2.addWeighted(dog_small, 0.6, dog_large, 0.4, 0)

    # 4) 멀티스케일 Top-hat
    tophat_result = np.zeros_like(gray)
    for ksize in [21, 35, 55, 75]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        tophat_result = cv2.max(tophat_result, tophat)

    # 5) 결합
    combined_signal = cv2.addWeighted(
        dog_combined.astype(np.float32), 0.5,
        tophat_result.astype(np.float32), 0.5, 0
    ).astype(np.uint8)

    # 6) 임계값
    _, binary = cv2.threshold(combined_signal, sensitivity, 255, cv2.THRESH_BINARY)

    # 7) 모폴로지
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)

    # 8) 디시 마스크 적용
    binary = cv2.bitwise_and(binary, dish_mask)

    # 9) 윤곽선 필터링
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        cx_int, cy_int, r_int = int(x), int(y), max(int(r), 4)

        # 크기별 색상
        if area < 200:
            color = (255, 180, 50)   # 🟡 작은 콜로니
        elif area < 2000:
            color = (57, 211, 83)    # 🟢 중간 콜로니
        else:
            color = (100, 180, 255)  # 🔵 큰 콜로니

        cv2.circle(result, (cx_int, cy_int), r_int + 4, color, 3)
        cv2.circle(result, (cx_int, cy_int), 4, color, -1)

    # 디시 경계 표시
    cx_d, cy_d, _, r_d = dish_info
    cv2.circle(result, (cx_d, cy_d), r_d, (150, 150, 255), 2)

    return result, count, binary, dish_info


# ── 메인 ─────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    h_orig, w_orig = img_array.shape[:2]
    max_dim = 2000
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS))
        st.info(f"📐 이미지 크기 조정: {w_orig}×{h_orig} → {new_w}×{new_h}")

    with st.spinner("🔬 콜로니 감지 중..."):
        result_array, colony_count, debug_binary, dish_info = detect_colonies(
            img_array, sensitivity, clahe_clip, min_area, max_area, min_circ, dish_margin
        )

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown("**원본**")
        st.image(img_array, use_container_width=True)
    with col_result:
        st.markdown("**감지 결과** (🟡소 🟢중 🔵대)")
        st.image(result_array, use_container_width=True)

    with st.expander("🔍 디버그: 이진화 결과 보기", expanded=False):
        st.markdown("<p class='tip-text'>흰 점 = 감지된 영역. 콜로니 위치와 맞는지 확인하세요.</p>",
                    unsafe_allow_html=True)
        st.image(debug_binary, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<div class='big-count'>{colony_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='count-label'>colonies detected · CLAHE + DoG</div>",
                unsafe_allow_html=True)

    h, w = img_array.shape[:2]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("감지된 콜로니", f"{colony_count}개")
    col_b.metric("분석 크기", f"{w}×{h}")
    col_c.metric("밀도", f"{colony_count / (w * h) * 1e6:.1f} / Mpx")

    st.markdown("---")
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
    • <strong>가장자리 콜로니 누락</strong> → 테두리 감지 범위 높이기 (0.95 이상)<br>
    • <strong>작은 점이 노이즈로 잡힐 때</strong> → 최소 콜로니 면적 높이기<br><br>
    📌 <strong>디버그 탭</strong> — 이진화 결과에서 흰 점이 콜로니 위치와 맞는지 확인하면<br>
    어떤 파라미터를 조정해야 할지 바로 알 수 있습니다.
    </p></div>
    """, unsafe_allow_html=True)
