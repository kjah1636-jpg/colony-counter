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

mode = st.radio(
    "감지 방식 선택",
    ["🤖 AI 모드 (Cellpose — 밀집 콜로니 추천)", "🔧 OpenCV 모드 (빠름 — 간격 넓은 콜로니 추천)"],
    horizontal=True
)
use_cellpose = mode.startswith("🤖")

if use_cellpose:
    st.markdown(
        "<p class='tip-text'>Cellpose 딥러닝 모델로 붙어있는 콜로니도 분리해서 카운팅합니다.<br>"
        "⚠️ 첫 실행 시 모델 다운로드로 1~2분 걸릴 수 있습니다.</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p class='tip-text'>OpenCV 멀티채널 알고리즘으로 빠르게 카운팅합니다.<br>"
        "녹색 형광 콜로니와 흰색 콜로니를 각각 감지합니다.</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

uploaded = st.file_uploader("이미지 업로드 (JPG / PNG / TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

with st.expander("⚙️ 감지 파라미터 조정", expanded=False):
    st.markdown("<p class='tip-text'>결과가 잘 안 맞을 때 값을 조절해 보세요.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        dish_margin = st.slider("테두리 반사 제거 비율", 0.75, 0.99, 0.93, step=0.01,
            help="높일수록 테두리 가장자리까지 더 많이 감지합니다. 0.99면 거의 끝까지.")
        if use_cellpose:
            cell_diameter = st.slider("콜로니 평균 지름 (px)", 3, 80, 15,
                help="콜로니 하나의 대략적인 픽셀 크기.")
            flow_threshold = st.slider("경계 민감도 (flow threshold)", 0.1, 1.0, 0.4, step=0.05)
            cellprob_threshold = st.slider("콜로니 확신도", -4.0, 4.0, 0.0, step=0.5)
        else:
            tophat_thresh = st.slider("흰색 콜로니 감지 민감도", 5, 40, 15,
                help="낮을수록 더 많이 감지합니다.")
            green_sensitivity = st.slider("녹색 형광 콜로니 민감도", 0, 80, 40,
                help="낮을수록 더 많이 감지합니다. 녹색 콜로니가 없으면 100으로 올리세요.")
    with col2:
        min_area = st.slider("최소 콜로니 면적 (px²)", 10, 500, 80,
            help="이보다 작은 점은 무시합니다.")
        max_area = st.slider("최대 콜로니 면적 (px²)", 1000, 50000, 20000,
            help="이보다 큰 영역은 무시합니다.")
        if not use_cellpose:
            min_circ = st.slider("최소 원형도", 0.1, 0.8, 0.25, step=0.05,
                help="1.0이 완전한 원. 낮출수록 불규칙한 모양도 감지합니다.")


# ── 디시 마스크: 개선된 버전 ──────────────────────────
def get_dish_mask(gray, margin):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # 여러 파라미터로 시도해서 가장 잘 맞는 원 찾기
    circles = None
    for param2 in [40, 30, 20, 15]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
            param1=50, param2=param2,
            minRadius=int(min(h, w) * 0.3),
            maxRadius=int(min(h, w) * 0.55)
        )
        if circles is not None:
            break

    mask = np.zeros((h, w), dtype=np.uint8)
    dish_info = None

    if circles is not None:
        cx, cy, cr = np.round(circles[0][0]).astype(int)
        # margin을 높게 설정해도 이미지 밖으로 나가지 않도록 클램핑
        r_applied = min(int(cr * margin), min(h, w) // 2 - 5)
        cv2.circle(mask, (cx, cy), r_applied, 255, -1)
        dish_info = (cx, cy, cr, r_applied)
    else:
        # 원 감지 실패 시 이미지 중앙 80% 영역 사용
        cx, cy = w // 2, h // 2
        r_applied = int(min(h, w) * 0.45)
        cv2.circle(mask, (cx, cy), r_applied, 255, -1)
        dish_info = (cx, cy, r_applied, r_applied)

    return mask, dish_info


# ── 녹색 형광 콜로니 감지 ────────────────────────────
def detect_green_colonies(image_array, sensitivity_thresh):
    """HSV 색공간에서 녹색 형광 콜로니만 분리"""
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # 녹색 HSV 범위 (형광 녹색 포함)
    lower_green = np.array([35, sensitivity_thresh, 80])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return green_mask


# ── 흰색/밝은 콜로니 감지 (Top-hat) ──────────────────
def detect_white_colonies(image_array, tophat_thresh):
    """Top-hat으로 밝은 콜로니 감지"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 다양한 크기 대응을 위해 멀티스케일 Top-hat
    result = np.zeros_like(gray)
    for ksize in [25, 35, 50]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        result = cv2.max(result, tophat)

    _, binary = cv2.threshold(result, tophat_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary


# ── OpenCV 통합 감지 ──────────────────────────────────
def detect_opencv(image_array, tophat_thresh, green_sensitivity, min_area, max_area, min_circ, dish_margin):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    dish_mask, dish_info = get_dish_mask(gray, dish_margin)

    # 녹색 + 흰색 콜로니 각각 감지
    green_binary = detect_green_colonies(image_array, green_sensitivity)
    white_binary = detect_white_colonies(image_array, tophat_thresh)

    # 합치기 (OR)
    combined = cv2.bitwise_or(green_binary, white_binary)

    # 디시 마스크 적용
    combined = cv2.bitwise_and(combined, dish_mask)

    # 연결된 컴포넌트로 콜로니 감지
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image_array.copy()
    count = 0
    rejected = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            rejected += 1
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circ:
            rejected += 1
            continue

        count += 1
        (x, y), r = cv2.minEnclosingCircle(cnt)

        # 녹색 콜로니면 초록색, 흰색 콜로니면 노란색으로 표시
        cx_int, cy_int = int(x), int(y)
        roi_mask = np.zeros(green_binary.shape, dtype=np.uint8)
        cv2.drawContours(roi_mask, [cnt], -1, 255, -1)
        is_green = cv2.bitwise_and(green_binary, roi_mask).sum() > cv2.bitwise_and(white_binary, roi_mask).sum()

        color = (57, 211, 83) if is_green else (255, 220, 50)
        cv2.circle(result, (cx_int, cy_int), int(r) + 4, color, 3)
        cv2.circle(result, (cx_int, cy_int), 4, color, -1)

        # 번호 표시 (선택: 콜로니가 많으면 주석처리)
        # cv2.putText(result, str(count), (cx_int+int(r)+5, cy_int),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return result, count, dish_info


# ── Cellpose 감지 (기존 유지) ─────────────────────────
@st.cache_resource(show_spinner="🤖 Cellpose 모델 로딩 중...")
def load_cellpose_model():
    from cellpose import models
    if hasattr(models, "CellposeModel"):
        return models.CellposeModel(gpu=False)
    else:
        return models.Cellpose(model_type="cyto3", gpu=False)


def detect_cellpose(image_array, cell_diameter, flow_threshold, cellprob_threshold,
                    min_area, max_area, dish_margin):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    dish_mask, _ = get_dish_mask(gray, dish_margin)

    model = load_cellpose_model()
    eval_result = model.eval(
        image_array,
        diameter=cell_diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=int(min_area)
    )
    masks = eval_result[0]

    result = image_array.copy()
    count = 0
    for label in np.unique(masks):
        if label == 0:
            continue
        cell_mask = np.uint8(masks == label)
        if cv2.bitwise_and(cell_mask, dish_mask).sum() < cell_mask.sum() * 0.5:
            continue
        area = int(cell_mask.sum())
        if area < min_area or area > max_area:
            continue
        count += 1
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (x, y), r = cv2.minEnclosingCircle(cnts[0])
            cv2.circle(result, (int(x), int(y)), max(int(r) + 2, 4), (57, 211, 83), 2)
            cv2.circle(result, (int(x), int(y)), 3, (57, 211, 83), -1)
    return result, count


# ── 메인 ─────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    if use_cellpose:
        h_orig, w_orig = img_array.shape[:2]
        max_dim = 1500
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS))
            st.info(f"📐 이미지 크기 조정: {w_orig}×{h_orig} → {new_w}×{new_h}")

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown("**원본**")
        st.image(img_array, use_container_width=True)

    if use_cellpose:
        try:
            with st.spinner("🤖 Cellpose로 콜로니 감지 중..."):
                result_array, colony_count = detect_cellpose(
                    img_array, cell_diameter, flow_threshold, cellprob_threshold,
                    min_area, max_area, dish_margin
                )
            method_label = "Cellpose AI"
        except Exception as e:
            st.error(f"Cellpose 오류: {e}")
            st.stop()
    else:
        with st.spinner("🔧 OpenCV로 콜로니 감지 중..."):
            result_array, colony_count, dish_info = detect_opencv(
                img_array, tophat_thresh, green_sensitivity,
                min_area, max_area, min_circ, dish_margin
            )
            # 디시 감지 영역 시각화
            if dish_info:
                cx, cy, cr, r_applied = dish_info
                cv2.circle(result_array, (cx, cy), r_applied, (100, 100, 255), 2)
        method_label = "OpenCV"

    with col_result:
        st.markdown("**감지 결과** (🟢 녹색콜로니 / 🟡 흰색콜로니 / 🔵 디시경계)")
        st.image(result_array, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<div class='big-count'>{colony_count}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='count-label'>colonies detected · {method_label}</div>", unsafe_allow_html=True)

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
        file_name=f"colony_{method_label}_{colony_count}.png",
        mime="image/png"
    )

else:
    st.markdown("""
    <div class='card'><p class='tip-text'>
    📌 <strong>모드 선택 가이드</strong><br><br>
    • <strong>🤖 AI 모드 (Cellpose)</strong> — 콜로니가 빽빽하게 붙어있을 때.<br>
    • <strong>🔧 OpenCV 모드</strong> — 녹색 형광 + 흰색 콜로니 각각 감지. 빠르고 가벼움.<br><br>
    📌 <strong>OpenCV 파라미터 가이드</strong><br><br>
    • 가장자리 콜로니 누락 → <strong>테두리 반사 제거 비율 높이기</strong> (0.93 → 0.97)<br>
    • 녹색 콜로니가 안 잡힐 때 → <strong>녹색 형광 민감도 낮추기</strong><br>
    • 흰색 콜로니가 안 잡힐 때 → <strong>흰색 콜로니 감지 민감도 낮추기</strong><br>
    • 노이즈가 많이 잡힐 때 → <strong>최소 콜로니 면적 높이기</strong>
    </p></div>
    """, unsafe_allow_html=True)
