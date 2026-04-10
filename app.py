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
.mode-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-left: 8px; }
.mode-ai { background: #1f3a5f; color: #58a6ff; }
.mode-cv { background: #1a3a1a; color: #39d353; }
div[data-testid="stFileUploader"] { background: #161b22; border: 2px dashed #30363d; border-radius: 12px; padding: 20px; }
div[data-testid="stFileUploader"]:hover { border-color: #39d353; }
.stButton > button { background-color: #238636; color: white; border: none; border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 14px; padding: 12px 28px; width: 100%; transition: all 0.2s; }
.stButton > button:hover { background-color: #2ea043; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(57,211,83,0.3); }
hr { border-color: #30363d; }
.tip-text { color: #8b949e; font-size: 13px; line-height: 1.6; }
.warn-box { background: #2d1f00; border: 1px solid #f0883e; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #f0883e; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── 모드 선택 ─────────────────────────────────────────
st.markdown("# 🔬 Colony Counter")

mode = st.radio(
    "감지 방식 선택",
    ["🤖 AI 모드 (Cellpose — 밀집 콜로니 추천)", "🔧 OpenCV 모드 (빠름 — 간격 넓은 콜로니 추천)"],
    horizontal=True
)
use_cellpose = mode.startswith("🤖")

if use_cellpose:
    st.markdown("<p class='tip-text'>Cellpose 딥러닝 모델로 붙어있는 콜로니도 분리해서 카운팅합니다.<br>⚠️ 첫 실행 시 모델 다운로드로 1~2분 걸릴 수 있습니다.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='tip-text'>OpenCV Top-hat 알고리즘으로 빠르게 카운팅합니다. 콜로니 간격이 넓을 때 정확합니다.</p>", unsafe_allow_html=True)

st.markdown("---")

uploaded = st.file_uploader("이미지 업로드 (JPG / PNG / TIFF)", type=["jpg","jpeg","png","tif","tiff"])

# ── 파라미터 ──────────────────────────────────────────
with st.expander("⚙️ 감지 파라미터 조정", expanded=False):
    st.markdown("<p class='tip-text'>결과가 잘 안 맞을 때 조절해 보세요.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        dish_margin = st.slider("테두리 반사 제거 비율", 0.75, 0.97, 0.86, step=0.01,
            help="낮출수록 테두리를 더 많이 제외합니다.")
        if use_cellpose:
            cell_diameter = st.slider("콜로니 평균 지름 (px)", 5, 60, 15,
                help="콜로니 하나의 대략적인 픽셀 크기. 작으면 낮추고 크면 높이세요.")
        else:
            tophat_thresh = st.slider("감지 민감도", 5, 40, 18,
                help="낮을수록 더 많이 감지합니다.")
    with col2:
        min_area = st.slider("최소 콜로니 면적 (px²)", 10, 500, 250 if not use_cellpose else 30,
            help="이보다 작은 점은 무시합니다.")
        max_area = st.slider("최대 콜로니 면적 (px²)", 500, 30000, 15000,
            help="이보다 큰 영역은 무시합니다.")
        if not use_cellpose:
            min_circ = st.slider("최소 원형도", 0.1, 0.8, 0.35, step=0.05,
                help="1.0이 완전한 원. 낮을수록 불규칙한 모양도 감지.")

# ── OpenCV 함수 ───────────────────────────────────────
def get_dish_mask(gray, margin):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                                param1=50, param2=30,
                                minRadius=min(h,w)//4, maxRadius=min(h,w)//2)
    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        cx, cy, cr = np.round(circles[0][0]).astype(int)
        cv2.circle(mask, (cx, cy), int(cr * margin), 255, -1)
        return mask, (cx, cy, cr)
    else:
        mask[:] = 255
        return mask, None

def detect_opencv(image_array, tophat_thresh, min_area, max_area, min_circ, dish_margin):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mask, dish = get_dish_mask(gray, dish_margin)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    _, binary = cv2.threshold(tophat, tophat_thresh, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)
    binary = cv2.bitwise_and(binary, mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image_array.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        if 4 * np.pi * area / (perim ** 2) < min_circ:
            continue
        count += 1
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(result, (int(x), int(y)), int(r) + 3, (57, 211, 83), 3)
        cv2.circle(result, (int(x), int(y)), 4, (57, 211, 83), -1)
    return result, count

# ── Cellpose 함수 ─────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Cellpose 모델 로딩 중... (첫 실행만 오래 걸립니다)")
def load_cellpose_model():
    from cellpose import models
    return models.Cellpose(model_type="cyto3", gpu=False)

def detect_cellpose(image_array, cell_diameter, min_area, max_area, dish_margin):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mask_dish, dish = get_dish_mask(gray, dish_margin)

    model = load_cellpose_model()

    # Cellpose는 RGB 입력
    img_input = image_array.copy()

    masks, flows, styles, diams = model.eval(
        img_input,
        diameter=cell_diameter,
        channels=[0, 0],       # grayscale
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=int(min_area)
    )

    result = image_array.copy()
    count = 0
    unique_labels = np.unique(masks)

    for label in unique_labels:
        if label == 0:
            continue
        cell_mask = np.uint8(masks == label)

        # 디시 밖 제외
        overlap = cv2.bitwise_and(cell_mask, mask_dish)
        if overlap.sum() < cell_mask.sum() * 0.5:
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

# ── 메인 실행 ─────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    # 큰 이미지는 리사이즈 (Cellpose 메모리 절약)
    h_orig, w_orig = img_array.shape[:2]
    max_dim = 1500
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS))
        st.info(f"📐 이미지가 커서 {w_orig}×{h_orig} → {new_w}×{new_h}로 축소했습니다.")

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown("**원본**")
        st.image(img_array, use_container_width=True)

    if use_cellpose:
        try:
            with st.spinner("🤖 Cellpose로 콜로니 감지 중..."):
                result_array, colony_count = detect_cellpose(
                    img_array, cell_diameter, min_area, max_area, dish_margin)
            method_label = "Cellpose AI"
        except Exception as e:
            st.error(f"Cellpose 오류: {e}\n\nOpenCV 모드로 전환해서 시도해보세요.")
            st.stop()
    else:
        with st.spinner("🔧 OpenCV로 콜로니 감지 중..."):
            result_array, colony_count = detect_opencv(
                img_array, tophat_thresh, min_area, max_area, min_circ, dish_margin)
        method_label = "OpenCV"

    with col_result:
        st.markdown("**감지 결과**")
        st.image(result_array, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<div class='big-count'>{colony_count}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='count-label'>colonies detected ({method_label})</div>", unsafe_allow_html=True)

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
    • <strong>AI 모드 (Cellpose)</strong> — 콜로니가 빽빽하게 붙어있을 때. 첫 실행 시 모델 다운로드로 1~2분 소요.<br>
    • <strong>OpenCV 모드</strong> — 콜로니 간격이 충분할 때. 빠르고 가볍게 동작.<br><br>
    📌 <strong>파라미터 가이드</strong><br><br>
    • 콜로니가 덜 잡힐 때 → 감지 민감도 낮추기 / 콜로니 지름 줄이기<br>
    • 노이즈가 잡힐 때 → 최소 면적 높이기 / 테두리 반사 제거 비율 낮추기<br>
    • 붙어있는 콜로니가 하나로 잡힐 때 → AI 모드로 전환
    </p></div>
    """, unsafe_allow_html=True)
