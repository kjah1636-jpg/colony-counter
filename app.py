import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Colony Counter Pro", page_icon="🔬", layout="wide")

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
.stButton > button:hover { background-color: #2ea043; transform: translateY(-1px); }
hr { border-color: #30363d; }
.tip-text { color: #8b949e; font-size: 13px; line-height: 1.6; }
.warn-text { color: #f0883e; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔬 Colony Counter Pro")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ════════════════════════════════════════════════════════════════════════════

def get_dish_mask(gray, margin):
    """허프 변환으로 페트리 디시 경계 자동 감지"""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    circles = None
    for p2 in [50, 40, 30, 20, 15, 10, 8]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=p2,
            minRadius=int(min(h, w) * 0.28),
            maxRadius=int(min(h, w) * 0.58)
        )
        if circles is not None:
            break
    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        cx, cy, cr = np.round(circles[0][0]).astype(int)
        r_applied = min(int(cr * margin), min(h, w) // 2 - 5)
        cv2.circle(mask, (cx, cy), r_applied, 255, -1)
        return mask, (cx, cy, cr, r_applied), True
    else:
        # 감지 실패 시 이미지 중앙 기준 추정
        cx, cy = w // 2, h // 2
        r_applied = int(min(h, w) * 0.44)
        cv2.circle(mask, (cx, cy), r_applied, 255, -1)
        return mask, (cx, cy, r_applied, r_applied), False


def tophat_correction(enhanced, r_d):
    """
    [핵심 개선] 모폴로지 Top-hat 변환으로 불균일 배경 조명 제거.

    조명이 한쪽에서 들어오거나 그라데이션이 있어도,
    콜로니 크기보다 큰 구조(배경)를 제거하고 콜로니만 강조합니다.
    - White Top-hat : 밝은 콜로니 (어두운 배지) 강조
    - Black Top-hat : 어두운 콜로니 (밝은 배지) 강조
    """
    ksize = max(int(r_d * 0.13), 15) * 2 + 1   # 홀수 보장
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    white_th = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT,  kernel)
    black_th = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    return white_th, black_th


def auto_detect_colony_mode(gray, dish_mask):
    """
    [핵심 개선] 배지 영역 평균 밝기로 콜로니 극성 자동 판단.

    - 배지 평균 > 130 : 밝은 배지 → 어두운 콜로니 (Black Top-hat)
    - 배지 평균 ≤ 130 : 어두운 배지 → 밝은 콜로니 (White Top-hat)
    """
    vals = gray[dish_mask > 0]
    if len(vals) == 0:
        return "bright"
    return "dark" if float(np.mean(vals)) > 130 else "bright"


def watershed_segmentation(binary_clean, img_rgb, dist_thresh_ratio):
    """
    [핵심 개선] Marker-based Watershed: 붙어있는 콜로니를 개별 영역으로 분리.

    기존의 단순 Distance Transform + 임계값 방식 대비:
    - Sure foreground (콜로니 중심) / Sure background / Unknown 영역을 분리
    - OpenCV watershed로 경계 정확히 결정
    Returns: (labels, dist_map)
    """
    dist = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return None, dist

    # Sure foreground: 거리 맵에서 콜로니 중심 추출
    _, sure_fg = cv2.threshold(dist, dist_thresh_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background: 팽창으로 확장
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(binary_clean, k7, iterations=3)

    # Unknown = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 연결 요소 레이블링으로 초기 마커 생성
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1           # 배경을 1로 올리고
    markers[unknown == 255] = 0     # Unknown 영역 = 0 (watershed가 결정)

    # Watershed 적용
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    labels = cv2.watershed(img_bgr, markers.copy())
    return labels, dist


def detect_colonies(image_array, params):
    clahe_clip   = params['clahe_clip']
    adapt_C      = params['adapt_C']
    dist_thresh  = params['dist_thresh']
    min_area_pct = params['min_area_pct']   # 접시 면적 대비 %
    max_area_pct = params['max_area_pct']
    min_circ     = params['min_circ']
    dish_margin  = params['dish_margin']
    colony_mode  = params['colony_mode']
    use_color    = params['use_color']

    # ── 1. 리사이즈 (1500px 기준으로 통일) ──────────────
    h_o, w_o = image_array.shape[:2]
    scale = 1500 / max(h_o, w_o)
    if scale < 1.0:
        img = np.array(Image.fromarray(image_array).resize(
            (int(w_o * scale), int(h_o * scale)), Image.LANCZOS))
    else:
        img = image_array.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── 2. 페트리 디시 마스크 ────────────────────────────
    dish_mask, dish_info, dish_detected = get_dish_mask(gray, dish_margin)
    cx_d, cy_d, _, r_d = dish_info
    dish_pixel_area = np.pi * r_d ** 2

    # [핵심 개선] 면적 임계값을 접시 픽셀 면적 대비 %로 변환
    # → 사진 해상도나 촬영 거리가 달라져도 일관된 결과
    min_area = (min_area_pct / 100.0) * dish_pixel_area
    max_area = (max_area_pct / 100.0) * dish_pixel_area

    # ── 3. CLAHE 대비 향상 ────────────────────────────────
    clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe_obj.apply(gray)

    # ── 4. [신규] Top-hat 배경 보정 ─────────────────────
    white_th, black_th = tophat_correction(enhanced, r_d)

    # ── 5. [신규] 콜로니 극성 판단 ──────────────────────
    if colony_mode == "자동 감지":
        mode = auto_detect_colony_mode(gray, dish_mask)
    elif "밝은" in colony_mode:
        mode = "bright"
    else:
        mode = "dark"

    tophat_ch = white_th if mode == "bright" else black_th
    base_ch   = enhanced if mode == "bright" else cv2.bitwise_not(enhanced)

    # ── 6. [개선] 다채널 합성 ────────────────────────────
    # Top-hat(배경 제거) + 원본 밝기 혼합
    working = cv2.addWeighted(tophat_ch, 0.65, base_ch, 0.35, 0)
    if use_color:
        # [신규] HSV 채도 채널 추가 — 콜로니-배지 색 차이 활용
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        working = cv2.addWeighted(working, 0.75, sat, 0.25, 0)

    # ── 7. 멀티스케일 적응형 이진화 ──────────────────────
    binary_union = np.zeros(gray.shape, dtype=np.uint8)
    for bs in [21, 31, 51, 71]:
        b = cv2.adaptiveThreshold(
            working, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs, adapt_C
        )
        b[dish_mask == 0] = 0
        binary_union = cv2.bitwise_or(binary_union, b)

    # ── 8. 모폴로지 정리 ─────────────────────────────────
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary_union, cv2.MORPH_OPEN,  k3, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, k5, iterations=2)
    binary_clean[dish_mask == 0] = 0

    # ── 9. [핵심 개선] Watershed 분리 ───────────────────
    labels, dist_map = watershed_segmentation(binary_clean, img, dist_thresh)

    # ── 10. 콜로니 필터링 ────────────────────────────────
    colonies = []

    if labels is not None:
        for label_id in np.unique(labels):
            if label_id <= 1:   # 배경(1) 및 경계(-1) 제외
                continue
            component = np.uint8(labels == label_id) * 255
            cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ < min_circ:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            size = ('small'  if area < dish_pixel_area * 0.0005 else
                    'medium' if area < dish_pixel_area * 0.003  else 'large')
            colonies.append({
                'id':   len(colonies) + 1,
                'x':    int(x),  'y': int(y),
                'r':    max(int(r) + 3, 5),
                'area': float(area),
                'circ': round(float(circ), 3),
                'size': size
            })
    else:
        # Fallback: Watershed 실패 시 Distance Transform 방식
        if dist_map.max() > 0:
            _, sfg = cv2.threshold(dist_map, dist_thresh * dist_map.max(), 255, 0)
            sfg = np.uint8(sfg)
            cnts, _ = cv2.findContours(sfg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if not (min_area <= area <= max_area):
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circ = 4 * np.pi * area / (peri ** 2)
                if circ < min_circ:
                    continue
                (x, y), r = cv2.minEnclosingCircle(cnt)
                size = ('small'  if area < dish_pixel_area * 0.0005 else
                        'medium' if area < dish_pixel_area * 0.003  else 'large')
                colonies.append({
                    'id':   len(colonies) + 1,
                    'x':    int(x),  'y': int(y),
                    'r':    max(int(r) + 3, 5),
                    'area': float(area),
                    'circ': round(float(circ), 3),
                    'size': size
                })

    return img, colonies, binary_clean, tophat_ch, dish_info, dish_detected, dist_map


def draw_result(img, colonies, dish_info, excluded_ids=None):
    """콜로니 번호·마킹 표시. 제외된 콜로니는 빨간 X로 표시."""
    if excluded_ids is None:
        excluded_ids = set()
    result = img.copy()
    cx_d, cy_d, _, r_d = dish_info
    color_map = {
        'small':  (255, 180,  50),
        'medium': ( 57, 211,  83),
        'large':  (100, 180, 255)
    }
    for col in colonies:
        x, y, r = col['x'], col['y'], col['r']
        if col['id'] in excluded_ids:
            # 제외: 빨간 X 표시
            cv2.circle(result, (x, y), r, (200, 50, 50), 1)
            cv2.line(result, (x - r//2, y - r//2), (x + r//2, y + r//2), (200, 50, 50), 2)
            cv2.line(result, (x + r//2, y - r//2), (x - r//2, y + r//2), (200, 50, 50), 2)
            continue
        color = color_map[col['size']]
        cv2.circle(result, (x, y), r, color, 2)
        cv2.circle(result, (x, y), 3, color, -1)
        # 번호 (콜로니 크기에 비례하는 폰트)
        fs = max(0.25, min(0.50, r / 18.0))
        cv2.putText(result, str(col['id']),
                    (x - 6, y - r - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    # 접시 경계
    cv2.circle(result, (cx_d, cy_d), r_d, (150, 150, 255), 2)
    return result


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

uploaded = st.file_uploader(
    "이미지 업로드 (JPG / PNG / TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

pcol1, pcol2 = st.columns(2)
with pcol1:
    with st.expander("⚙️ 기본 파라미터", expanded=True):
        st.markdown("<p class='tip-text'>일반적으로 기본값에서 시작하세요.</p>",
                    unsafe_allow_html=True)
        dish_margin = st.slider("테두리 감지 범위", 0.80, 0.98, 0.93, 0.01,
            help="높일수록 가장자리 콜로니까지 더 감지")
        adapt_C     = st.slider("감지 민감도 (C)", -20, -1, -5, 1,
            help="낮을수록 더 민감하게 감지 (노이즈 증가 가능). 기본 -5")
        clahe_clip  = st.slider("대비 향상 강도", 1.0, 8.0, 3.0, 0.5,
            help="높일수록 흐릿한 콜로니 보정")
        dist_thresh = st.slider("콜로니 분리 민감도", 0.10, 0.50, 0.25, 0.05,
            help="낮을수록 붙어있는 콜로니 더 잘 분리")

with pcol2:
    with st.expander("⚙️ 고급 파라미터", expanded=True):
        st.markdown("<p class='tip-text'>콜로니 유형·크기 설정</p>",
                    unsafe_allow_html=True)
        colony_mode = st.selectbox(
            "콜로니 유형",
            ["자동 감지", "밝은 콜로니 (어두운 배지)", "어두운 콜로니 (밝은 배지)"],
            help="자동 감지가 안 될 때 수동 선택")
        use_color   = st.checkbox("색상(채도) 채널 활용", value=True,
            help="콜로니와 배지 사이 색 차이가 있을 때 감지 정확도 향상")
        min_circ    = st.slider("최소 원형도", 0.10, 0.70, 0.25, 0.05,
            help="낮을수록 불규칙한 모양도 감지")

        st.markdown("""<p class='tip-text'>
        ⚠️ <b>크기 임계값</b>은 접시 면적 대비 %로 설정됩니다.<br>
        → 사진 해상도나 촬영 거리가 달라져도 일관됩니다.
        </p>""", unsafe_allow_html=True)
        min_area_pct = st.slider("최소 콜로니 크기 (접시면적 %)",
            0.001, 0.500, 0.010, 0.001, format="%.3f",
            help="0.010% ≈ 약 133px² (1500px 기준)")
        max_area_pct = st.slider("최대 콜로니 크기 (접시면적 %)",
            0.5, 20.0, 3.0, 0.5, format="%.1f",
            help="3.0% ≈ 약 40,000px²")

if uploaded:
    image     = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)
    h_orig, w_orig = img_array.shape[:2]

    params = dict(
        clahe_clip=clahe_clip, adapt_C=adapt_C, dist_thresh=dist_thresh,
        min_area_pct=min_area_pct, max_area_pct=max_area_pct,
        min_circ=min_circ, dish_margin=dish_margin,
        colony_mode=colony_mode, use_color=use_color
    )

    with st.spinner("🔬 콜로니 감지 중..."):
        img_res, colonies, binary_clean, tophat_img, dish_info, dish_ok, dist_map = \
            detect_colonies(img_array, params)

    if not dish_ok:
        st.warning(
            "⚠️ 페트리 디시 경계 자동 감지 실패 — "
            "이미지에 디시 전체가 포함되었는지 확인하거나, "
            "테두리 감지 범위를 조정하세요.")

    # ── 수동 보정 ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖊️ 수동 보정")
    st.markdown(
        f"<p class='tip-text'>자동 감지: <b>{len(colonies)}개</b> — "
        "아래 번호 이미지를 확인한 뒤, 잘못 감지된 번호를 쉼표로 입력하면 제외됩니다.</p>",
        unsafe_allow_html=True)
    exclude_str = st.text_input(
        "제외할 콜로니 번호 (예: 3, 7, 12)", value="",
        placeholder="번호 입력 후 Enter — 해당 콜로니가 빨간 X로 표시됩니다")
    excluded_ids: set = set()
    if exclude_str.strip():
        excluded_ids = {int(t) for t in exclude_str.replace(",", " ").split()
                        if t.strip().isdigit()}
    final_count = len([c for c in colonies if c['id'] not in excluded_ids])

    result_img = draw_result(img_res, colonies, dish_info, excluded_ids)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**원본**")
        st.image(img_array, use_container_width=True)
    with c2:
        st.markdown("**감지 결과** 🟡소형 🟢중형 🔵대형")
        st.image(result_img, use_container_width=True)

    # ── 카운트 표시 ───────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div class='big-count'>{final_count}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='count-label'>colonies detected · "
        "Watershed + Top-hat Background Correction</div>",
        unsafe_allow_html=True)

    ma, mb, mc, md = st.columns(4)
    ma.metric("자동 감지",   f"{len(colonies)}개")
    mb.metric("제외됨",      f"{len(excluded_ids & {c['id'] for c in colonies})}개")
    mc.metric("최종 카운트", f"{final_count}개")
    md.metric("원본 크기",   f"{w_orig}×{h_orig}")

    if colonies:
        ns = sum(1 for c in colonies if c['size'] == 'small'  and c['id'] not in excluded_ids)
        nm = sum(1 for c in colonies if c['size'] == 'medium' and c['id'] not in excluded_ids)
        nl = sum(1 for c in colonies if c['size'] == 'large'  and c['id'] not in excluded_ids)
        st.caption(f"크기 분포: 🟡소형 {ns}개 · 🟢중형 {nm}개 · 🔵대형 {nl}개")

    # ── 디버그 ────────────────────────────────────────────
    with st.expander("🔍 디버그: 중간 처리 결과", expanded=False):
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Top-hat 배경 보정 결과**")
            st.image(tophat_img, use_container_width=True,
                     caption="콜로니 위치에 밝은 점이 있어야 정상입니다.")
        with d2:
            st.markdown("**이진화 + 모폴로지**")
            st.image(binary_clean, use_container_width=True,
                     caption="흰 영역 = 콜로니 후보 영역")
        with d3:
            st.markdown("**Distance Transform**")
            dist_vis = np.zeros_like(dist_map, dtype=np.uint8)
            if dist_map.max() > 0:
                dist_vis = (dist_map / dist_map.max() * 255).astype(np.uint8)
            st.image(dist_vis, use_container_width=True,
                     caption="밝을수록 콜로니 중심에 가까움")

    # ── 다운로드 ─────────────────────────────────────────
    st.markdown("---")
    buf = io.BytesIO()
    Image.fromarray(result_img).save(buf, format="PNG")
    st.download_button(
        "📥 결과 이미지 다운로드",
        buf.getvalue(),
        f"colony_result_{final_count}.png",
        "image/png"
    )

    with st.expander("📌 파라미터 조정 가이드", expanded=False):
        st.markdown("""
        <div class='card'><p class='tip-text'>
        • 콜로니가 <strong>덜 잡힐 때</strong><br>
        &nbsp;&nbsp;→ 감지 민감도(C) 낮추기 (-5 → -10) / 최소 크기 % 낮추기<br><br>
        • 노이즈가 <strong>많이 잡힐 때</strong><br>
        &nbsp;&nbsp;→ 감지 민감도 높이기 (-5 → -3) / 최소 크기 % 높이기 / 원형도 높이기<br><br>
        • <strong>콜로니 유형 잘못 판단</strong>될 때<br>
        &nbsp;&nbsp;→ '밝은/어두운 콜로니' 직접 선택<br><br>
        • <strong>붙어있는 콜로니</strong>가 하나로 잡힐 때<br>
        &nbsp;&nbsp;→ 콜로니 분리 민감도 낮추기 (0.15 이하)<br><br>
        • <strong>가장자리 콜로니 누락</strong><br>
        &nbsp;&nbsp;→ 테두리 감지 범위 높이기 (0.95 이상)<br><br>
        • 🔍 <strong>디버그 탭</strong> → Top-hat 이미지에서 콜로니 위치에 밝은 점이 있는지 먼저 확인
        </p></div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='card'><p class='tip-text'>
    <strong>✅ v2 주요 개선사항</strong><br><br>
    ✅ <strong>Top-hat 배경 보정</strong>: 불균일 조명 자동 제거 — 사진마다 다른 조명에 강건해짐<br>
    ✅ <strong>콜로니 극성 자동 감지</strong>: 밝은/어두운 콜로니 자동 구분<br>
    ✅ <strong>Marker-based Watershed</strong>: 붙어있는 콜로니를 개별 분리 (이전: 단순 Distance Transform)<br>
    ✅ <strong>접시 크기 비례 면적 임계값</strong>: 해상도·촬영 거리 달라져도 동일 결과<br>
    ✅ <strong>색상(채도) 채널 활용</strong>: 콜로니-배지 색 차이가 있으면 감도 향상<br>
    ✅ <strong>수동 보정 기능</strong>: 잘못 감지된 번호를 입력해 제외 가능<br><br>
    📌 <strong>사용 방법</strong><br>
    1. 이미지 업로드 → 기본값으로 먼저 실행<br>
    2. 감지 결과 이미지의 번호 확인<br>
    3. 잘못 감지된 번호를 수동 보정란에 입력
    </p></div>
    """, unsafe_allow_html=True)
