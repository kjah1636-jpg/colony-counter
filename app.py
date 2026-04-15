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
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔬 Colony Counter Pro v3")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# 핵심 함수
# ════════════════════════════════════════════════════════════════════════════

def get_dish_mask(gray, margin):
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
    cx, cy = w // 2, h // 2
    r_applied = int(min(h, w) * 0.44)
    cv2.circle(mask, (cx, cy), r_applied, 255, -1)
    return mask, (cx, cy, r_applied, r_applied), False


def tophat_correction(enhanced, r_d):
    """
    배경 불균일 조명 제거 (Top-hat 변환).
    커널 크기 = 가장 큰 콜로니보다 크게 설정 → 배경 그라데이션만 제거하고 콜로니는 보존.
    """
    ksize = max(int(r_d * 0.13), 15) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    white_th = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT,  kernel)
    black_th = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    return white_th, black_th


def auto_detect_mode(gray, dish_mask):
    vals = gray[dish_mask > 0]
    if len(vals) == 0:
        return "bright"
    return "dark" if float(np.mean(vals)) > 130 else "bright"


def get_size_label(area, dpa):
    if area < dpa * 0.0005:
        return 'small'
    elif area < dpa * 0.003:
        return 'medium'
    return 'large'


def split_blob_watershed(comp_mask, img_rgb, dist_thresh):
    """
    큰 연결 덩어리(붙은 콜로니)를 Watershed로 분리.
    작은 콜로니에는 쓰지 않음 → 작은 콜로니 소실 문제 방지.
    """
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return [comp_mask]

    # ★ 이 blob 내부에서만 상대 임계값 적용 → 작은 blob에서도 피크 보존
    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(comp_mask, k5, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    labels = cv2.watershed(img_bgr, markers.copy())

    result = []
    for lid in np.unique(labels):
        if lid <= 1:
            continue
        result.append(np.uint8(labels == lid) * 255)
    return result if result else [comp_mask]


def detect_colonies(image_array, params):
    clahe_clip   = params['clahe_clip']
    adapt_C      = params['adapt_C']
    dist_thresh  = params['dist_thresh']
    min_area_px  = params['min_area_px']   # 픽셀 절댓값 (1500px 기준)
    max_area_pct = params['max_area_pct']  # 접시 면적 대비 %
    min_circ     = params['min_circ']
    dish_margin  = params['dish_margin']
    colony_mode  = params['colony_mode']
    use_color    = params['use_color']

    # ── 1. 리사이즈 ──────────────────────────────────────
    h_o, w_o = image_array.shape[:2]
    scale = 1500 / max(h_o, w_o)
    if scale < 1.0:
        img = np.array(Image.fromarray(image_array).resize(
            (int(w_o * scale), int(h_o * scale)), Image.LANCZOS))
    else:
        img = image_array.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── 2. 페트리 디시 마스크 ────────────────────────────
    dish_mask, dish_info, dish_ok = get_dish_mask(gray, dish_margin)
    cx_d, cy_d, _, r_d = dish_info
    dish_pixel_area = np.pi * r_d ** 2

    min_area = float(min_area_px)
    max_area = (max_area_pct / 100.0) * dish_pixel_area

    # ── 3. CLAHE + Top-hat 배경 보정 ─────────────────────
    clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe_obj.apply(gray)
    white_th, black_th = tophat_correction(enhanced, r_d)

    # ── 4. 콜로니 극성 판단 ──────────────────────────────
    if colony_mode == "자동 감지":
        mode = auto_detect_mode(gray, dish_mask)
    elif "밝은" in colony_mode:
        mode = "bright"
    else:
        mode = "dark"

    tophat_ch = white_th if mode == "bright" else black_th
    base_ch   = enhanced if mode == "bright" else cv2.bitwise_not(enhanced)

    # ── 5. 채널 합성 ─────────────────────────────────────
    working = cv2.addWeighted(tophat_ch, 0.65, base_ch, 0.35, 0)
    if use_color:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        working = cv2.addWeighted(working, 0.75, hsv[:, :, 1], 0.25, 0)

    # ── 6. 멀티스케일 적응형 이진화 ──────────────────────
    binary_union = np.zeros(gray.shape, dtype=np.uint8)
    for bs in [21, 31, 51, 71]:
        b = cv2.adaptiveThreshold(
            working, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, bs, adapt_C)
        b[dish_mask == 0] = 0
        binary_union = cv2.bitwise_or(binary_union, b)

    # ── 7. 모폴로지 정리 ─────────────────────────────────
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary_union, cv2.MORPH_OPEN,  k3, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, k5, iterations=2)
    binary_clean[dish_mask == 0] = 0

    # ── 8. Hybrid 감지: 작은 콜로니는 직접, 큰 덩어리는 Watershed ──
    #
    # ★ v2 문제 핵심 수정:
    #   v2는 전체 이미지에 Watershed를 적용 → 전역 Distance Transform 임계값이
    #   작은 콜로니 피크를 날려버려 감지 실패.
    #
    #   v3는 연결된 덩어리(blob) 단위로 처리:
    #   - 작은/중간 blob → 직접 윤곽선 검출 (소실 없음)
    #   - 큰 blob (여러 콜로니가 붙은 것) → 해당 blob 내부에서만 Watershed
    #     (로컬 상대 임계값 → 작은 피크도 보존)

    split_threshold = max_area * 1.5   # 이보다 크면 복수 콜로니로 간주

    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        binary_clean, connectivity=8)

    candidate_masks = []
    for i in range(1, num_labels):
        blob_area = int(stats[i, cv2.CC_STAT_AREA])
        if blob_area < min_area * 0.5:   # 너무 작은 노이즈 → 스킵
            continue
        comp_mask = np.uint8(labels_map == i) * 255

        if blob_area > split_threshold:
            # 큰 덩어리: Watershed로 분리
            sub = split_blob_watershed(comp_mask, img, dist_thresh)
            candidate_masks.extend(sub)
        else:
            candidate_masks.append(comp_mask)

    # ── 9. 윤곽선 추출 및 필터링 ─────────────────────────
    colonies = []
    for mask in candidate_masks:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        colonies.append({
            'id':   len(colonies) + 1,
            'x':    int(x),  'y': int(y),
            'r':    max(int(r) + 3, 5),
            'area': float(area),
            'circ': round(float(circ), 3),
            'size': get_size_label(area, dish_pixel_area)
        })

    # Distance Transform (디버그용)
    dist_map = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)

    return img, colonies, binary_clean, tophat_ch, dish_info, dish_ok, dist_map


def draw_result(img, colonies, dish_info, excluded_ids=None):
    if excluded_ids is None:
        excluded_ids = set()
    result = img.copy()
    cx_d, cy_d, _, r_d = dish_info
    color_map = {'small': (255, 180, 50), 'medium': (57, 211, 83), 'large': (100, 180, 255)}

    for col in colonies:
        x, y, r = col['x'], col['y'], col['r']
        if col['id'] in excluded_ids:
            cv2.circle(result, (x, y), r, (200, 50, 50), 1)
            cv2.line(result, (x - r//2, y - r//2), (x + r//2, y + r//2), (200, 50, 50), 2)
            cv2.line(result, (x + r//2, y - r//2), (x - r//2, y + r//2), (200, 50, 50), 2)
            continue
        color = color_map[col['size']]
        cv2.circle(result, (x, y), r, color, 2)
        cv2.circle(result, (x, y), 3, color, -1)
        fs = max(0.25, min(0.50, r / 18.0))
        cv2.putText(result, str(col['id']),
                    (x - 6, y - r - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

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
        dish_margin = st.slider("테두리 감지 범위", 0.80, 0.98, 0.93, 0.01,
            help="높일수록 가장자리 콜로니까지 감지")
        adapt_C     = st.slider("감지 민감도 (C)", -20, -1, -7, 1,
            help="낮을수록 더 민감 (노이즈 증가 가능). 기본 -7")
        clahe_clip  = st.slider("대비 향상 강도", 1.0, 8.0, 3.0, 0.5)
        dist_thresh = st.slider("콜로니 분리 민감도", 0.10, 0.50, 0.20, 0.05,
            help="낮을수록 붙어있는 콜로니 더 잘 분리 (큰 덩어리에만 적용됨)")

with pcol2:
    with st.expander("⚙️ 고급 파라미터", expanded=True):
        colony_mode = st.selectbox("콜로니 유형",
            ["자동 감지", "밝은 콜로니 (어두운 배지)", "어두운 콜로니 (밝은 배지)"],
            help="자동이 안 될 때 직접 선택")
        use_color   = st.checkbox("색상(채도) 채널 활용", value=True)
        min_circ    = st.slider("최소 원형도", 0.10, 0.70, 0.20, 0.05,
            help="낮을수록 불규칙 모양도 감지")

        st.markdown("""<p style='color:#8b949e;font-size:13px'>
        <b>최소 크기</b>는 1500px 리사이즈 기준 픽셀 면적 (px²)입니다.<br>
        작은 점까지 잡으려면 낮추세요 (기본 20).
        </p>""", unsafe_allow_html=True)
        min_area_px  = st.slider("최소 콜로니 면적 (px²)", 5, 200, 20, 5,
            help="1500px 리사이즈 기준. 항상 일정한 스케일.")
        max_area_pct = st.slider("최대 콜로니 크기 (접시면적 %)", 0.5, 20.0, 4.0, 0.5,
            format="%.1f",
            help="접시 크기 대비 %. 해상도가 달라져도 일관됨.")

if uploaded:
    image     = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)
    h_orig, w_orig = img_array.shape[:2]

    params = dict(
        clahe_clip=clahe_clip, adapt_C=adapt_C, dist_thresh=dist_thresh,
        min_area_px=min_area_px, max_area_pct=max_area_pct,
        min_circ=min_circ, dish_margin=dish_margin,
        colony_mode=colony_mode, use_color=use_color
    )

    with st.spinner("🔬 콜로니 감지 중..."):
        img_res, colonies, binary_clean, tophat_img, dish_info, dish_ok, dist_map = \
            detect_colonies(img_array, params)

    if not dish_ok:
        st.warning("⚠️ 페트리 디시 경계 자동 감지 실패 — 테두리 감지 범위를 조정하거나 이미지에 디시 전체가 포함되었는지 확인하세요.")

    # ── 수동 보정 ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖊️ 수동 보정")
    st.markdown(
        f"<p style='color:#8b949e;font-size:13px'>자동 감지: <b>{len(colonies)}개</b> — "
        "아래 번호 이미지 확인 후 잘못된 번호를 쉼표로 입력하세요.</p>",
        unsafe_allow_html=True)
    exclude_str = st.text_input("제외할 콜로니 번호 (예: 3, 7, 12)", value="",
                                 placeholder="Enter 후 적용")
    excluded_ids: set = set()
    if exclude_str.strip():
        excluded_ids = {int(t) for t in exclude_str.replace(",", " ").split()
                        if t.strip().isdigit()}
    valid_excluded = excluded_ids & {c['id'] for c in colonies}
    final_count = len(colonies) - len(valid_excluded)

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
        "<div class='count-label'>colonies detected · Hybrid Detection + Top-hat Correction</div>",
        unsafe_allow_html=True)

    ma, mb, mc, md = st.columns(4)
    ma.metric("자동 감지",   f"{len(colonies)}개")
    mb.metric("제외됨",      f"{len(valid_excluded)}개")
    mc.metric("최종 카운트", f"{final_count}개")
    md.metric("원본 크기",   f"{w_orig}×{h_orig}")

    if colonies:
        ns = sum(1 for c in colonies if c['size']=='small'  and c['id'] not in excluded_ids)
        nm = sum(1 for c in colonies if c['size']=='medium' and c['id'] not in excluded_ids)
        nl = sum(1 for c in colonies if c['size']=='large'  and c['id'] not in excluded_ids)
        st.caption(f"크기 분포: 🟡소형 {ns}개 · 🟢중형 {nm}개 · 🔵대형 {nl}개")

    # ── 디버그 ────────────────────────────────────────────
    with st.expander("🔍 디버그: 중간 처리 결과", expanded=False):
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Top-hat 배경 보정**")
            st.image(tophat_img, use_container_width=True,
                     caption="콜로니 위치에 밝은 점이 있어야 함")
        with d2:
            st.markdown("**이진화 결과**")
            st.image(binary_clean, use_container_width=True,
                     caption="흰 영역 = 콜로니 후보")
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
    st.download_button("📥 결과 이미지 다운로드", buf.getvalue(),
                       f"colony_result_{final_count}.png", "image/png")

    with st.expander("📌 파라미터 조정 가이드", expanded=False):
        st.markdown("""
        <div class='card'><p class='tip-text'>
        🔍 먼저 <strong>디버그 → 이진화 결과</strong>에서 흰 점이 작은 콜로니 위치에 있는지 확인!<br><br>
        • 작은 콜로니가 <strong>안 잡힐 때</strong><br>
        &nbsp;&nbsp;→ 감지 민감도(C) 낮추기 (-7 → -12) + 최소 면적 낮추기 (20 → 10)<br><br>
        • 노이즈(먼지, 기포)가 <strong>많이 잡힐 때</strong><br>
        &nbsp;&nbsp;→ 감지 민감도 높이기 (-7 → -4) + 최소 면적 높이기 + 원형도 높이기<br><br>
        • 콜로니가 <strong>너무 많이 나뉘어</strong> 잡힐 때<br>
        &nbsp;&nbsp;→ 콜로니 분리 민감도 높이기 (0.30 이상)<br><br>
        • <strong>붙어있는 콜로니</strong>가 하나로 잡힐 때<br>
        &nbsp;&nbsp;→ 콜로니 분리 민감도 낮추기 (0.10)<br><br>
        • 사진마다 결과 다를 때<br>
        &nbsp;&nbsp;→ 콜로니 유형 ('밝은/어두운') 수동 선택
        </p></div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='card'><p class='tip-text'>
    <strong>v3 핵심 개선</strong> (v2 대비)<br><br>
    ✅ <strong>Hybrid 감지 방식</strong><br>
    &nbsp;&nbsp;작은 콜로니 → 직접 윤곽선 검출 (소실 없음)<br>
    &nbsp;&nbsp;붙어있는 큰 덩어리 → Blob 내부에서만 Watershed (로컬 임계값)<br><br>
    ✅ <strong>최소 면적 단위 변경</strong>: 픽셀(px²) 절댓값 → 더 직관적<br>
    ✅ <strong>기본값 최적화</strong>: 작은 콜로니도 기본값에서 감지 가능<br>
    ✅ <strong>Top-hat 배경 보정</strong> 유지: 사진 달라져도 안정적<br><br>
    📌 사용 방법: 이미지 업로드 → 기본값 실행 → 번호 확인 → 수동 보정
    </p></div>
    """, unsafe_allow_html=True)
