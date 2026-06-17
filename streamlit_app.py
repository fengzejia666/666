"""
双缝干涉条纹间距测量 — Streamlit Web 界面
启动方式: streamlit run streamlit_app.py
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from fringe_core import FringeBridge
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import hashlib

st.set_page_config(page_title="条纹间距测量", layout="wide")
st.title("双缝干涉条纹间距测量工具")

# ---- 初始化 session_state ----
for key, default in [
    ("fb", FringeBridge()),
    ("calib_points", []),
    ("calibrated", False),
    ("last_result", None),
    ("temp_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

fb = st.session_state.fb

# ---- 侧边栏: 上传图片 ----
st.sidebar.header("📁 1. 加载图片")
uploaded = st.sidebar.file_uploader("选择条纹图片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
    suffix = os.path.splitext(uploaded.name)[1] or ".png"
    new_path = os.path.join(tempfile.gettempdir(), f"fringe_{file_hash}{suffix}")

    with open(new_path, "wb") as f:
        f.write(file_bytes)

    if st.session_state.temp_path != new_path:
        try:
            fb.load_image(new_path)
            st.session_state.temp_path = new_path
            st.session_state.calib_points = []
            st.session_state.calibrated = False
            st.session_state.last_result = None
        except Exception as e:
            st.sidebar.error(f"加载失败: {e}")

if fb.has_image:
    w, h = fb.image_size
    st.sidebar.caption(f"尺寸: {w} × {h} px")
    if fb.is_calibrated:
        st.sidebar.caption(f"比例尺: {fb.pixel_per_mm:.2f} px/mm")

# ---- 主体内容 ----
if not fb.has_image:
    st.info("👈 请在左侧上传条纹图片")
    st.stop()

# ==================== 标定: 自定义 canvas 组件 ====================
st.subheader("📏 2. 标定")

col_img, col_ctrl = st.columns([3, 2])

with col_img:
    img_rgb = fb.get_original_image()
    pil_img = Image.fromarray(img_rgb)

    # 限制显示宽度，缩放坐标还原为原始图像像素
    display_w = min(w, 650)
    display_h = int(h * display_w / w)
    scale_x = w / display_w
    scale_y = h / display_h

    canvas = st_canvas(
        background_image=pil_img,
        drawing_mode="point" if not st.session_state.calibrated else "transform",
        stroke_width=3,
        stroke_color="#ff3333",
        point_display_radius=5,
        key="calib_canvas",
        width=display_w,
        height=display_h,
    )

    if not st.session_state.calibrated:
        raw_points = []
        if canvas.json_data is not None:
            objs = canvas.json_data.get("objects", [])
            raw_points = [
                (int(float(o["left"]) * scale_x),
                 int(float(o["top"]) * scale_y))
                for o in objs[:2]
            ]
        if raw_points:
            st.session_state.calib_points = raw_points

with col_ctrl:
    pts = st.session_state.calib_points
    n = len(pts)

    if not st.session_state.calibrated:
        if n == 0:
            st.warning("在左侧图像上点击第 1 个标定点")
        elif n == 1:
            st.info(f"第 1 点: ({pts[0][0]}, {pts[0][1]})")
            st.warning("请点击第 2 个标定点")
        else:
            st.success(f"第 1 点: ({pts[0][0]}, {pts[0][1]})")
            st.success(f"第 2 点: ({pts[1][0]}, {pts[1][1]})")
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            pix_dist = np.sqrt(dx ** 2 + dy ** 2)
            st.metric("像素距离", f"{pix_dist:.1f} px")
    else:
        st.success("标定已完成 ✅")
        st.caption(
            f"点 1: ({pts[0][0]}, {pts[0][1]})  |  "
            f"点 2: ({pts[1][0]}, {pts[1][1]})  |  "
            f"比例尺: {fb.pixel_per_mm:.2f} px/mm"
        )

    # ---- 标定操作按钮 ----
    actual_mm = st.number_input(
        "实际距离 (mm)", min_value=0.001, value=10.0, step=1.0, format="%.3f",
        key="actual_mm"
    )
    col_btn1, col_btn2 = st.columns(2)
    if not st.session_state.calibrated:
        btn_apply = col_btn1.button("✅ 应用标定", key="btn_apply_calib")
        btn_clear = col_btn2.button("🔄 清除", key="btn_clear_calib")
    else:
        btn_apply = col_btn1.button("✅ 应用标定", key="btn_apply_calib")
        btn_clear = col_btn2.button("🔄 重新标定", key="btn_clear_calib")

    if btn_apply and not st.session_state.calibrated and n >= 2:
        try:
            fb.set_calibration(pts[0], pts[1], actual_mm)
            st.session_state.calibrated = True
        except ValueError as e:
            st.error(str(e))

    if btn_clear:
        if st.session_state.calibrated:
            st.session_state.calibrated = False
            st.session_state.calib_points = []
            st.session_state.last_result = None
        else:
            st.session_state.calib_points = []

st.markdown("---")

# ==================== 分析: 始终渲染 ====================
st.subheader("🔬 3. 分析")

col_img2, col_ctrl2 = st.columns([3, 2])

with col_ctrl2:
    st.markdown("**分析参数**")
    clahe_clip = st.slider("对比度增强 (CLAHE)", 0.5, 10.0, 2.0, 0.5, key="clahe_clip")
    blur_size = st.slider("模糊核", 3, 31, 5, 2, key="blur_size")
    prominence = st.slider("峰值灵敏度", 0.005, 0.5, 0.05, 0.005, format="%.3f", key="prominence")
    min_dist = st.slider("最小峰间距 (px)", 3, 200, 10, 1, key="min_dist")

    btn_analyze = st.button("🚀 执行分析", key="btn_analyze")
    if btn_analyze and st.session_state.calibrated:
        try:
            result = fb.analyze(
                clahe_clip=clahe_clip, blur_size=blur_size,
                prominence_ratio=prominence, min_distance=min_dist,
            )
            st.session_state.last_result = result
        except RuntimeError as e:
            st.error(str(e))

    st.markdown("---")

    r = st.session_state.last_result
    if r is not None:
        st.markdown("### 📊 测量结果")
        c1, c2, c3 = st.columns(3)
        c1.metric("条纹数", f"{r['num_fringes']} 条")
        c2.metric("平均间距", f"{r['mean_spacing_mm']:.4f} mm")
        c3.metric("比例尺", f"{r['pixel_per_mm']:.2f} px/mm")
        st.caption(
            f"标准差: {r['std_spacing_mm']:.4f} mm  |  "
            f"像素间距: {r['mean_spacing_px']:.2f} px"
        )
    else:
        if st.session_state.calibrated:
            st.info("调整参数后点击「执行分析」")
        else:
            st.info("请先完成标定（上传图片 → 点击两点 → 输入实际距离 → 应用标定）")

with col_img2:
    r = st.session_state.last_result
    if r is not None:
        st.image(fb.get_marked_image(), caption="分析结果 — 蓝线标记条纹位置", use_column_width=True)

        st.markdown("### 📈 一维投影信号")
        x, y, peaks = fb.get_signal_data()
        if x is not None:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.plot(x, y, "b-", linewidth=1.5, label="投影信号")
            if peaks is not None and len(peaks) > 0:
                ax.plot(peaks, y[peaks], "rx", markersize=6,
                        markeredgewidth=1.5, label=f"峰值 (n={len(peaks)})")
            ax.set_xlabel("像素位置 X")
            ax.set_ylabel("平均亮度")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

        if st.button("📥 导出 TXT", key="btn_export"):
            txt = f"""双缝干涉条纹间距测量结果
{'=' * 50}
图片尺寸: {r['image_size'][0]} x {r['image_size'][1]} px
检测条纹数: {r['num_fringes']}
平均间距: {r['mean_spacing_px']:.2f} px = {r['mean_spacing_mm']:.4f} mm
标准差:   {r['std_spacing_px']:.2f} px = {r['std_spacing_mm']:.4f} mm
比例尺:   {r['pixel_per_mm']:.2f} px/mm

条纹位置 (像素 X):
"""
            for i, p in enumerate(r['peaks']):
                txt += f"  条纹 {i + 1:>3d}: x = {p:.1f}\n"
            txt += "\n间隔明细:\n"
            for i, sp_mm in enumerate(r['spacings_mm']):
                txt += f"  {i + 1:>3d}-{i + 2:>3d}: {sp_mm:.4f} mm  ({r['spacings_px'][i]:.2f} px)\n"
            txt += "=" * 50 + "\n"
            st.download_button("⬇ 下载 TXT", txt, file_name="fringe_results.txt")
    else:
        st.image(fb.get_original_image(), caption="原始图像", use_column_width=True)
