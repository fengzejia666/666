# ======================================================================
# 你的原始代码 (完全保留, 可独立运行)
# ======================================================================
import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, simpledialog
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 全局变量
img = None
points = []
pixel_per_mm = None


def mouse_click(event, x, y, flags, param):
    """鼠标点击事件：用于标定物理尺寸"""
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Calibration", img)


def select_image():
    """打开文件对话框选择图片"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择条纹图片",
        filetypes=[("图片", "*.jpg *.png *.bmp *.jpeg")]
    )
    if not file_path:
        exit()
    return file_path


def calibrate_and_measure(image_path):
    global img, points, pixel_per_mm
    points = []

    if not os.path.exists(image_path):
        print("图片路径不存在！")
        return

    # 读取图片并防止中文路径报错
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("图片读取失败！")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp_img = img.copy()

    # 1. 交互式标定
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_click)
    cv2.imshow("Calibration", img)
    print("请在图像上点击两个点用于标定比例尺，完成后按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration")

    if len(points) < 2:
        print("未完成两点标定，程序退出。")
        return

    root = Tk()
    root.withdraw()
    actual_mm = simpledialog.askfloat("输入", "请输入这两点间的实际长度（毫米）：")
    if actual_mm is None or actual_mm <= 0:
        print("未输入有效的物理长度，程序退出。")
        return

    # 计算比例尺
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    pix_dist = np.sqrt(dx ** 2 + dy ** 2)
    pixel_per_mm = pix_dist / actual_mm

    # 2. 图像预处理 (CLAHE + 高斯模糊)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)

    # 3. 压缩为一维信号并平滑
    hist = cv2.reduce(blur, 0, cv2.REDUCE_AVG).reshape(-1)

    # 4. 核心改进：使用 scipy 的 find_peaks 寻找波峰
    # prominence: 波峰相对于周围谷底的最小突出度 (这里设为极差的 5%，非常自适应)
    # distance: 两个条纹之间的最小像素距离 (防止一个条纹上出现多个假峰)
    prominence_threshold = np.ptp(hist) * 0.05
    peaks, properties = find_peaks(hist, prominence=prominence_threshold, distance=10)

    if len(peaks) < 2:
        print("检测到的条纹数量不足（少于2条），无法计算间距。")
        return

    # 5. 计算结果
    spacings_pix = np.diff(peaks)
    mean_pix = np.mean(spacings_pix)
    mean_mm = mean_pix / pixel_per_mm

    print("\n===== 测量结果 =====")
    print(f"共检测到 {len(peaks)} 条条纹")
    print(f"平均间距: {mean_pix:.2f} 像素 = {mean_mm:.3f} mm")

    # 6. 结果可视化：在原图上画线
    for x in peaks:
        cv2.line(temp_img, (x, 0), (x, temp_img.shape[0]), (0, 0, 255), 1)

    cv2.imshow("Result (Press any key to show signal graph)", temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 7. 信号可视化：展示一维波形和寻峰结果 (便于排错和理解)
    plt.figure(figsize=(10, 4))
    plt.plot(hist, label="1D Projection Signal", color='blue')
    plt.plot(peaks, hist[peaks], "x", color='red', markersize=8, label="Detected Peaks")
    plt.title("Fringe Projection Signal & Peak Detection")
    plt.xlabel("Pixel Position (X-axis)")
    plt.ylabel("Average Brightness")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    image_path = select_image()
    calibrate_and_measure(image_path)


# ======================================================================
# 适配器：把你的原始代码桥接到 PyQt5 前端
# （不修改你上面的任何代码）
# ======================================================================

class FringeBridge:
    """将原始分析逻辑包装成 PyQt5 前端可调用的接口"""

    def __init__(self):
        self._image_bgr = None
        self._image_rgb = None
        self._gray = None
        self._blur_cache = None
        self._hist_cache = None
        self._peaks_cache = None
        self._pixel_per_mm = None
        self._calib_points = []
        self._image_w = 0
        self._image_h = 0

    # ---- 图片加载 (复用你的 imdecode 方式) ----

    def load_image(self, path):
        """加载图片，支持中文路径。同你原代码的加载方式"""
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("图片读取失败，请检查文件路径或格式")
        self._image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._image_h, self._image_w = bgr.shape[:2]
        self._blur_cache = None
        self._hist_cache = None
        self._peaks_cache = None
        self._pixel_per_mm = None
        self._calib_points = []
        return True

    # ---- 属性 ----

    @property
    def has_image(self):
        return self._image_rgb is not None

    @property
    def is_calibrated(self):
        return self._pixel_per_mm is not None

    @property
    def image_size(self):
        return (self._image_w, self._image_h)

    @property
    def pixel_per_mm(self):
        return self._pixel_per_mm

    # ---- 标定 (同你原代码的计算方式) ----

    def set_calibration(self, p1, p2, actual_mm):
        """设置标定参数，同你原代码的像素距离/实际长度计算"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        pix_dist = np.sqrt(dx ** 2 + dy ** 2)
        if pix_dist == 0:
            raise ValueError("两个标定点不能重合")
        if actual_mm <= 0:
            raise ValueError("实际长度必须大于 0")
        self._pixel_per_mm = pix_dist / actual_mm
        self._calib_points = [list(p1), list(p2)]

    # ---- 分析 (复用你原代码的整个算法流程) ----

    def analyze(self, clahe_clip=2.0, clahe_grid=8, blur_size=5,
                prominence_ratio=0.05, min_distance=10):
        """执行条纹分析 — 算法与你原代码完全一致"""
        if not self.has_image:
            raise RuntimeError("请先加载图片")
        if not self.is_calibrated:
            raise RuntimeError("请先完成标定")

        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                tileGridSize=(clahe_grid, clahe_grid))
        enhanced = clahe.apply(self._gray)

        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        self._blur_cache = cv2.GaussianBlur(enhanced, (k, k), 0)

        self._hist_cache = cv2.reduce(self._blur_cache, 0,
                                      cv2.REDUCE_AVG).reshape(-1)

        prominence = np.ptp(self._hist_cache) * prominence_ratio
        self._peaks_cache, _ = find_peaks(
            self._hist_cache, prominence=prominence, distance=min_distance
        )
        self._peaks_cache.sort()

        if len(self._peaks_cache) < 2:
            self._peaks_cache = None
            raise RuntimeError(f"检测到的条纹不足，请调整分析参数")

        spacings_pix = np.diff(self._peaks_cache)
        mean_pix = float(np.mean(spacings_pix))
        mean_mm = mean_pix / self._pixel_per_mm

        return {
            'num_fringes': len(self._peaks_cache),
            'mean_spacing_px': mean_pix,
            'mean_spacing_mm': mean_mm,
            'std_spacing_px': float(np.std(spacings_pix)),
            'std_spacing_mm': float(np.std(spacings_pix) / self._pixel_per_mm),
            'pixel_per_mm': self._pixel_per_mm,
            'peaks': self._peaks_cache.tolist(),
            'spacings_px': spacings_pix.tolist(),
            'spacings_mm': (spacings_pix / self._pixel_per_mm).tolist(),
            'calib_points': self._calib_points,
            'image_size': self.image_size,
        }

    # ---- 图像获取 ----

    def get_marked_image(self):
        """返回标记了条纹线的图像 (RGB)，同你原代码的 cv2.line 方式"""
        if not self.has_image:
            return None
        img = self._image_rgb.copy()
        if self._peaks_cache is not None:
            h = img.shape[0]
            for x in self._peaks_cache:
                cv2.line(img, (int(round(x)), 0),
                         (int(round(x)), h), (255, 0, 0), 2)
        return img

    def get_original_image(self):
        return self._image_rgb if self.has_image else None

    def get_signal_data(self):
        """返回 (x, y, peaks) 用于 matplotlib 绘图"""
        if self._hist_cache is None:
            return None, None, None
        x = np.arange(len(self._hist_cache))
        peaks = self._peaks_cache.copy() if self._peaks_cache is not None else None
        return x, self._hist_cache, peaks
