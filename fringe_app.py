"""
双缝干涉条纹间距测量工具 - PyQt5 前端界面

运行方式:
    pip install PyQt5 opencv-python numpy scipy matplotlib
    python fringe_app.py
"""

import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFormLayout, QDoubleSpinBox,
    QStatusBar, QAction, QFileDialog, QMessageBox, QSplitter,
    QSpinBox, QFrame, QToolBar, QSizePolicy,
    QGraphicsView, QGraphicsScene,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QPainter,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from fringe_core import FringeBridge


# ======================================================================
# 工具函数
# ======================================================================

def cv2_to_qimage(rgb_array):
    """将 OpenCV RGB numpy 数组转换为 QImage"""
    if rgb_array is None:
        return None
    h, w, ch = rgb_array.shape
    bytes_per_line = ch * w
    img = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return img.copy()  # 深拷贝以确保数据独立


# ======================================================================
# 图像查看器组件
# ======================================================================

class ImageViewer(QGraphicsView):
    """可交互的图像查看器，支持缩放和标定点绘制"""

    point_clicked = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._overlay_items = []
        self._mode = 'normal'
        self._zoom = 0
        self._min_zoom = -20
        self._max_zoom = 20

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setBackgroundBrush(QBrush(QColor(60, 60, 60)))

    # ---- 图像管理 ----

    def set_image(self, qimage):
        """设置显示的图像"""
        self._scene.clear()
        self._overlay_items.clear()
        self._pixmap_item = None

        if qimage is not None:
            pixmap = QPixmap.fromImage(qimage)
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fit_view()

        self._zoom = 0

    def fit_view(self):
        """适应窗口大小"""
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._zoom = 0

    def zoom_in(self):
        if self._zoom < self._max_zoom:
            self._zoom += 1
            self.scale(1.15, 1.15)

    def zoom_out(self):
        if self._zoom > self._min_zoom:
            self._zoom -= 1
            self.scale(0.87, 0.87)

    def has_image(self):
        return self._pixmap_item is not None

    # ---- 交互模式 ----

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        """设置交互模式: 'normal' 或 'calibrate'"""
        self._mode = mode
        if mode == 'calibrate':
            self.setCursor(Qt.CrossCursor)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    # ---- 覆盖图形 (标定点/线) ----

    def clear_overlays(self):
        """清除所有覆盖图形"""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def add_calibration_point(self, pos, color=QColor(255, 50, 50), radius=6):
        """在图像上添加标定点 (十字 + 外圈)"""
        if self._pixmap_item is None:
            return
        pen = QPen(color, 2)
        brush = QBrush(color)
        x, y = pos.x(), pos.y()

        self._overlay_items.append(
            self._scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, pen, brush)
        )
        cs = radius + 4
        self._overlay_items.append(
            self._scene.addLine(x - cs, y, x + cs, y, pen)
        )
        self._overlay_items.append(
            self._scene.addLine(x, y - cs, x, y + cs, pen)
        )

    def add_calibration_line(self, p1, p2, color=QColor(50, 255, 50)):
        """添加标定连线 (虚线)"""
        pen = QPen(color, 2, Qt.DashLine)
        item = self._scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
        self._overlay_items.append(item)

    # ---- 事件处理 ----

    def mousePressEvent(self, event):
        if self._mode == 'calibrate' and self._pixmap_item is not None:
            scene_pos = self.mapToScene(event.pos())
            img_rect = self._pixmap_item.boundingRect()
            if img_rect.contains(scene_pos):
                pixel_pos = QPointF(int(scene_pos.x()), int(scene_pos.y()))
                self.point_clicked.emit(pixel_pos)
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 0.87
        if zoom_in and self._zoom < self._max_zoom:
            self._zoom += 1
            self.scale(factor, factor)
        elif not zoom_in and self._zoom > self._min_zoom:
            self._zoom -= 1
            self.scale(factor, factor)
        else:
            event.ignore()


# ======================================================================
# Matplotlib 信号图组件
# ======================================================================

class SignalCanvas(FigureCanvas):
    """嵌入的 matplotlib 画布，用于显示一维投影信号"""

    def __init__(self, parent=None):
        # 设置 matplotlib 中文字体
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        self.figure = Figure(figsize=(5, 2), dpi=100)
        self.figure.set_tight_layout(True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.clear()

    def clear(self):
        self.axes.clear()
        self.axes.set_title("一维投影信号", fontsize=10)
        self.axes.set_xlabel("像素位置 (X)", fontsize=8)
        self.axes.set_ylabel("平均亮度", fontsize=8)
        self.axes.tick_params(labelsize=7)
        self.axes.grid(True, alpha=0.3)
        self.draw()

    def plot_signal(self, x, y, peaks=None):
        """绘制信号和检测到的峰值"""
        self.axes.clear()
        self.axes.plot(x, y, 'b-', linewidth=1.5, label="投影信号")
        if peaks is not None and len(peaks) > 0:
            self.axes.plot(peaks, y[peaks], 'rx', markersize=6,
                           markeredgewidth=1.5, label=f"峰值 (n={len(peaks)})")
        self.axes.set_title("一维投影信号", fontsize=10)
        self.axes.set_xlabel("像素位置 (X)", fontsize=8)
        self.axes.set_ylabel("平均亮度", fontsize=8)
        self.axes.tick_params(labelsize=7)
        self.axes.legend(fontsize=8)
        self.axes.grid(True, alpha=0.3)
        self.draw()


# ======================================================================
# 主窗口
# ======================================================================

class MainWindow(QMainWindow):
    """应用程序主窗口"""

    # 状态常量
    STATE_IDLE = 0
    STATE_LOADED = 1
    STATE_CALIBRATED = 2
    STATE_ANALYZED = 3

    def __init__(self):
        super().__init__()
        self.analyzer = FringeBridge()
        self._state = self.STATE_IDLE
        self._calib_points = []
        self._show_result = False
        self._last_results = None
        self._current_image_path = None

        self._init_ui()
        self._update_ui_state()

    # ==================== UI 构建 ====================

    def _init_ui(self):
        self.setWindowTitle("双缝干涉条纹间距测量工具")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        # 先创建中央部件（包括 image_viewer），再创建菜单/工具栏
        self._create_central_widget()
        self._create_menu_bar()
        self._create_toolbar()

        self.statusBar().showMessage("就绪 - 请加载图片")

        # 连接信号
        self.image_viewer.point_clicked.connect(self._on_image_clicked)

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # ---- 文件 ----
        file_menu = menubar.addMenu("文件(&F)")
        act = QAction("加载图片(&O)...", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._load_image)
        file_menu.addAction(act)

        file_menu.addSeparator()

        self.export_act = QAction("导出结果(&E)...", self)
        self.export_act.setShortcut("Ctrl+E")
        self.export_act.triggered.connect(self._export_results)
        file_menu.addAction(self.export_act)

        file_menu.addSeparator()

        act = QAction("退出(&X)", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # ---- 视图 ----
        view_menu = menubar.addMenu("视图(&V)")

        act = QAction("适应窗口", self)
        act.setShortcut("Ctrl+F")
        act.triggered.connect(self.image_viewer.fit_view)
        view_menu.addAction(act)

        act = QAction("放大", self)
        act.setShortcut("Ctrl++")
        act.triggered.connect(self.image_viewer.zoom_in)
        view_menu.addAction(act)

        act = QAction("缩小", self)
        act.setShortcut("Ctrl+-")
        act.triggered.connect(self.image_viewer.zoom_out)
        view_menu.addAction(act)

        view_menu.addSeparator()

        self.show_orig_act = QAction("显示原图", self)
        self.show_orig_act.setEnabled(False)
        self.show_orig_act.triggered.connect(lambda: self._toggle_view(False))
        view_menu.addAction(self.show_orig_act)

        self.show_result_act = QAction("显示结果", self)
        self.show_result_act.setEnabled(False)
        self.show_result_act.triggered.connect(lambda: self._toggle_view(True))
        view_menu.addAction(self.show_result_act)

        # ---- 帮助 ----
        help_menu = menubar.addMenu("帮助(&H)")

        act = QAction("关于(&A)", self)
        act.triggered.connect(self._show_about)
        help_menu.addAction(act)

    def _create_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        act = QAction("加载图片", self)
        act.triggered.connect(self._load_image)
        toolbar.addAction(act)

        toolbar.addSeparator()

        self.calib_tool_act = QAction("标定", self)
        self.calib_tool_act.setEnabled(False)
        self.calib_tool_act.triggered.connect(self._toggle_calibration)
        toolbar.addAction(self.calib_tool_act)

        self.analyze_tool_act = QAction("分析", self)
        self.analyze_tool_act.setEnabled(False)
        self.analyze_tool_act.triggered.connect(self._run_analysis)
        toolbar.addAction(self.analyze_tool_act)

        toolbar.addSeparator()

        self.export_tool_act = QAction("导出结果", self)
        self.export_tool_act.setEnabled(False)
        self.export_tool_act.triggered.connect(self._export_results)
        toolbar.addAction(self.export_tool_act)

    def _create_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ========== 上部分：图像 + 控制面板 ==========
        top_splitter = QSplitter(Qt.Horizontal)

        # --- 左侧：图像查看器 ---
        viewer_container = QWidget()
        vl = QVBoxLayout(viewer_container)
        vl.setContentsMargins(0, 0, 0, 0)

        self.image_viewer = ImageViewer()
        vl.addWidget(self.image_viewer)

        # 原图/结果 切换按钮
        self.view_toggle_btn = QPushButton("显示原图")
        self.view_toggle_btn.setCheckable(True)
        self.view_toggle_btn.setVisible(False)
        self.view_toggle_btn.clicked.connect(self._on_toggle_clicked)
        vl.addWidget(self.view_toggle_btn)

        top_splitter.addWidget(viewer_container)

        # --- 右侧：控制面板 ---
        right_panel = QWidget()
        right_panel.setMaximumWidth(330)
        right_panel.setMinimumWidth(260)
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        # 图像信息
        self._build_info_group(rl)
        # 标定控制
        self._build_calib_group(rl)
        # 分析参数
        self._build_param_group(rl)
        # 测量结果
        self._build_result_group(rl)

        rl.addStretch()
        top_splitter.addWidget(right_panel)
        top_splitter.setSizes([750, 280])

        main_layout.addWidget(top_splitter, 1)

        # ========== 底部：信号图 ==========
        bottom_widget = QWidget()
        bl = QVBoxLayout(bottom_widget)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)

        self.signal_canvas = SignalCanvas()
        nav = NavigationToolbar(self.signal_canvas, bottom_widget)
        bl.addWidget(nav)
        bl.addWidget(self.signal_canvas)
        main_layout.addWidget(bottom_widget, 0)

    # ---- 右侧面板子组件 ----

    def _build_info_group(self, parent_layout):
        grp = QGroupBox("图像信息")
        layout = QVBoxLayout(grp)
        self.info_label = QLabel("未加载图像")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        parent_layout.addWidget(grp)
        self._info_group = grp

    def _build_calib_group(self, parent_layout):
        grp = QGroupBox("尺寸标定")
        layout = QVBoxLayout(grp)

        self.calib_status = QLabel("点击「标定」后在图像上点击两个点")
        self.calib_status.setWordWrap(True)
        layout.addWidget(self.calib_status)

        form = QFormLayout()
        self.point1_label = QLabel("---")
        self.point2_label = QLabel("---")
        form.addRow("点 1:", self.point1_label)
        form.addRow("点 2:", self.point2_label)

        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0.001, 9999)
        self.dist_spin.setSuffix(" mm")
        self.dist_spin.setDecimals(3)
        self.dist_spin.setValue(10.0)
        self.dist_spin.setEnabled(False)
        form.addRow("实际距离:", self.dist_spin)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.calib_btn = QPushButton("标定")
        self.calib_btn.clicked.connect(self._toggle_calibration)
        self.apply_calib_btn = QPushButton("应用")
        self.apply_calib_btn.setEnabled(False)
        self.apply_calib_btn.clicked.connect(self._apply_calibration)
        self.clear_calib_btn = QPushButton("清除")
        self.clear_calib_btn.setEnabled(False)
        self.clear_calib_btn.clicked.connect(self._clear_calibration)
        btn_row.addWidget(self.calib_btn)
        btn_row.addWidget(self.apply_calib_btn)
        btn_row.addWidget(self.clear_calib_btn)
        layout.addLayout(btn_row)

        self.scale_label = QLabel("比例尺: 未设置")
        layout.addWidget(self.scale_label)
        parent_layout.addWidget(grp)
        self._calib_group = grp

    def _build_param_group(self, parent_layout):
        grp = QGroupBox("分析参数")
        form = QFormLayout(grp)

        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(0.5, 10)
        self.clahe_spin.setValue(2.0)
        self.clahe_spin.setSingleStep(0.5)
        form.addRow("对比度增强:", self.clahe_spin)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(3, 31)
        self.blur_spin.setValue(5)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setSuffix(" px")
        form.addRow("模糊核:", self.blur_spin)

        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.005, 0.5)
        self.prominence_spin.setValue(0.05)
        self.prominence_spin.setSingleStep(0.005)
        self.prominence_spin.setDecimals(3)
        form.addRow("峰值灵敏度:", self.prominence_spin)

        self.dist_param_spin = QSpinBox()
        self.dist_param_spin.setRange(3, 200)
        self.dist_param_spin.setValue(10)
        self.dist_param_spin.setSuffix(" px")
        form.addRow("最小峰间距:", self.dist_param_spin)

        parent_layout.addWidget(grp)

    def _build_result_group(self, parent_layout):
        self._result_group = QGroupBox("测量结果")
        layout = QVBoxLayout(self._result_group)

        self.result_text = QLabel("尚未分析")
        self.result_text.setWordWrap(True)
        self.result_text.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self.result_text)

        self.analyze_btn = QPushButton("执行分析")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.analyze_btn.setMinimumHeight(36)
        layout.addWidget(self.analyze_btn)

        parent_layout.addWidget(self._result_group)

    # ==================== 核心功能 ====================

    def _load_image(self):
        """加载图片"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择条纹图片", "",
            "图片文件 (*.jpg *.png *.bmp *.jpeg);;所有文件 (*)"
        )
        if not path:
            return

        try:
            self.analyzer.load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "图片加载失败", str(e))
            return

        self._current_image_path = path
        self._state = self.STATE_LOADED
        self._calib_points = []
        self._show_result = False
        self._last_results = None

        # 显示原图
        qimg = cv2_to_qimage(self.analyzer.get_original_image())
        self.image_viewer.set_image(qimg)
        self.image_viewer.set_mode('normal')

        # 更新信息
        w, h = self.analyzer.image_size
        name = os.path.basename(path)
        self.info_label.setText(f"文件: {name}\n尺寸: {w} x {h} 像素")

        # 重置 UI
        self._clear_calibration(keep_mode=True)
        self.result_text.setText("尚未分析")
        self.signal_canvas.clear()
        self.view_toggle_btn.setVisible(False)
        self.scale_label.setText("比例尺: 未设置")

        self._update_ui_state()
        self.statusBar().showMessage(f"已加载: {name} ({w}x{h})")

    def _toggle_calibration(self):
        """切换标定模式"""
        if self.image_viewer.mode == 'calibrate':
            # 退出标定
            self.image_viewer.set_mode('normal')
            self.calib_btn.setText("标定")
            self.calib_tool_act.setText("标定")
            self.calib_status.setText("标定已取消")
            self.statusBar().showMessage("标定已取消")
            self._update_ui_state()
            return

        # 如果已分析，回退到已加载状态
        if self._state >= self.STATE_ANALYZED:
            self._state = self.STATE_LOADED
            self._last_results = None
            self._show_result = False
            self._calib_points = []
            self.view_toggle_btn.setVisible(False)
            qimg = cv2_to_qimage(self.analyzer.get_original_image())
            self.image_viewer.set_image(qimg)
            self.signal_canvas.clear()
            self.result_text.setText("尚未分析")

        # 进入标定
        self._calib_points = []
        self.image_viewer.clear_overlays()
        self.point1_label.setText("---")
        self.point2_label.setText("---")
        self.dist_spin.setEnabled(False)
        self.apply_calib_btn.setEnabled(False)

        self.image_viewer.set_mode('calibrate')
        self.calib_btn.setText("取消")
        self.calib_tool_act.setText("取消")
        self.calib_status.setText("请在图像上点击第 1 个标定点")
        self.statusBar().showMessage("标定模式: 点击图像上的两个点作为参考")

        self._update_ui_state()

    def _on_image_clicked(self, pos):
        """图像点击事件 (标定模式)"""
        if len(self._calib_points) >= 2:
            return

        self._calib_points.append((pos.x(), pos.y()))
        n = len(self._calib_points)

        color = QColor(255, 50, 50) if n == 1 else QColor(255, 150, 50)
        self.image_viewer.add_calibration_point(pos, color)

        if n == 1:
            self.point1_label.setText(f"({int(pos.x())}, {int(pos.y())})")
            self.calib_status.setText("已选第 1 点, 请点击第 2 点")
        elif n == 2:
            self.point2_label.setText(f"({int(pos.x())}, {int(pos.y())})")
            p1 = QPointF(*self._calib_points[0])
            p2 = QPointF(*self._calib_points[1])
            self.image_viewer.add_calibration_line(p1, p2)

            dx = self._calib_points[1][0] - self._calib_points[0][0]
            dy = self._calib_points[1][1] - self._calib_points[0][1]
            pix_dist = np.sqrt(dx ** 2 + dy ** 2)

            self.calib_status.setText(
                f"两点已选定 (像素距离: {pix_dist:.1f} px)\n"
                f"输入实际长度后点击「应用」"
            )
            self.dist_spin.setEnabled(True)
            self.apply_calib_btn.setEnabled(True)
            self.clear_calib_btn.setEnabled(True)
            self.statusBar().showMessage(
                f"标定点已选, 像素距离: {pix_dist:.1f} px, 请输入实际长度"
            )

    def _apply_calibration(self):
        """应用标定"""
        if len(self._calib_points) != 2:
            return

        p1, p2 = self._calib_points
        actual_mm = self.dist_spin.value()

        try:
            self.analyzer.set_calibration(p1, p2, actual_mm)
        except ValueError as e:
            QMessageBox.warning(self, "标定错误", str(e))
            return

        self._state = self.STATE_CALIBRATED

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        pix_dist = np.sqrt(dx ** 2 + dy ** 2)

        self.scale_label.setText(
            f"比例尺: {self.analyzer.pixel_per_mm:.2f} px/mm\n"
            f"(像素 {pix_dist:.1f} px = {actual_mm:.3f} mm)"
        )

        # 退出标定模式
        self.image_viewer.set_mode('normal')
        self.calib_btn.setText("重标定")
        self.calib_tool_act.setText("标定")
        self.apply_calib_btn.setEnabled(False)
        self.dist_spin.setEnabled(False)
        self.calib_status.setText("标定完成")

        self.statusBar().showMessage(
            f"标定完成: {self.analyzer.pixel_per_mm:.2f} px/mm"
        )
        self._update_ui_state()

    def _clear_calibration(self, keep_mode=False):
        """清除标定"""
        self._calib_points = []
        self.image_viewer.clear_overlays()
        self.point1_label.setText("---")
        self.point2_label.setText("---")
        self.dist_spin.setEnabled(False)
        self.apply_calib_btn.setEnabled(False)
        self.clear_calib_btn.setEnabled(False)
        self.calib_status.setText("点击「标定」后在图像上点击两个点")

        if not keep_mode:
            self.image_viewer.set_mode('normal')
            self.calib_btn.setText("标定")
            self.calib_tool_act.setText("标定")

    def _run_analysis(self):
        """执行条纹分析"""
        if self._state < self.STATE_CALIBRATED:
            return

        params = dict(
            clahe_clip=self.clahe_spin.value(),
            blur_size=self.blur_spin.value(),
            prominence_ratio=self.prominence_spin.value(),
            min_distance=self.dist_param_spin.value(),
        )

        self.statusBar().showMessage("正在分析...")
        QApplication.processEvents()

        try:
            results = self.analyzer.analyze(**params)
        except RuntimeError as e:
            QMessageBox.warning(self, "分析失败", str(e))
            self.statusBar().showMessage("分析失败")
            return

        self._state = self.STATE_ANALYZED
        self._last_results = results
        self._show_result = True

        # 显示结果图像
        qimg = cv2_to_qimage(self.analyzer.get_marked_image())
        self.image_viewer.set_image(qimg)
        self.image_viewer.set_mode('normal')
        self.view_toggle_btn.setVisible(True)
        self.view_toggle_btn.setText("显示原图")
        self.view_toggle_btn.setChecked(True)

        # 显示结果
        r = results
        self.result_text.setText(
            f"\n检测到 {r['num_fringes']} 条条纹\n\n"
            f"平均间距:\n"
            f"  {r['mean_spacing_px']:.2f}  像素\n"
            f"  {r['mean_spacing_mm']:.4f}  mm\n\n"
            f"标准差:\n"
            f"  {r['std_spacing_px']:.2f}  像素\n"
            f"  {r['std_spacing_mm']:.4f}  mm\n\n"
            f"比例尺: {r['pixel_per_mm']:.2f} px/mm"
        )

        # 信号图
        x, y, peaks = self.analyzer.get_signal_data()
        if x is not None:
            self.signal_canvas.plot_signal(x, y, peaks)

        self.statusBar().showMessage(
            f"分析完成: {r['num_fringes']} 条条纹, "
            f"平均间距 {r['mean_spacing_mm']:.4f} mm"
        )
        self._update_ui_state()

    def _toggle_view(self, show_result=None):
        """切换原图/结果图显示"""
        if not self.analyzer.has_image:
            return

        if show_result is not None:
            self._show_result = show_result
        else:
            self._show_result = not self._show_result

        if self._show_result and self._last_results is not None:
            qimg = cv2_to_qimage(self.analyzer.get_marked_image())
            self.view_toggle_btn.setText("显示原图")
        else:
            qimg = cv2_to_qimage(self.analyzer.get_original_image())
            self.view_toggle_btn.setText("显示结果")

        self.image_viewer.set_image(qimg)

    def _on_toggle_clicked(self, checked):
        """切换按钮点击"""
        self._toggle_view(show_result=checked)

    def _export_results(self):
        """导出分析结果到文本文件"""
        if self._last_results is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "fringe_results.txt",
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if not path:
            return

        try:
            r = self._last_results
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("  双缝干涉条纹间距测量结果\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"图片: {self._current_image_path}\n")
                f.write(f"图像尺寸: {r['image_size'][0]} x {r['image_size'][1]} px\n")
                f.write(f"检测条纹数: {r['num_fringes']}\n\n")
                f.write(f"平均间距: {r['mean_spacing_px']:.2f} px = "
                        f"{r['mean_spacing_mm']:.4f} mm\n")
                f.write(f"标准差:   {r['std_spacing_px']:.2f} px = "
                        f"{r['std_spacing_mm']:.4f} mm\n")
                f.write(f"比例尺:   {r['pixel_per_mm']:.2f} px/mm\n\n")
                f.write("-" * 50 + "\n")
                f.write("条纹位置 (像素坐标 X):\n")
                for i, p in enumerate(r['peaks']):
                    f.write(f"  条纹 {i + 1:>3d}: x = {p:.1f}\n")
                f.write("\n间隔明细:\n")
                for i, sp_mm in enumerate(r['spacings_mm']):
                    f.write(f"  {i + 1:>3d}-{i + 2:>3d}: {sp_mm:.4f} mm  "
                            f"({r['spacings_px'][i]:.2f} px)\n")
                f.write("\n" + "=" * 50 + "\n")

            # 同时导出标记图像
            img_path = os.path.splitext(path)[0] + "_marked.png"
            marked_bgr = cv2.cvtColor(self.analyzer.get_marked_image(),
                                      cv2.COLOR_RGB2BGR)
            cv2.imencode('.png', marked_bgr)[1].tofile(img_path)

            self.statusBar().showMessage(
                f"结果已导出: {os.path.basename(path)}, "
                f"{os.path.basename(img_path)}"
            )
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "<h3>双缝干涉条纹间距测量工具</h3>"
            "<p>基于计算机视觉的干涉条纹自动分析</p>"
            "<hr>"
            "<p><b>流程:</b> 加载图片 → 标定比例尺 → 自动分析</p>"
            "<p><b>技术栈:</b> Python · OpenCV · SciPy · PyQt5</p>"
        )

    # ==================== 状态管理 ====================

    def _update_ui_state(self):
        """根据当前状态更新界面控件的启用状态"""
        s = self._state

        self.calib_tool_act.setEnabled(s >= self.STATE_LOADED)
        self._calib_group.setEnabled(s >= self.STATE_LOADED)

        self.analyze_tool_act.setEnabled(s >= self.STATE_CALIBRATED)
        self.analyze_btn.setEnabled(s >= self.STATE_CALIBRATED)

        self._result_group.setEnabled(s >= self.STATE_LOADED)

        has_result = s >= self.STATE_ANALYZED
        self.show_orig_act.setEnabled(has_result)
        self.show_result_act.setEnabled(has_result)
        self.export_act.setEnabled(has_result)
        self.export_tool_act.setEnabled(has_result)


# ======================================================================
# 程序入口
# ======================================================================

def main():
    # 确保高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("条纹间距测量")
    app.setOrganizationName("FringeTool")

    # Fusion 风格 (跨平台一致性)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    """
双缝干涉条纹间距测量工具 - PyQt5 前端界面

运行方式:
    pip install PyQt5 opencv-python numpy scipy matplotlib
    python fringe_app.py
"""

import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFormLayout, QDoubleSpinBox,
    QStatusBar, QAction, QFileDialog, QMessageBox, QSplitter,
    QSpinBox, QFrame, QToolBar, QSizePolicy,
    QGraphicsView, QGraphicsScene, QDialog,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QPainter,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from fringe_core import FringeBridge


def cv2_to_qimage(rgb_array):
    """将 OpenCV RGB numpy 数组转换为 QImage"""
    if rgb_array is None:
        return None
    h, w, ch = rgb_array.shape
    bytes_per_line = ch * w
    img = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return img.copy()


class ImageViewer(QGraphicsView):
    """可交互的图像查看器，支持缩放和标定点绘制"""

    point_clicked = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._overlay_items = []
        self._mode = 'normal'
        self._zoom = 0
        self._min_zoom = -20
        self._max_zoom = 20

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setBackgroundBrush(QBrush(QColor(60, 60, 60)))

    def set_image(self, qimage):
        self._scene.clear()
        self._overlay_items.clear()
        self._pixmap_item = None

        if qimage is not None:
            pixmap = QPixmap.fromImage(qimage)
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fit_view()

        self._zoom = 0

    def fit_view(self):
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._zoom = 0

    def zoom_in(self):
        if self._zoom < self._max_zoom:
            self._zoom += 1
            self.scale(1.15, 1.15)

    def zoom_out(self):
        if self._zoom > self._min_zoom:
            self._zoom -= 1
            self.scale(0.87, 0.87)

    def has_image(self):
        return self._pixmap_item is not None

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        self._mode = mode
        if mode == 'calibrate':
            self.setCursor(Qt.CrossCursor)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def add_calibration_point(self, pos, color=QColor(255, 50, 50), radius=6):
        if self._pixmap_item is None:
            return
        pen = QPen(color, 2)
        brush = QBrush(color)
        x, y = pos.x(), pos.y()

        self._overlay_items.append(
            self._scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, pen, brush)
        )
        cs = radius + 4
        self._overlay_items.append(self._scene.addLine(x - cs, y, x + cs, y, pen))
        self._overlay_items.append(self._scene.addLine(x, y - cs, x, y + cs, pen))

    def add_calibration_line(self, p1, p2, color=QColor(50, 255, 50)):
        pen = QPen(color, 2, Qt.DashLine)
        item = self._scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
        self._overlay_items.append(item)

    def mousePressEvent(self, event):
        if self._mode == 'calibrate' and self._pixmap_item is not None:
            scene_pos = self.mapToScene(event.pos())
            img_rect = self._pixmap_item.boundingRect()
            if img_rect.contains(scene_pos):
                pixel_pos = QPointF(int(scene_pos.x()), int(scene_pos.y()))
                self.point_clicked.emit(pixel_pos)
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 0.87
        if zoom_in and self._zoom < self._max_zoom:
            self._zoom += 1
            self.scale(factor, factor)
        elif not zoom_in and self._zoom > self._min_zoom:
            self._zoom -= 1
            self.scale(factor, factor)
        else:
            event.ignore()


class SignalCanvas(FigureCanvas):
    """嵌入的 matplotlib 画布，用于显示一维投影信号"""

    def __init__(self, parent=None):
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        self.figure = Figure(figsize=(5, 2), dpi=100)
        self.figure.set_tight_layout(True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.clear()

    def clear(self):
        self.axes.clear()
        self.axes.set_title("一维投影信号", fontsize=10)
        self.axes.set_xlabel("像素位置 (X)", fontsize=8)
        self.axes.set_ylabel("平均亮度", fontsize=8)
        self.axes.tick_params(labelsize=7)
        self.axes.grid(True, alpha=0.3)
        self.draw()

    def plot_signal(self, x, y, peaks=None):
        self.axes.clear()
        self.axes.plot(x, y, 'b-', linewidth=1.5, label="投影信号")
        if peaks is not None and len(peaks) > 0:
            self.axes.plot(peaks, y[peaks], 'rx', markersize=6,
                           markeredgewidth=1.5, label=f"峰值 (n={len(peaks)})")
        self.axes.set_title("一维投影信号", fontsize=10)
        self.axes.set_xlabel("像素位置 (X)", fontsize=8)
        self.axes.set_ylabel("平均亮度", fontsize=8)
        self.axes.tick_params(labelsize=7)
        self.axes.legend(fontsize=8)
        self.axes.grid(True, alpha=0.3)
        self.draw()


class SignalGraphDialog(QDialog):
    """独立显示一维投影信号图的窗口"""

    def __init__(self, x, y, peaks, parent=None):
        super().__init__(parent)
        self.setWindowTitle("一维投影信号")
        self.resize(900, 450)
        self.setMinimumSize(600, 300)

        layout = QVBoxLayout(self)
        canvas = SignalCanvas()
        canvas.plot_signal(x, y, peaks)

        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        self.setAttribute(Qt.WA_DeleteOnClose)


class MainWindow(QMainWindow):
    """应用程序主窗口"""

    STATE_IDLE = 0
    STATE_LOADED = 1
    STATE_CALIBRATED = 2
    STATE_ANALYZED = 3

    def __init__(self):
        super().__init__()
        self.analyzer = FringeBridge()
        self._state = self.STATE_IDLE
        self._calib_points = []
        self._show_result = False
        self._last_results = None
        self._current_image_path = None
        self._signal_data = (None, None, None)

        self._init_ui()
        self._update_ui_state()

    def _init_ui(self):
        self.setWindowTitle("双缝干涉条纹间距测量工具")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        self._create_central_widget()
        self._create_menu_bar()
        self._create_toolbar()

        self.statusBar().showMessage("就绪 - 请加载图片")

        self.image_viewer.point_clicked.connect(self._on_image_clicked)

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")
        act = QAction("加载图片(&O)...", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._load_image)
        file_menu.addAction(act)

        file_menu.addSeparator()

        self.export_act = QAction("导出结果(&E)...", self)
        self.export_act.setShortcut("Ctrl+E")
        self.export_act.triggered.connect(self._export_results)
        file_menu.addAction(self.export_act)

        file_menu.addSeparator()

        act = QAction("退出(&X)", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        view_menu = menubar.addMenu("视图(&V)")

        act = QAction("适应窗口", self)
        act.setShortcut("Ctrl+F")
        act.triggered.connect(self.image_viewer.fit_view)
        view_menu.addAction(act)

        act = QAction("放大", self)
        act.setShortcut("Ctrl++")
        act.triggered.connect(self.image_viewer.zoom_in)
        view_menu.addAction(act)

        act = QAction("缩小", self)
        act.setShortcut("Ctrl+-")
        act.triggered.connect(self.image_viewer.zoom_out)
        view_menu.addAction(act)

        view_menu.addSeparator()

        self.show_orig_act = QAction("显示原图", self)
        self.show_orig_act.setEnabled(False)
        self.show_orig_act.triggered.connect(lambda: self._toggle_view(False))
        view_menu.addAction(self.show_orig_act)

        self.show_result_act = QAction("显示结果", self)
        self.show_result_act.setEnabled(False)
        self.show_result_act.triggered.connect(lambda: self._toggle_view(True))
        view_menu.addAction(self.show_result_act)

        help_menu = menubar.addMenu("帮助(&H)")
        act = QAction("关于(&A)", self)
        act.triggered.connect(self._show_about)
        help_menu.addAction(act)

    def _create_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        act = QAction("加载图片", self)
        act.triggered.connect(self._load_image)
        toolbar.addAction(act)

        toolbar.addSeparator()

        self.calib_tool_act = QAction("标定", self)
        self.calib_tool_act.setEnabled(False)
        self.calib_tool_act.triggered.connect(self._toggle_calibration)
        toolbar.addAction(self.calib_tool_act)

        self.analyze_tool_act = QAction("分析", self)
        self.analyze_tool_act.setEnabled(False)
        self.analyze_tool_act.triggered.connect(self._run_analysis)
        toolbar.addAction(self.analyze_tool_act)

        toolbar.addSeparator()

        self.export_tool_act = QAction("导出结果", self)
        self.export_tool_act.setEnabled(False)
        self.export_tool_act.triggered.connect(self._export_results)
        toolbar.addAction(self.export_tool_act)

    def _create_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        top_splitter = QSplitter(Qt.Horizontal)

        viewer_container = QWidget()
        vl = QVBoxLayout(viewer_container)
        vl.setContentsMargins(0, 0, 0, 0)

        self.image_viewer = ImageViewer()
        vl.addWidget(self.image_viewer)

        self.view_toggle_btn = QPushButton("显示原图")
        self.view_toggle_btn.setCheckable(True)
        self.view_toggle_btn.setVisible(False)
        self.view_toggle_btn.clicked.connect(self._on_toggle_clicked)
        vl.addWidget(self.view_toggle_btn)

        top_splitter.addWidget(viewer_container)

        right_panel = QWidget()
        right_panel.setMaximumWidth(330)
        right_panel.setMinimumWidth(260)
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self._build_info_group(rl)
        self._build_calib_group(rl)
        self._build_param_group(rl)
        self._build_result_group(rl)

        rl.addStretch()
        top_splitter.addWidget(right_panel)
        top_splitter.setSizes([750, 280])

        main_layout.addWidget(top_splitter, 1)

    def _build_info_group(self, parent_layout):
        grp = QGroupBox("图像信息")
        layout = QVBoxLayout(grp)
        self.info_label = QLabel("未加载图像")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        parent_layout.addWidget(grp)
        self._info_group = grp

    def _build_calib_group(self, parent_layout):
        grp = QGroupBox("尺寸标定")
        layout = QVBoxLayout(grp)

        self.calib_status = QLabel("点击「标定」后在图像上点击两个点")
        self.calib_status.setWordWrap(True)
        layout.addWidget(self.calib_status)

        form = QFormLayout()
        self.point1_label = QLabel("---")
        self.point2_label = QLabel("---")
        form.addRow("点 1:", self.point1_label)
        form.addRow("点 2:", self.point2_label)

        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0.001, 9999)
        self.dist_spin.setSuffix(" mm")
        self.dist_spin.setDecimals(3)
        self.dist_spin.setValue(10.0)
        self.dist_spin.setEnabled(False)
        form.addRow("实际距离:", self.dist_spin)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.calib_btn = QPushButton("标定")
        self.calib_btn.clicked.connect(self._toggle_calibration)
        self.apply_calib_btn = QPushButton("应用")
        self.apply_calib_btn.setEnabled(False)
        self.apply_calib_btn.clicked.connect(self._apply_calibration)
        self.clear_calib_btn = QPushButton("清除")
        self.clear_calib_btn.setEnabled(False)
        self.clear_calib_btn.clicked.connect(self._clear_calibration)
        btn_row.addWidget(self.calib_btn)
        btn_row.addWidget(self.apply_calib_btn)
        btn_row.addWidget(self.clear_calib_btn)
        layout.addLayout(btn_row)

        self.scale_label = QLabel("比例尺: 未设置")
        layout.addWidget(self.scale_label)
        parent_layout.addWidget(grp)
        self._calib_group = grp

    def _build_param_group(self, parent_layout):
        grp = QGroupBox("分析参数")
        form = QFormLayout(grp)

        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(0.5, 10)
        self.clahe_spin.setValue(2.0)
        self.clahe_spin.setSingleStep(0.5)
        form.addRow("对比度增强:", self.clahe_spin)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(3, 31)
        self.blur_spin.setValue(5)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setSuffix(" px")
        form.addRow("模糊核:", self.blur_spin)

        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.005, 0.5)
        self.prominence_spin.setValue(0.05)
        self.prominence_spin.setSingleStep(0.005)
        self.prominence_spin.setDecimals(3)
        form.addRow("峰值灵敏度:", self.prominence_spin)

        self.dist_param_spin = QSpinBox()
        self.dist_param_spin.setRange(3, 200)
        self.dist_param_spin.setValue(10)
        self.dist_param_spin.setSuffix(" px")
        form.addRow("最小峰间距:", self.dist_param_spin)

        parent_layout.addWidget(grp)

    def _build_result_group(self, parent_layout):
        self._result_group = QGroupBox("测量结果")
        layout = QVBoxLayout(self._result_group)

        self.result_text = QLabel("尚未分析")
        self.result_text.setWordWrap(True)
        self.result_text.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self.result_text)

        self.analyze_btn = QPushButton("执行分析")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.analyze_btn.setMinimumHeight(36)
        layout.addWidget(self.analyze_btn)

        self.signal_view_btn = QPushButton("查看信号图")
        self.signal_view_btn.setEnabled(False)
        self.signal_view_btn.clicked.connect(self._view_signal_graph)
        layout.addWidget(self.signal_view_btn)

        parent_layout.addWidget(self._result_group)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择条纹图片", "",
            "图片文件 (*.jpg *.png *.bmp *.jpeg);;所有文件 (*)"
        )
        if not path:
            return

        try:
            self.analyzer.load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "图片加载失败", str(e))
            return

        self._current_image_path = path
        self._state = self.STATE_LOADED
        self._calib_points = []
        self._show_result = False
        self._last_results = None
        self._signal_data = (None, None, None)

        qimg = cv2_to_qimage(self.analyzer.get_original_image())
        self.image_viewer.set_image(qimg)
        self.image_viewer.set_mode('normal')

        w, h = self.analyzer.image_size
        name = os.path.basename(path)
        self.info_label.setText(f"文件: {name}\n尺寸: {w} x {h} 像素")

        self._clear_calibration(keep_mode=True)
        self.result_text.setText("尚未分析")
        self.view_toggle_btn.setVisible(False)
        self.scale_label.setText("比例尺: 未设置")

        self._update_ui_state()
        self.statusBar().showMessage(f"已加载: {name} ({w}x{h})")

    def _toggle_calibration(self):
        if self.image_viewer.mode == 'calibrate':
            self.image_viewer.set_mode('normal')
            self.calib_btn.setText("标定")
            self.calib_tool_act.setText("标定")
            self.calib_status.setText("标定已取消")
            self.statusBar().showMessage("标定已取消")
            self._update_ui_state()
            return

        if self._state >= self.STATE_ANALYZED:
            self._state = self.STATE_LOADED
            self._last_results = None
            self._show_result = False
            self._calib_points = []
            self._signal_data = (None, None, None)
            self.view_toggle_btn.setVisible(False)
            qimg = cv2_to_qimage(self.analyzer.get_original_image())
            self.image_viewer.set_image(qimg)
            self.result_text.setText("尚未分析")

        self._calib_points = []
        self.image_viewer.clear_overlays()
        self.point1_label.setText("---")
        self.point2_label.setText("---")
        self.dist_spin.setEnabled(False)
        self.apply_calib_btn.setEnabled(False)

        self.image_viewer.set_mode('calibrate')
        self.calib_btn.setText("取消")
        self.calib_tool_act.setText("取消")
        self.calib_status.setText("请在图像上点击第 1 个标定点")
        self.statusBar().showMessage("标定模式: 点击图像上的两个点作为参考")

        self._update_ui_state()

    def _on_image_clicked(self, pos):
        if len(self._calib_points) >= 2:
            return

        self._calib_points.append((pos.x(), pos.y()))
        n = len(self._calib_points)

        color = QColor(255, 50, 50) if n == 1 else QColor(255, 150, 50)
        self.image_viewer.add_calibration_point(pos, color)

        if n == 1:
            self.point1_label.setText(f"({int(pos.x())}, {int(pos.y())})")
            self.calib_status.setText("已选第 1 点, 请点击第 2 点")
        elif n == 2:
            self.point2_label.setText(f"({int(pos.x())}, {int(pos.y())})")
            p1 = QPointF(*self._calib_points[0])
            p2 = QPointF(*self._calib_points[1])
            self.image_viewer.add_calibration_line(p1, p2)

            dx = self._calib_points[1][0] - self._calib_points[0][0]
            dy = self._calib_points[1][1] - self._calib_points[0][1]
            pix_dist = np.sqrt(dx ** 2 + dy ** 2)

            self.calib_status.setText(
                f"两点已选定 (像素距离: {pix_dist:.1f} px)\n"
                f"输入实际长度后点击「应用」"
            )
            self.dist_spin.setEnabled(True)
            self.apply_calib_btn.setEnabled(True)
            self.clear_calib_btn.setEnabled(True)
            self.statusBar().showMessage(
                f"标定点已选, 像素距离: {pix_dist:.1f} px, 请输入实际长度"
            )

    def _apply_calibration(self):
        if len(self._calib_points) != 2:
            return

        p1, p2 = self._calib_points
        actual_mm = self.dist_spin.value()

        try:
            self.analyzer.set_calibration(p1, p2, actual_mm)
        except ValueError as e:
            QMessageBox.warning(self, "标定错误", str(e))
            return

        self._state = self.STATE_CALIBRATED

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        pix_dist = np.sqrt(dx ** 2 + dy ** 2)

        self.scale_label.setText(
            f"比例尺: {self.analyzer.pixel_per_mm:.2f} px/mm\n"
            f"(像素 {pix_dist:.1f} px = {actual_mm:.3f} mm)"
        )

        self.image_viewer.set_mode('normal')
        self.calib_btn.setText("重标定")
        self.calib_tool_act.setText("标定")
        self.apply_calib_btn.setEnabled(False)
        self.dist_spin.setEnabled(False)
        self.calib_status.setText("标定完成")

        self.statusBar().showMessage(
            f"标定完成: {self.analyzer.pixel_per_mm:.2f} px/mm"
        )
        self._update_ui_state()

    def _clear_calibration(self, keep_mode=False):
        self._calib_points = []
        self.image_viewer.clear_overlays()
        self.point1_label.setText("---")
        self.point2_label.setText("---")
        self.dist_spin.setEnabled(False)
        self.apply_calib_btn.setEnabled(False)
        self.clear_calib_btn.setEnabled(False)
        self.calib_status.setText("点击「标定」后在图像上点击两个点")

        if not keep_mode:
            self.image_viewer.set_mode('normal')
            self.calib_btn.setText("标定")
            self.calib_tool_act.setText("标定")

    def _run_analysis(self):
        if self._state < self.STATE_CALIBRATED:
            return

        params = dict(
            clahe_clip=self.clahe_spin.value(),
            blur_size=self.blur_spin.value(),
            prominence_ratio=self.prominence_spin.value(),
            min_distance=self.dist_param_spin.value(),
        )

        self.statusBar().showMessage("正在分析...")
        QApplication.processEvents()

        try:
            results = self.analyzer.analyze(**params)
        except RuntimeError as e:
            QMessageBox.warning(self, "分析失败", str(e))
            self.statusBar().showMessage("分析失败")
            return

        self._state = self.STATE_ANALYZED
        self._last_results = results
        self._show_result = True

        qimg = cv2_to_qimage(self.analyzer.get_marked_image())
        self.image_viewer.set_image(qimg)
        self.image_viewer.set_mode('normal')
        self.view_toggle_btn.setVisible(True)
        self.view_toggle_btn.setText("显示原图")
        self.view_toggle_btn.setChecked(True)

        r = results
        self.result_text.setText(
            f"\n检测到 {r['num_fringes']} 条条纹\n\n"
            f"平均间距:\n"
            f"  {r['mean_spacing_px']:.2f}  像素\n"
            f"  {r['mean_spacing_mm']:.4f}  mm\n\n"
            f"标准差:\n"
            f"  {r['std_spacing_px']:.2f}  像素\n"
            f"  {r['std_spacing_mm']:.4f}  mm\n\n"
            f"比例尺: {r['pixel_per_mm']:.2f} px/mm"
        )

        x, y, peaks = self.analyzer.get_signal_data()
        self._signal_data = (x, y, peaks)
        self.signal_view_btn.setEnabled(x is not None)

        self.statusBar().showMessage(
            f"分析完成: {r['num_fringes']} 条条纹, "
            f"平均间距 {r['mean_spacing_mm']:.4f} mm"
        )
        self._update_ui_state()

    def _toggle_view(self, show_result=None):
        if not self.analyzer.has_image:
            return

        if show_result is not None:
            self._show_result = show_result
        else:
            self._show_result = not self._show_result

        if self._show_result and self._last_results is not None:
            qimg = cv2_to_qimage(self.analyzer.get_marked_image())
            self.view_toggle_btn.setText("显示原图")
        else:
            qimg = cv2_to_qimage(self.analyzer.get_original_image())
            self.view_toggle_btn.setText("显示结果")

        self.image_viewer.set_image(qimg)

    def _on_toggle_clicked(self, checked):
        self._toggle_view(show_result=checked)

    def _view_signal_graph(self):
        x, y, peaks = self._signal_data
        if x is None:
            return
        self._signal_dialog = SignalGraphDialog(x, y, peaks, self)
        self._signal_dialog.show()

    def _export_results(self):
        if self._last_results is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "fringe_results.txt",
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if not path:
            return

        try:
            r = self._last_results
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("  双缝干涉条纹间距测量结果\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"图片: {self._current_image_path}\n")
                f.write(f"图像尺寸: {r['image_size'][0]} x {r['image_size'][1]} px\n")
                f.write(f"检测条纹数: {r['num_fringes']}\n\n")
                f.write(f"平均间距: {r['mean_spacing_px']:.2f} px = "
                        f"{r['mean_spacing_mm']:.4f} mm\n")
                f.write(f"标准差:   {r['std_spacing_px']:.2f} px = "
                        f"{r['std_spacing_mm']:.4f} mm\n")
                f.write(f"比例尺:   {r['pixel_per_mm']:.2f} px/mm\n\n")
                f.write("-" * 50 + "\n")
                f.write("条纹位置 (像素坐标 X):\n")
                for i, p in enumerate(r['peaks']):
                    f.write(f"  条纹 {i+1:>3d}: x = {p:.1f}\n")
                f.write("\n间隔明细:\n")
                for i, sp_mm in enumerate(r['spacings_mm']):
                    f.write(f"  {i+1:>3d}-{i+2:>3d}: {sp_mm:.4f} mm  "
                            f"({r['spacings_px'][i]:.2f} px)\n")
                f.write("\n" + "=" * 50 + "\n")

            img_path = os.path.splitext(path)[0] + "_marked.png"
            marked_bgr = cv2.cvtColor(self.analyzer.get_marked_image(),
                                      cv2.COLOR_RGB2BGR)
            cv2.imencode('.png', marked_bgr)[1].tofile(img_path)

            self.statusBar().showMessage(
                f"结果已导出: {os.path.basename(path)}, "
                f"{os.path.basename(img_path)}"
            )
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "<h3>双缝干涉条纹间距测量工具</h3>"
            "<p>基于计算机视觉的干涉条纹自动分析</p>"
            "<hr>"
            "<p><b>流程:</b> 加载图片 → 标定比例尺 → 自动分析</p>"
            "<p><b>技术栈:</b> Python · OpenCV · SciPy · PyQt5</p>"
        )

    def _update_ui_state(self):
        s = self._state

        self.calib_tool_act.setEnabled(s >= self.STATE_LOADED)
        self._calib_group.setEnabled(s >= self.STATE_LOADED)

        self.analyze_tool_act.setEnabled(s >= self.STATE_CALIBRATED)
        self.analyze_btn.setEnabled(s >= self.STATE_CALIBRATED)

        self._result_group.setEnabled(s >= self.STATE_LOADED)

        has_result = s >= self.STATE_ANALYZED
        self.show_orig_act.setEnabled(has_result)
        self.show_result_act.setEnabled(has_result)
        self.export_act.setEnabled(has_result)
        self.export_tool_act.setEnabled(has_result)

        if s < self.STATE_ANALYZED:
            self.signal_view_btn.setEnabled(False)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("条纹间距测量")
    app.setOrganizationName("FringeTool")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
