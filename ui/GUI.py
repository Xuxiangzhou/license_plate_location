import sys
import cv2
from PIL import Image
from collections import Counter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QFrame, QScrollArea,
    QGroupBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt

from Locate.config import Config
from Locate.locate import predict_license_plate  # 引入封装的预测函数
from Recognition.recognition import recognize_license_plate

""""
此代码为系统GUI页面，基于pyqt5开发，实现了图片识别和车牌识别，接入了rstp视频流。
"""
class LicensePlateRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("智慧交通——车牌识别系统")
        self.setGeometry(400, 200, 1200, 740)

        # Variables
        self.model_path = Config.model_path  # 模型路径
        self.timer = QTimer()
        self.image_path = None
        self.video_path = None
        self.video_capture = None
        self.is_recognizing = False  # 是否正在识别
        self.is_playing = False #是否正在播放
        self.current_frame_position = 0  # 当前视频帧位置
        self.current_frame = None
        self.is_camera_open = False
        #背景图片
        background_image_path =Config.background_path # 背景图片路径
        self.setStyleSheet(f"QMainWindow {{border-image: url({background_image_path}) 0 0 0 0 stretch stretch;}}")

        # 创建主容器面板
        main_panel = QWidget()
        main_panel.setStyleSheet(f"QMainWindow {{border-image: url({background_image_path}) 0 0 0 0 stretch stretch;}}")

        # 主水平布局
        main_layout = QHBoxLayout(main_panel)
        main_layout.setContentsMargins(20, 20, 20, 20)  # 内边距优化
        main_layout.setSpacing(30)  # 左右面板间距

        # 左侧图像面板
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("""
               background: rgba(230,230,230,0.8);
               border: 2px solid #A0A0A0;
               border-radius: 8px;
           """)
        # left_panel.setFixedSize(440,440)

        # 右侧控制面板
        right_panel = QFrame()
        right_panel.setStyleSheet("""
               background: rgba(245,245,245,0.9);
               border: 2px solid #C0C0C0;
               border-radius: 8px;
               padding: 15px;
           """)
        # 左侧垂直布局
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 图像显示区域（带滚动条）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # 自适应尺寸[2](@ref)
        self.scroll_area.setStyleSheet("background: transparent;")

        # 图像标签容器
        self.image_container = QLabel()
        self.image_container.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_container)

        left_layout.addWidget(self.scroll_area)
        # 右侧垂直布局
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        # 信息展示模块
        info_group = QGroupBox("识别信息")
        info_group.setStyleSheet("""
            QGroupBox { 
                font: bold 14px; 
                border: 1px solid gray; 
                margin-top: 10px;
            }
        """)
        info_layout = QVBoxLayout(info_group)
        #车牌提取
        self.right_label = QLabel("车牌提取:")
        self.right_label.setFont(QFont("Arial", 12))
        self.right_label.setStyleSheet("background-color: rgba(208, 208, 208, 0);")
        # self.right_label.setFixedSize(240, 50)
        #车牌区域显示
        self.cropped_plate_label = QLabel()
        self.cropped_plate_label.setStyleSheet("background-color: rgba(240, 240, 240, 0.8); border: 1px solid black;")
        self.cropped_plate_label.setFixedSize(312, 120)
        # 车牌号码
        self.right_result = QLabel("车牌号码:")
        self.right_result.setFont(QFont("Arial", 12))
        self.right_result.setStyleSheet("background-color: rgba(208, 208, 208, 0);")
        # self.right_label.setFixedSize(240, 50)
        # 车牌号码显示
        self.cropped_plate_result = QLabel()
        self.cropped_plate_result.setStyleSheet("background-color: rgba(240, 240, 240, 0.8); border: 1px solid black;")
        self.cropped_plate_result.setFixedSize(312,120)
        #添加组件
        info_layout.addWidget(self.right_label)
        info_layout.addWidget(self.cropped_plate_label)
        info_layout.addWidget(self.right_result)
        info_layout.addWidget(self.cropped_plate_result)
        #按钮布局
        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)
        # 按钮
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.open_image)

        self.select_video_btn = QPushButton("选择视频")
        self.select_video_btn.clicked.connect(self.open_video)
        #调用本机相机
        self.open_camera_btn = QPushButton("打开相机")
        self.open_camera_btn.clicked.connect(self.toggle_camera)
        #添加RTMP视频流
        self.get_rtmp_btn = QPushButton("添加视频")
        self.get_rtmp_btn.clicked.connect(self.open_stream)

        button_layout.addWidget(self.select_image_btn)
        button_layout.addWidget(self.select_video_btn)
        button_layout.addWidget(self.open_camera_btn)
        button_layout.addWidget(self.get_rtmp_btn)
        button_layout.addStretch(1)  # 弹性空间
        # 按钮布局
        button_group_down = QWidget()
        button_layout_down = QHBoxLayout(button_group_down)
        self.start_stop_recognition_btn = QPushButton("开始识别")
        self.start_stop_recognition_btn.clicked.connect(self.toggle_recognition)
        self.start_stop_play_btn = QPushButton("开始播放")
        self.start_stop_play_btn.clicked.connect(self.toggle_player)
        button_layout_down.addWidget(self.start_stop_recognition_btn)
        button_layout_down.addWidget(self.start_stop_play_btn)
        # 整合右侧布局
        right_layout.addWidget(info_group)
        right_layout.addWidget(button_group)
        right_layout.addWidget(button_group_down)
        right_layout.addStretch(1)  # 底部弹性填充
        main_layout.addWidget(left_panel, stretch=7)  # 左侧占70%宽度
        main_layout.addWidget(right_panel, stretch=3)  # 右侧占30%

        # 设置中心控件
        self.setCentralWidget(main_panel)
    def open_image(self):
        """
        选择图片
        """
        self.image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if self.image_path:
            self.video_path = None  # 清除视频选择
            image = cv2.imread(self.image_path)
            self.display_frame(image)


    def open_video(self):
        """
        选择视频
        """
        self.local= True  #视频是否为本地
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv)")
        if self.video_path:
            self.image_path = None  # 清除图片选择
            self.current_frame_position = 0  # 重置帧位置
            self.video_capture = cv2.VideoCapture(self.video_path)
            ret, frame = self.video_capture.read()
            self.display_frame(frame)

    def toggle_camera(self):
        """
        打开或关闭相机
        """
        if not self.is_camera_open:
            # 打开相机
            self.video_path = 0
            self.image_path = None
            self.local = False
            self.is_playing = True
            self.is_camera_open = True
            self.open_camera_btn.setText("关闭相机")  # 更新按钮文本
            self.start_stop_play_btn.setText("暂停播放")
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.timer.start(30)
            self.timer.timeout.connect(self.update_frame)
        else:
            # 关闭相机
            self.is_playing = False
            self.is_camera_open = False
            self.open_camera_btn.setText("打开相机")  # 更新按钮文本
            self.start_stop_play_btn.setText("开始播放")
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()



    def open_stream(self):
        """
        打开远程视频流
        """
        self.local = False
        self.video_path, ok = QInputDialog.getText(self, "添加视频", "请输入RTMP视频流地址：")
        if ok and self.video_path:
            self.image_path = None
            self.video_capture = cv2.VideoCapture(self.video_path)
            # 显示第一帧
            ret, frame = self.video_capture.read()
            self.display_frame(frame)



    def toggle_recognition(self):
        """
        开启/停止识别
        """
        if self.is_recognizing == False:
            self.is_recognizing = True
            if self.is_playing :
                self.start_stop_recognition_btn.setText("停止识别")
            else:
                image= cv2.imread(self.image_path)
                self.process_image(image)
                self.is_recognizing = False
        else :
            self.is_recognizing = False
            if self.is_playing :
                self.start_stop_recognition_btn.setText("开始识别")



    def toggle_player(self):
        """
        开始/暂停播放
        """
        if self.video_capture is None:  # 确保 video_capture 已初始化
            self.video_capture = cv2.VideoCapture(self.video_path)
        #暂停播放
        if self.is_playing:
            self.is_playing = False  #播放状态改为False
            self.timer.stop()  # 停止定时器，从而暂停播放
            self.start_stop_play_btn.setText("开始播放")
            if self.local:
                self.current_frame_position = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))  # 保存当前帧位置

        else:
            self.is_playing = True
            self.start_stop_play_btn.setText("暂停播放")
            self.timer.start(60)  # 启动定时器，从而开始播放
            if self.local:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_position)  # 设置帧位置
                # 确保定时器信号只绑定一次
            try:
                self.timer.timeout.disconnect(self.update_frame)  # 先解绑信号
            except TypeError:
                pass  # 如果没有连接，忽略异常
            self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        # 更新左侧图片显示
        ret, frame = self.video_capture.read()
        if ret :
            self.display_frame(frame)
            if self.is_recognizing:
                self.process_frame(frame)
        else:
            print("读取视频失败")

    def process_frame(self, frame):
        """
        处理视频帧
        """
        original_image_with_box, cropped_plate = predict_license_plate(self.model_path, frame)
        if original_image_with_box is not None:
            self.display_frame(original_image_with_box)
        if cropped_plate is not None:
            self.display_cropped_plate(cropped_plate)

    def process_image(self, image):
        """
        处理图片
        """
        original_image_with_box, cropped_plate = predict_license_plate(self.model_path, image)
        original_image_with_box = cv2.resize(original_image_with_box, (640, 640))
        if original_image_with_box is not None:
            self.display_frame(original_image_with_box)
        if cropped_plate is not None:
            self.display_cropped_plate(cropped_plate)

    def display_frame(self, frame):
        """
        显示左侧视频帧/图片
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        q_image = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_container.setPixmap(pixmap)

    def display_cropped_plate(self, cropped_plate):
        """
        显示右侧车牌区域，加入多帧融合逻辑
        """
        # 调整车牌图片的尺寸和颜色空间
        cropped_plate = cv2.resize(cropped_plate, (284, 120))
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
        height, width, channel = cropped_plate.shape
        step = channel * width
        q_image = QImage(cropped_plate.data, width, height, step, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.cropped_plate_label.setPixmap(pixmap)

        # 车牌号码识别
        cropped_plate_pil = Image.fromarray(cropped_plate)
        plate_number = recognize_license_plate(cropped_plate_pil, Config.reconize_model)

        # 设置字体样式
        font = QFont()
        font.setFamily("Arial")  # 字体名称
        font.setPointSize(28)  # 字体大小
        font.setBold(True)  # 加粗
        if self.is_playing:
            # 初始化多帧融合缓冲区（如果尚未初始化）
            if not hasattr(self, 'frame_buffer'):
                self.frame_buffer = []
                self.max_frame_buffer_size = 30 # 车牌号码更新间隔帧数

            # 添加当前帧的识别结果到缓冲区
            if plate_number:
                self.frame_buffer.append(plate_number)
                if len(self.frame_buffer) > self.max_frame_buffer_size:
                    self.frame_buffer.pop(0)

                # 多帧融合：统计缓冲区中最频繁的车牌号码
                final_plate_number = self.fuse_plate_numbers(self.frame_buffer)
                # 显示最终车牌号码
                self.cropped_plate_result.setFont(font)
                self.cropped_plate_result.setText(final_plate_number if final_plate_number else "")
                print(final_plate_number)
            else:
                # 清空车牌号码显示
                self.cropped_plate_result.clear()
                print("未检测到车牌号码")
        else:
            # 显示最终车牌号码
            self.cropped_plate_result.setFont(font)
            self.cropped_plate_result.setText(plate_number if plate_number else "")

    def fuse_plate_numbers(self, frame_buffer):
        """
        多帧融合，返回最可能的车牌号码
        """
        if not frame_buffer:
            return ""
        # 使用Counter统计识别结果的出现频率
        plate_counts = Counter(frame_buffer)
        # 返回出现频率最高的车牌号码
        return plate_counts.most_common(1)[0][0]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())