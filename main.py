# import sys
# from PyQt5 import QtWidgets, QtCore
# from PyQt5.QtWidgets import QFileDialog
# from PyQt5.QtGui import QPixmap, QImage
# import cv2
# from ultralytics import YOLO
# from yolo import Ui_MainWindow
#
#
# class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
#     def __init__(self):
#         try:
#             super().__init__()
#             self.setupUi(self)
#             self.model = YOLO("best.pt")
#             self.timer = QtCore.QTimer()
#             self.timer.timeout.connect(self.update_frame)
#
#             self.cap = None
#             self.is_detection_active = False
#             self.current_frame = None
#
#             # 各个按钮绑定功能
#             self.picture_detect.clicked.connect(self.load_picture)
#             self.video_detect.clicked.connect(self.load_video)
#             self.camera_detect.clicked.connect(self.load_camera)
#             self.start_detect.clicked.connect(self.start_detection)
#             self.stop_detect.clicked.connect(self.stop_detection)
#             self.pause_detect.clicked.connect(self.pause_detection)
#
#         except Exception as e:
#             print(e)
#
#
#     def load_picture(self):
#         try:
#             fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png)")
#             self.is_detection_active = False
#
#             if fileName:
#                 if self.timer.isActive():
#                     self.timer.stop()
#                 if self.cap:
#                     self.cap.release()
#                     self.cap = None
#
#                 self.current_frame = cv2.imread(fileName)
#                 self.display_image(self.current_frame, self.original_image)
#                 results = self.model.predict(self.current_frame)
#                 self.detected_frame = results[0].plot()  # 获取检测结果的帧并保存
#                 self.display_image(self.detected_frame, self.detected_image)
#         except Exception as e:
#             print(e)
#
#     def load_video(self):
#         fileName, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
#         if fileName:
#             if self.cap:
#                 self.cap.release()
#                 self.cap = None
#
#             self.cap = cv2.VideoCapture(fileName)
#
#             if self.cap.isOpened():
#                 ret, frame = self.cap.read()
#                 if ret:
#                     self.current_frame = frame.copy()
#                     self.display_image(frame, self.original_image)
#                     self.display_image(frame, self.detected_image)
#                 else:
#                     QtWidgets.QMessageBox.warning(self, 'Error', '无法读取视频文件的第一帧。')
#
#
#     def load_camera(self):
#         self.is_detection_active = False
#         if self.cap:
#             self.cap.release()
#         self.cap = cv2.VideoCapture(0)
#         self.timer.start(20)
#
#     def update_frame(self):
#         if self.cap:
#             ret, frame = self.cap.read()
#             if ret:
#                 self.current_frame = frame.copy()
#                 self.display_image(frame, self.original_image)
#
#                 if self.is_detection_active:
#                     results = self.model.predict(frame)
#                     self.detected_frame = results[0].plot()  # 获取检测结果的帧并保存
#                     self.display_image(self.detected_frame, self.detected_image)
#
#     def start_detection(self):
#         if self.cap and not self.cap.isOpened():
#             self.cap.open(self.fileName)
#         if self.cap and not self.timer.isActive():
#             self.timer.start(20)
#         self.is_detection_active = True
#
#     def pause_detection(self):
#             self.is_detection_active = False
#             if self.timer.isActive():
#                 self.timer.stop()
#
#     def display_image(self, frame, target_label):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         height, width, channel = frame.shape
#         step = channel * width
#         qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
#         pixmap = QPixmap.fromImage(qImg)
#         scaled_pixmap = pixmap.scaled(target_label.size(), QtCore.Qt.KeepAspectRatio)
#         target_label.setPixmap(scaled_pixmap)
#
#
#     def stop_detection(self):
#         self.is_detection_active = False
#
#         if self.timer.isActive():
#             self.timer.stop()
#
#         if self.cap:
#             self.cap.release()
#             self.cap = None
#
#         self.clear_display(self.original_image)
#         self.clear_display(self.detected_image)
#
#     def clear_display(self, target_label):
#         target_label.clear()
#         target_label.setText('')
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())



import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
from ultralytics import YOLO
from yolo import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setupUi(self)
            self.model = YOLO("best.pt")  # 确保模型路径正确
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_frame)

            self.cap = None
            self.is_detection_active = False
            self.current_frame = None

            # 车辆计数相关变量
            self.line_y_red = 430  # 红线位置
            self.class_counts = {}  # 车辆计数
            self.crossed_ids = set()  # 已通过红线的车辆ID
            self.tracked_objects = {}  # 跟踪车辆的中心点

            # 各个按钮绑定功能
            self.picture_detect.clicked.connect(self.load_picture)
            self.video_detect.clicked.connect(self.load_video)
            self.camera_detect.clicked.connect(self.load_camera)
            self.start_detect.clicked.connect(self.start_detection)
            self.stop_detect.clicked.connect(self.stop_detection)
            self.pause_detect.clicked.connect(self.pause_detection)

        except Exception as e:
            print(f"Initialization error: {e}")

    def load_picture(self):
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png)")
            self.is_detection_active = False

            if fileName:
                if self.timer.isActive():
                    self.timer.stop()
                if self.cap:
                    self.cap.release()
                    self.cap = None

                self.current_frame = cv2.imread(fileName)
                self.display_image(self.current_frame, self.original_image)
                results = self.model.predict(self.current_frame)
                self.detected_frame = results[0].plot()  # 获取检测结果的帧并保存
                self.display_image(self.detected_frame, self.detected_image)
        except Exception as e:
            print(f"Error loading picture: {e}")

    def load_video(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if fileName:
            if self.cap:
                self.cap.release()
                self.cap = None

            self.cap = cv2.VideoCapture(fileName)

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    self.display_image(frame, self.original_image)
                    self.display_image(frame, self.detected_image)
                else:
                    QtWidgets.QMessageBox.warning(self, 'Error', '无法读取视频文件的第一帧。')

    def load_camera(self):
        self.is_detection_active = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_image(frame, self.original_image)

                if self.is_detection_active:
                    results = self.model.track(frame, persist=True)  # 启用目标跟踪
                    self.detected_frame = results[0].plot()

                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu()
                        class_indices = results[0].boxes.cls.int().cpu().tolist()
                        confidences = results[0].boxes.conf.cpu()
                        track_ids = results[0].boxes.id  # 获取跟踪 ID

                        cv2.line(frame, (690, self.line_y_red), (1130, self.line_y_red), (0, 0, 255), 3)

                        for i, (box, class_idx, conf) in enumerate(zip(boxes, class_indices, confidences)):
                            x1, y1, x2, y2 = map(int, box)
                            cx = (x1 + x2) // 2  # 计算中心点
                            cy = (y1 + y2) // 2
                            class_name = self.model.names[class_idx]

                            track_id = int(track_ids[i]) if track_ids is not None else None  # 获取 track_id

                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # 只在目标首次穿过红线时计数
                            if track_id is not None and cy > self.line_y_red and track_id not in self.crossed_ids:
                                self.crossed_ids.add(track_id)
                                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

                    # 显示车辆计数信息
                    y_offset = 30
                    for class_name, count in self.class_counts.items():
                        cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y_offset += 30

                    self.display_image(frame, self.detected_image)

    def start_detection(self):
        if self.cap and not self.cap.isOpened():
            self.cap.open(self.fileName)
        if self.cap and not self.timer.isActive():
            self.timer.start(20)
        self.is_detection_active = True

    def pause_detection(self):
        self.is_detection_active = False
        if self.timer.isActive():
            self.timer.stop()

    def display_image(self, frame, target_label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        scaled_pixmap = pixmap.scaled(target_label.size(), QtCore.Qt.KeepAspectRatio)
        target_label.setPixmap(scaled_pixmap)

    def stop_detection(self):
        self.is_detection_active = False

        if self.timer.isActive():
            self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.clear_display(self.original_image)
        self.clear_display(self.detected_image)

        # 重置车辆计数相关变量
        self.class_counts = {}
        self.crossed_ids = set()
        self.tracked_objects = {}

    def clear_display(self, target_label):
        target_label.clear()
        target_label.setText('')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())