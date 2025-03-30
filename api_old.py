from fastapi import FastAPI, Query, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import torch
import os
import cv2
import json
from collections import defaultdict
from result_vedio import fourcc
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List

# 创建文件夹
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 初始化全局变量
track_history = defaultdict(lambda: [])  # 保存目标ID和其历史位置
vehicle_in = 0  # 进入车辆计数
vehicle_out = 0  # 离开车辆计数
line_y_red = 430  # 基准线位置

# 加载 YOLO 模型
app = FastAPI()


# 允许跨域访问（确保前端可以访问后端资源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（可根据需要修改）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件访问（映射 /results/ 到 results 目录）
results_dir = "E:/PythonDemo/yolov8_vihicleCounting/results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

app.mount("/results", StaticFiles(directory=results_dir), name="results")


try:
    model = YOLO("E:\\PythonDemo\\yolov8_vihicleCounting\\best.pt")  # 请确保该路径正确
    model.eval()
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    model = None  # 避免程序直接崩溃

# 定义检测结果的数据结构
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

# 允许上传文件
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "上传成功", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 检测接口
@app.get("/detect")
async def detect(
    type: str = Query(..., description="检测类型: 'image' 或 'video'"),
    filename: str = Query(..., description="图像或视频的本地路径")
):
    if type == "image":
        result_path = process_image(filename)  # 假设这是图像处理函数
    elif type == "video":
        result_path = process_video(filename)  # 假设这是视频处理函数
    else:
        raise HTTPException(status_code=400, detail="无效的类型参数，使用 'image' 或 'video'")

    # 返回检测结果文件的相对路径
    return JSONResponse(content={"message": "检测完成", "result_path": os.path.basename(result_path)})

def process_image(filename):
    """ 处理图像 """
    image = cv2.imread(filename)
    if image is None:
        raise HTTPException(status_code=400, detail="无效的图像文件")

    results = model.predict(image)
    detections = []

    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        class_name = model.names[int(cls)]
        detections.append(
            DetectionResult(
                class_name=class_name,
                confidence=float(conf),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2)
            )
        )
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    cv2.imwrite(result_path, image)
    return result_path


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """ 在图像上绘制目标框和标签 """
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, 2 / 3, txt_color, thickness=1, lineType=cv2.LINE_AA)

def process_video(filename):
    """ 处理视频，并添加车辆计数功能 """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="无效的视频文件")

    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(result_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=0.3, persist=True)  # 使用模型跟踪目标
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []

        for track_id, box in zip(track_ids, results[0].boxes.data):
            if box[-1] == 2:  # 小汽车目标
                box_label(frame, box, '#' + str(track_id) + ' car', (167, 146, 11))
                x1, y1, x2, y2 = box[:4]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                track = track_history[track_id]
                track.append((float(x), float(y)))

                if len(track) > 1:
                    _, h = track[-2]
                    if h < line_y_red and y >= line_y_red:
                        vehicle_out += 1
                    elif h > line_y_red and y <= line_y_red:
                        vehicle_in += 1

        cv2.line(frame, (30, line_y_red), (frame_size[0] - 30, line_y_red), (25, 33, 189), 2)
        cv2.putText(frame, 'in: ' + str(vehicle_in), (595, line_y_red - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, 'out: ' + str(vehicle_out), (573, line_y_red + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return result_path
# def process_video(filename):
#     """ 处理视频 """
#     cap = cv2.VideoCapture(filename)
#     if not cap.isOpened():
#         raise HTTPException(status_code=400, detail="无效的视频文件")
#
#     result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
#     # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fourcc = cv2.VideoWriter_fourcc(*"avc1")
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     out = cv2.VideoWriter(result_path, fourcc, fps, frame_size)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         results = model.predict(frame)
#         for det in results[0].boxes.data.cpu().numpy():
#             x1, y1, x2, y2, conf, cls = det
#             class_name = model.names[int(cls)]
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         out.write(frame)
#
#     cap.release()
#     out.release()
#     return result_path
#

# 弃用
# def process_video(filename):
#     """ 处理视频并进行车辆计数 """
#     cap = cv2.VideoCapture(filename)
#     if not cap.isOpened():
#         raise HTTPException(status_code=400, detail="无效的视频文件")
#
#     result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
#     fourcc = cv2.VideoWriter_fourcc(*"avc1")
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     size = (frame_width, frame_height)
#     out = cv2.VideoWriter(result_path, fourcc, fps, size)
#
#     # 车辆计数相关变量
#     line_y_red = frame_height - 200  # 红线位置
#     buffer_height = 20  # 缓冲区高度
#     class_counts = {}  # 车辆计数
#     crossed_ids = set()  # 已通过红线的车辆ID
#     track_history = defaultdict(list)  # 保存每个目标的历史位置
#
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#
#         # 在帧上运行YOLOv8跟踪，persist为True表示保留跟踪信息，conf为0.3表示只检测置信值大于0.3的目标
#         results = model.track(frame, conf=0.3, persist=True)
#
#         # 得到该帧的各个目标的ID
#         if results[0].boxes.id is not None:
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#         else:
#             track_ids = []  # 如果没有检测到目标，则track_ids为空列表
#
#         # 遍历该帧的所有目标
#         for track_id, box in zip(track_ids, results[0].boxes.data):
#             class_id = int(box[-1])  # 获取目标类别ID
#             if class_id in [2, 5, 7, 3]:  # 只处理小汽车、巴士、卡车、摩托车
#                 # 绘制该目标的矩形框
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"#{track_id} {model.names[class_id]}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                 # 得到该目标矩形框的中心点坐标(x, y)
#                 cx = (x1 + x2) // 2
#                 cy = (y1 + y2) // 2
#
#                 # 绘制中心点（红点）
#                 cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
#
#                 # 保存目标的历史位置
#                 track_history[track_id].append((cx, cy))
#
#                 # 检查目标是否首次穿过红线
#                 if track_id not in crossed_ids:
#                     if len(track_history[track_id]) > 1:
#                         prev_cy = track_history[track_id][-2][1]
#                         if prev_cy > line_y_red + buffer_height and cy < line_y_red:
#                             crossed_ids.add(track_id)
#                             class_name = model.names[class_id]
#                             class_counts[class_name] = class_counts.get(class_name, 0) + 1
#
#         # 绘制红线
#         cv2.line(frame, (0, line_y_red), (frame_width, line_y_red), color=(0, 0, 255), thickness=2)
#
#         # 显示车辆计数信息
#         y_offset = 30
#         for class_name, count in class_counts.items():
#             cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             y_offset += 30
#
#         # 写入保存
#         out.write(frame)
#
#     cap.release()
#     out.release()
#     return result_path

# 提供检测结果文件的访问接口
@app.get("/results/{filename}")
async def get_result_file(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果文件未找到")

    # 以流式方式读取文件
    def iterfile():
        with open(file_path, mode="rb") as file:
            while chunk := file.read(1024 * 1024):  # 每次读取 1MB
                yield chunk

    return StreamingResponse(iterfile(), media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)