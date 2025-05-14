# from fastapi import FastAPI, Query, HTTPException, File, UploadFile, Request
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import shutil
# import os
# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO
# from pydantic import BaseModel

# # 创建文件夹
# UPLOAD_DIR = "uploads"
# RESULT_DIR = "results"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RESULT_DIR, exist_ok=True)

# # 初始化全局变量
# track_history = {}  # 保存目标ID和历史位置
# vehicle_in = 0  # 进入车辆计数
# vehicle_out = 0  # 离开车辆计数

# # 加载 YOLO 模型
# app = FastAPI()

# # 允许跨域访问
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 配置静态文件访问
# app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

# try:
#     model = YOLO("best.pt")  # 替换为你的模型路径
#     model.eval()
# except Exception as e:
#     print(f"模型加载失败: {str(e)}")
#     model = None

# class DetectionResult(BaseModel):
#     class_name: str
#     confidence: float
#     x1: int
#     y1: int
#     x2: int
#     y2: int

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         return {"message": "上传成功", "file_path": file_path}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# @app.get("/detect")
# async def detect(type: str = Query(...), filename: str = Query(...)):
#     if type == "image":
#         result_path = process_image(filename)
#     elif type == "video":
#         result_path = process_video(filename)
#     else:
#         raise HTTPException(status_code=400, detail="无效的类型参数")
#     return JSONResponse(content={"message": "检测完成", "result_path": os.path.basename(result_path)})

# def process_image(filename):
#     image = cv2.imread(filename)
#     if image is None:
#         raise HTTPException(status_code=400, detail="无效的图像文件")

#     results = model.predict(image)
#     for det in results[0].boxes.data.cpu().numpy():
#         x1, y1, x2, y2, conf, cls = det
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#     result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
#     cv2.imwrite(result_path, image)
#     return result_path

# def process_video(filename):
#     """处理视频，并添加车辆计数、目标框和中心点"""
#     cap = cv2.VideoCapture(filename)
#     if not cap.isOpened():
#         raise Exception("无效的视频文件")

#     result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
#     fourcc = cv2.VideoWriter_fourcc(*"avc1")
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     out = cv2.VideoWriter(result_path, fourcc, fps, frame_size)

#     global track_history, vehicle_in, vehicle_out
#     track_history.clear()  # 清空历史轨迹，避免前一个视频数据影响当前视频
#     vehicle_in, vehicle_out = 0, 0  # 重新计数

#     # 初始化卡尔曼滤波器
#     kf = cv2.KalmanFilter(4, 2)
#     kf.transitionMatrix = np.array([[1, 0, 1, 0],
#                                     [0, 1, 0, 1],
#                                     [0, 0, 1, 0],
#                                     [0, 0, 0, 1]], np.float32)
#     kf.measurementMatrix = np.array([[1, 0, 0, 0],
#                                      [0, 1, 0, 0]], np.float32)
#     kf.processNoiseCov = np.array([[1, 0, 0, 0],
#                                    [0, 1, 0, 0],
#                                    [0, 0, 1, 0],
#                                    [0, 0, 0, 1]], np.float32) * 0.03

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model.track(frame, conf=0.3, persist=True)  # 目标检测 + 目标跟踪
#         if results[0].boxes.id is not None:
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#         else:
#             track_ids = []

#         for track_id, box in zip(track_ids, results[0].boxes.data):
#             x1, y1, x2, y2, conf, cls = map(int, box[:6])  # 获取目标边界框
#             class_name = model.names[cls]  # 获取类别名称

#             # 绘制目标框
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"#{track_id} {class_name} {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # 计算中心点
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # 画出中心红点

#             # 更新卡尔曼滤波器
#             kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
#             prediction = kf.predict()
#             predicted_cx, predicted_cy = int(prediction[0]), int(prediction[1])

#             # 更新 track_history
#             if track_id not in track_history:
#                 track_history[track_id] = []
#             track_history[track_id].append((predicted_cx, predicted_cy))

#             # 车辆进出计数
#             if len(track_history[track_id]) > 1:
#                 _, prev_y = track_history[track_id][-2]
#                 if prev_y < line_y_red and predicted_cy >= line_y_red:
#                     vehicle_out += 1
#                 elif prev_y > line_y_red and predicted_cy <= line_y_red:
#                     vehicle_in += 1

#         # 绘制基准线
#         line_y_red = frame_size[1] // 2  # 设置红线为视频高度的正中央
#         cv2.line(frame, (0, line_y_red), (frame_size[0], line_y_red), (25, 33, 189), 2)
#         cv2.putText(frame, f'in: {vehicle_in}', (595, line_y_red - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         cv2.putText(frame, f'out: {vehicle_out}', (573, line_y_red + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         out.write(frame)

#     cap.release()
#     out.release()
#     return result_path

# @app.get("/results/{filename}")
# async def get_result_file(filename: str):
#     file_path = os.path.join(RESULT_DIR, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="结果文件未找到")
#     return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)

from fastapi import FastAPI, Query, HTTPException, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pydantic import BaseModel

# 创建文件夹
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 初始化全局变量
track_history = {}  # 保存目标ID和历史位置
vehicle_in = 0  # 进入车辆计数
vehicle_out = 0  # 离开车辆计数

# 加载 YOLO 模型
app = FastAPI()

# 允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件访问
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

try:
    model = YOLO("best.pt")  # 替换为你的模型路径
    model.eval()
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    model = None

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "上传成功", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@app.get("/detect")
async def detect(type: str = Query(...), filename: str = Query(...)):
    if type == "image":
        result_path = process_image(filename)
    elif type == "video":
        result_path = process_video(filename)
    else:
        raise HTTPException(status_code=400, detail="无效的类型参数")
    return JSONResponse(content={"message": "检测完成", "result_path": os.path.basename(result_path)})

def process_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise HTTPException(status_code=400, detail="无效的图像文件")

    results = model.predict(image)
    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    cv2.imwrite(result_path, image)
    return result_path

def process_video(filename):
    """处理视频，并添加车辆计数、目标框和中心点"""
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise Exception("无效的视频文件")

    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(result_path, fourcc, fps, frame_size)

    global track_history, vehicle_in, vehicle_out
    track_history.clear()  # 清空历史轨迹，避免前一个视频数据影响当前视频
    vehicle_in, vehicle_out = 0, 0  # 重新计数

    line_y_red = frame_size[1] // 2  # 设置红线为视频高度的正中央
    counted_ids = set()  # 用于记录已经计数的ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=0.3, persist=True)  # 目标检测 + 目标跟踪
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []

        for track_id, box in zip(track_ids, results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = map(int, box[:6])  # 获取目标边界框
            class_name = model.names[cls]  # 获取类别名称

            # 绘制目标框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{track_id} {class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 计算中心点
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # 画出中心红点

            # 车辆进出计数
            if track_id not in counted_ids:
                if cy > line_y_red:
                    vehicle_out += 1
                    counted_ids.add(track_id)
                elif cy < line_y_red:
                    vehicle_in += 1
                    counted_ids.add(track_id)

        # 绘制基准线
        cv2.line(frame, (0, line_y_red), (frame_size[0], line_y_red), (25, 33, 189), 2)
        cv2.putText(frame, f'in: {vehicle_in}', (595, line_y_red - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f'out: {vehicle_out}', (573, line_y_red + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return result_path

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果文件未找到")
    return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)