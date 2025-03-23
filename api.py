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
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List

# 创建文件夹
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

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
    model = YOLO("best.pt")  # 请确保该路径正确
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

# def process_video(filename):
#     """ 处理视频 """
#     cap = cv2.VideoCapture(filename)
#     if not cap.isOpened():
#         raise HTTPException(status_code=400, detail="无效的视频文件")
#
#     result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

def process_video(filename):
    """ 处理视频 """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="无效的视频文件")

    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(result_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            class_name = model.names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return result_path

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