from fastapi import FastAPI, Query, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
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
from fastapi.responses import JSONResponse

# 创建文件夹
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 加载 YOLO 模型
app = FastAPI()

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，或者指定前端的域名如 ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

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

# @app.get("/detect")
# async def detect(
#     type: str = Query(..., description="检测类型: 'image' 或 'video'"),
#     filename: str = Query(..., description="图像或视频的本地路径")
# ):
#     # 其他部分保持不变
#
#     if type == "image":
#         result_path = process_image(filename)
#     elif type == "video":
#         result_path = process_video(filename)
#     else:
#         raise HTTPException(status_code=400, detail="无效的类型参数，使用 'image' 或 'video'")
#
#     # 返回相对路径
#     return JSONResponse(content={"message": "检测完成", "result_path": result_path})

# @app.get("/detect")
# async def detect(
#     type: str = Query(..., description="检测类型: 'image' 或 'video'"),
#     filename: str = Query(..., description="图像或视频的本地路径")
# ):
#     # 其他部分保持不变
#
#     if type == "image":
#         result_path = process_image(filename)  # 假设这是图像处理函数
#     elif type == "video":
#         result_path = process_video(filename)  # 假设这是视频处理函数
#     else:
#         raise HTTPException(status_code=400, detail="无效的类型参数，使用 'image' 或 'video'")
#
#     # 确保 result_path 是字符串类型
#     result_path = str(result_path)
#
#     # 返回可以正确序列化的 JSON 数据
#     return JSONResponse(content={"message": "检测完成", "result_path": result_path})
@app.get("/detect")
async def detect(
    type: str = Query(..., description="检测类型: 'image' 或 'video'"),
    filename: str = Query(..., description="图像或视频的本地路径")
):
    # 其他部分保持不变

    if type == "image":
        result_path = process_image(filename)  # 假设这是图像处理函数
    elif type == "video":
        result_path = process_video(filename)  # 假设这是视频处理函数
    else:
        raise HTTPException(status_code=400, detail="无效的类型参数，使用 'image' 或 'video'")

    # 确保 result_path 是字符串类型，并且是相对路径
    result_path = str(result_path)  # 将路径强制转换为字符串

    # 返回可以正确序列化的 JSON 数据
    return JSONResponse(content={"message": "检测完成", "result_path": result_path})

def process_image(filename):
    """ 处理图像 """
    image = cv2.imread(filename)
    if image is None:
        raise HTTPException(status_code=400, detail="无效的图像文件")

    # 打印调试信息
    print(f"正在处理图像: {filename}")

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
        # 绘制框和文本
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 确保结果路径正确
    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    print(f"保存处理后的图像到: {result_path}")
    cv2.imwrite(result_path, image)

    # 保存 JSON 结果
    json_result_path = save_json_results(detections, os.path.splitext(result_path)[0] + ".json")
    print(f"保存检测结果到: {json_result_path}")

    return JSONResponse(content={"message": "检测完成", "image_result_path": result_path, "json_result_path": json_result_path})

def process_video(filename):
    """ 处理视频 """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="无效的视频文件")

    # 视频输出路径
    result_path = os.path.join(RESULT_DIR, os.path.basename(filename))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(result_path, fourcc, fps, frame_size)

    frame_number = 0
    video_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理每一帧
        results = model.predict(frame)
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
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        video_detections.append({"frame": frame_number, "detections": detections})
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    print(f"保存处理后的视频到: {result_path}")

    # 保存 JSON 结果
    json_result_path = save_json_results(video_detections, os.path.splitext(result_path)[0] + ".json")
    print(f"保存视频检测结果到: {json_result_path}")

    return JSONResponse(content={"message": "视频检测完成", "video_result_path": result_path, "json_result_path": json_result_path})

def save_json_results(detections, result_path):
    """ 保存检测结果为 JSON """
    with open(result_path, "w") as f:
        json.dump({"detections": [d.model_dump() if isinstance(d, DetectionResult) else d for d in detections]}, f)
    return result_path


# 允许前端访问处理后的文件
@app.get("/results/{filename}")
async def get_result_file(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果文件未找到")
    return FileResponse(file_path)

# 启动 FastAPI 应用
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
