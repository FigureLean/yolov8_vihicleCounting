# 训练
# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# Train the model with 2 GPUs
# results = model.train(data="E:\\pythonDemo\\yolov8.3\\datasets\\trash\\trash.yaml", epochs=10, imgsz=640, device='0')
# results = model.train(data="E:\\pythonDemo\\yolov8.3\\datasets\\trash\\trash.yaml", epochs=10, imgsz=640,device='0')

from ultralytics import YOLO

if __name__ == '__main__':
    # 加载一个模型
    model = YOLO('yolov8n.yaml')  # 从YAML建立一个新模型
    # 训练模型
    results = model.train(
        data="E:\\pythonDemo\\yolov8.3\\datasets\\vehicle\\data.yaml",
        device='0',
        epochs=200,
        batch=8,
        verbose=False,
        imgsz=640)


# 也可以直接在终端输入命令
# yolo detect train data=datasets/trash/trash.yaml model=yolov8n.yaml pretrained=ultralytics/yolov8n.pt epochs=100 batch=4 lr0=0.01 resume=True

# import torch
# print(torch.cuda.is_available())  # 应该输出True
# print(torch.cuda.device_count())  # 输出你的GPU数量
# print(torch.cuda.get_device_name(0))  # 输出第一个GPU的名称