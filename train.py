# 训练
# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# Train the model with 2 GPUs
# results = model.train(data="E:\\pythonDemo\\yolov8.3\\datasets\\trash\\trash.yaml", epochs=10, imgsz=640, device='0')
# results = model.train(data="E:\\pythonDemo\\yolov8.3\\datasets\\trash\\trash.yaml", epochs=10, imgsz=640,device='0')

# from ultralytics import YOLO

# if __name__ == '__main__':
#     # 加载一个模型
#     model = YOLO('yolov8n.pt')  # 加载预训练权重
#     # 训练模型
#     results = model.train(
#         data= r"E:\PythonDemo\human_dataset\data.yaml",
#         device='0', #使用GPU训练
#         epochs=200, #训练轮数
#         batch=8,
#         verbose=False,
#         imgsz=640)


# 也可以直接在终端输入命令
# yolo detect train data=datasets/trash/trash.yaml model=yolov8n.yaml pretrained=ultralytics/yolov8n.pt epochs=100 batch=4 lr0=0.01 resume=True
