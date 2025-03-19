from ultralytics import YOLO
import warnings

def train_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = YOLO('ultralytics/cfg/models/v8/YOLOv8_NAM.yaml')  # 从YAML建立一个新模型
        results = model.train(
            data=r"E:\PythonDemo\yolov8_vihicleCounting\mktk_dataset\data.yaml",
            device='0',  # 使用GPU训练
            epochs=500,  # 训练轮数
            batch=8,
            verbose=False,
            imgsz=640
        )
        print(results)

if __name__ == '__main__':
    train_model()