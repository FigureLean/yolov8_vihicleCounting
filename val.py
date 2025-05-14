from ultralytics import YOLO

def main():
    pt_file_address = r"E:\\PythonDemo\\yolov8_vihicleCounting\\best.pt"  # 需要修改的地址
    model = YOLO(pt_file_address)
    metrics = model.val()  # 在执行这一行的时候，系统会自动找你训练时候的 yaml，若报错，看报错提示，需要将 yaml 放在对应位置

    # 运行上面代码结束后，在下面其实已经给出了相应 p、r、map
    # 在路径下会有 val 文件夹，里面会有各种曲线图
    # print(metrics)  # 查看 metrics 的所有存储内容
    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.75:", metrics.box.map75)
    print("Precision:", metrics.results_dict['metrics/precision(B)'])  # 获得更精确的值

if __name__ == '__main__':
    main()