import os
import random
import shutil
import time
import yaml
import logging

# 初始化日志
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, 'split_data.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOTrainDataSetGenerator:
    def __init__(self, origin_dataset_dir, train_dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 clear_train_dir=False):
        # 设置随机数种子
        random.seed(1233)

        self.origin_dataset_dir = origin_dataset_dir
        self.train_dataset_dir = train_dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.clear_train_dir = clear_train_dir

        assert self.train_ratio > 0.5, 'train_ratio must larger than 0.5'
        assert self.val_ratio > 0.01, 'val_ratio must larger than 0.01'
        assert self.test_ratio > 0.01, 'test_ratio must larger than 0.01'
        total_ratio = round(self.train_ratio + self.val_ratio + self.test_ratio)
        assert total_ratio == 1.0, 'train_ratio + val_ratio + test_ratio must equal 1.0'

    def generate(self):
        time_start = time.time()
        logging.info(f'start to split origin data set. \n'
                     f'origin_dataset_dir:{self.origin_dataset_dir},\n'
                     f'train_dataset_dir:{self.train_dataset_dir},\n'
                     f'train_ratio:{self.train_ratio},val_ratio:{self.val_ratio}, test_ratio:{self.test_ratio}')
        # 原始数据集的图像目录，标签目录，和类别文件路径
        origin_image_dir = os.path.join(self.origin_dataset_dir, 'images')
        origin_label_dir = os.path.join(self.origin_dataset_dir, 'labels')
        origin_classes_file = os.path.join(self.origin_dataset_dir, 'classes.txt')
        if not os.path.exists(origin_classes_file):
            logging.error(f'classes file is not found. classes_file:{origin_classes_file}')
            return
        else:
            origin_classes = {}
            with open(origin_classes_file, mode='r', encoding='utf-8') as f:
                for cls_id, cls_name in enumerate(f.readlines()):
                    cls_name = cls_name.strip()
                    if cls_name != '':
                        origin_classes[cls_id] = cls_name

        # 获取所有原始图像文件名（包括后缀名）
        origin_image_filenames = os.listdir(origin_image_dir)

        # 随机打乱文件名列表
        random.shuffle(origin_image_filenames)

        # 计算训练集、验证集和测试集的数量
        total_count = len(origin_image_filenames)
        train_count = int(total_count * self.train_ratio)
        val_count = int(total_count * self.val_ratio)
        test_count = total_count - train_count - val_count

        # 定义训练集文件夹路径
        if self.clear_train_dir and os.path.exists(self.train_dataset_dir):
            shutil.rmtree(self.train_dataset_dir, ignore_errors=True)
        train_dir = os.path.join(self.train_dataset_dir, 'train')
        val_dir = os.path.join(self.train_dataset_dir, 'val')
        test_dir = os.path.join(self.train_dataset_dir, 'test')
        train_image_dir = os.path.join(train_dir, 'images')
        train_label_dir = os.path.join(train_dir, 'labels')
        val_image_dir = os.path.join(val_dir, 'images')
        val_label_dir = os.path.join(val_dir, 'labels')
        test_image_dir = os.path.join(test_dir, 'images')
        test_label_dir = os.path.join(test_dir, 'labels')

        # 创建训练集输出文件夹
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)
        os.makedirs(test_image_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # 将图像和标签文件按设定的ratio划分到训练集，验证集，测试集中
        for i, filename in enumerate(origin_image_filenames):
            if i < train_count:
                output_image_dir = train_image_dir
                output_label_dir = train_label_dir
            elif i < train_count + val_count:
                output_image_dir = val_image_dir
                output_label_dir = val_label_dir
            else:
                output_image_dir = test_image_dir
                output_label_dir = test_label_dir
            src_img_name_no_ext = os.path.splitext(filename)[0]
            src_image_path = os.path.join(origin_image_dir, filename)
            src_label_path = os.path.join(origin_label_dir, src_img_name_no_ext + '.txt')
            if os.path.exists(src_label_path):
                # 复制图像文件
                dst_image_path = os.path.join(output_image_dir, filename)
                shutil.copy(src_image_path, dst_image_path)
                # 复制标签文件
                src_label_path = os.path.join(origin_label_dir, src_img_name_no_ext + '.txt')
                dst_label_path = os.path.join(output_label_dir, src_img_name_no_ext + '.txt')
                shutil.copy(src_label_path, dst_label_path)
            else:
                logging.error(f'no label file found for image file. img_file:{src_image_path}')
        train_dir = os.path.normpath(train_dir)
        val_dir = os.path.normpath(val_dir)
        test_dir = os.path.normpath(test_dir)
        logging.info(f'generate train, val, test data set. \n'
                     f'train_count:{train_count}, train_dir:{train_dir}\n'
                     f'val_count:{val_count}, val_dir:{val_dir}\n'
                     f'test_count:{test_count}, test_dir:{test_dir}')
        # 生成描述训练集的yaml文件
        data_dict = {
            'train': train_dir,
            'val': val_dir,
            'test': test_dir,
            'nc': len(origin_classes),
            'names': origin_classes
        }

        yaml_file_path = os.path.normpath(os.path.join(self.train_dataset_dir, 'data.yaml'))
        with open(yaml_file_path, mode='w', encoding='utf-8') as f:
            yaml.safe_dump(data_dict, f, default_flow_style=False, allow_unicode=True)
        logging.info(f'generate the `data.yaml`. data:{data_dict}, yaml_file_path:{yaml_file_path}')
        logging.info('end to ')

if __name__ == '__main__':
    g_origin_dataset_dir = r'E:\\PythonDemo\\yolov8_vihicleCounting\\dataset'
    g_train_dataset_dir = r'E:\\PythonDemo\\yolov8_vihicleCounting\\mktk_dataset'
    g_train_ratio = 0.7
    g_val_ratio = 0.15
    g_test_ratio = 0.15
    yolo_generator = YOLOTrainDataSetGenerator(g_origin_dataset_dir, g_train_dataset_dir, g_train_ratio, g_val_ratio,
                                               g_test_ratio, True)
    yolo_generator.generate()