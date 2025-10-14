# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com.
"""

import os

import torch

from ultralytics import YOLO

print("Visible GPU count:", torch.cuda.device_count())
print("Using device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
if __name__ == "__main__":
    # Load a model
    # model = YOLO(model="/home/gysj_wp/workspace/yolov12/yolov12m-turbo.pt")
    model = YOLO(model="/home/gysj_wp/workspace/yolov12/runs/train/exp16/weights/best.pt")
    root = "datasets/aokeng2/small/images/test"
    files = os.listdir(root)
    # # 创建临时文件夹用于存储翻转后的图片
    # temp_dir = "/home/gysj_wp/workspace/datasets/changed_img"
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    # file_paths=[os.path.join(root,name) for name in files if os.path.isfile(os.path.join(root,name)) and  name.lower().endswith(".bmp")]
    file_paths = []
    for file_name in files:
        file_path = os.path.join(root, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(".jpg"):
            # image = cv2.imread(file_path)
            # image = cv2.flip(image,0)
            # # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            # temp_file_path = os.path.join(temp_dir, file_name)
            # cv2.imwrite(temp_file_path,image)
            file_paths.append(file_path)

    print(model.get_parameter)
    model.predict(
        source=file_paths,
        save=True,
        show=False,
        project="/home/gysj_wp/workspace/yolov12/runs/detect/exp",
        save_txt=False,
        save_conf=False,
        device=4,
        classes=None,
    )
    # #删除临时文件夹的内容
    # for file in os.listdir(temp_dir):
    #     file_path = os.path.join(temp_dir,file)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    # print("删除临时文件夹的内容完成")
