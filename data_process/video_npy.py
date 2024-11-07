import os
import cv2
import numpy as np
import json


def convert_mp4_to_npy(mp4_list, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 存储每个视频的字典信息
    video_info_list = []

    for mp4_file in mp4_list:
        # 读取.mp4文件的绝对路径
        video_path = os.path.abspath(mp4_file)

        # 提取文件名并替换扩展名
        file_name = os.path.splitext(os.path.basename(mp4_file))[0]
        npy_file = f"{file_name}.npy"
        npy_path = os.path.join(output_folder, npy_file)

        # 加载视频内容并保存为npy文件
        video_data = []
        cap = cv2.VideoCapture(mp4_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_data.append(frame)
        cap.release()

        # 转换为numpy数组并保存
        np.save(npy_path, np.array(video_data))

        # 创建字典并添加到列表
        video_info = {
            "video_path": video_path,
            "npy_path": os.path.abspath(npy_path)
        }
        video_info_list.append(video_info)

    # 将所有视频的信息保存到JSON文件
    json_path = os.path.join(output_folder, "video_info.json")
    with open(json_path, "w") as json_file:
        json.dump(video_info_list, json_file, indent=4)

    print(f"所有视频信息已保存到: {json_path}")
