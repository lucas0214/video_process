from transformers import AutoModel, AutoTokenizer
import cv2
from tqdm import tqdm
import json
import numpy as np
import os
import tempfile
from PIL import Image


def load_model():
    tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/pre-trained/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('/home/ubuntu/pre-trained/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True,
                                      device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()

    return tokenizer, model


def frame_process(tokenizer, model, image_file):
    # plain texts OCR
    res = model.chat(tokenizer, image_file, ocr_type='ocr')

    # format texts OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='format')

    # fine-grained OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

    # multi-crop OCR:
    # res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
    # res = model.chat_crop(tokenizer, image_file, ocr_type='format')

    # render the formatted OCR results:
    # res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

    print(res)
    return res


def video_to_frames(video_path, save_dir, video):
    save_dir = save_dir + "/" + os.path.basename(video).split(".")[0] + "/" + os.path.basename(video_path).split(".")[0]
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建一个视频捕获对象
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 1

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 20 == 0:
            # 构造图像文件的路径
            frame_filename = os.path.join(save_dir, f'frame_{frame_count}.jpg')
            # 保存图像
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(os.path.abspath(frame_filename))

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    return frame_paths


def write_json(json_path, information_video):
    with open(json_path, 'a', encoding='utf-8') as file:
        json.dump(information_video, file, ensure_ascii=False, indent=4)
        file.write(",\n")


tokenizer, model = load_model()

with open("/opt/dlami/nvme/ocr_4o/ocr_process/video_infor.json", "r")as f:
    data = json.load(f)

video_info_list = []
for da in tqdm(data):
    data_dict = {}
    data_dict["input_video"] = da["input_video"]
    clips_list = []
    for clips in da["clips"]:
        clip_dict = {}
        clip_dict["clip_number"] = clips["clip_number"]
        clip_dict["start_time"] = clips["start_time"]
        clip_dict["end_time"] = clips["end_time"]
        clip_dict["filename"] = clips["filename"]

        frames = video_to_frames(clips["filename"], "/mnt/s3bucket/frames", da["input_video"])
        ocr_frames_list = []
        for frame in frames:
            ocr_frame = frame_process(tokenizer, model, frame)
            ocr_frames_list.append(ocr_frame)
        clip_dict["ocr"] = ocr_frames_list

        clips_list.append(clip_dict)

    data_dict["clips"] = clips_list

    write_json("/opt/dlami/nvme/ocr_4o/ocr_process/all_video_ocr.json", data_dict)