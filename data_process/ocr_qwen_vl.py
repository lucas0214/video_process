import json
import os.path

import torch
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def write_json(json_path, information_video):
    with open(json_path, 'a', encoding='utf-8') as file:
        json.dump(information_video, file, ensure_ascii=False, indent=4)
        file.write(", \n")


def load_qwen2_vl_model(model_path="/home/ubuntu/pre-trained/Qwen/Qwen2-VL-7B-Instruct"):
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# "Perform OCR on any words, numbers, or symbols that appear in the video_shot, and describe them in detail by combining the OCR-recognized content with the full visual context. That is, do not just provide isolated word or number recognition, but provide relatively complete and detailed descriptions. For example, you cannot directly describe 'the building marked 45' or 'the sign marked 'exit''. Instead, you should integrate them into the details of the entire video_shot description, ensuring that each description captures a natural and comprehensive context, and does not provide inferences and additional information."

def gen_ocr(model, processor, video_path, ocr_info):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_shot",
                    "video_shot": "file://" + video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text",
                 "text": "I will provide you with a list of OCR results for this video_shot, which is recognized in the order of the video_shot frames. Please describe the video_shot in detail and accurately based on the OCR list and the video_shot. All the recognition results in the OCR list must be accurately applied to the video_shot description. This is the OCR result list of the entire video_shot: ({}).".format(
                     ocr_info)}
            ],
        }
    ]
    """
    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]
    for frame in frame_list:
        frame_dict = {"type": "image", "image": "file://" + frame}
        messages[0]["content"].append(frame_dict)

    text_dict = {"type": "text", "text": "These images are video_shot frames from a video_shot. I will provide you with a list of OCR results of all video_shot frames. Please combine the OCR results of all frames to describe the entire video_shot in detail. The OCR results must be embedded in the description of the entire video_shot. The list of OCR results of all video_shot frames is as follows: {}".format(ocr_info)}
    messages[0]["content"].append(text_dict)
    """
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=400)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # 清理内存
    del inputs
    del generated_ids
    torch.cuda.empty_cache()  # 清理缓存

    print(output_text)

    return output_text


def get_jpg_files_paths(folder_path):
    # 初始化一个空列表来存储.jpg文件的绝对路径
    jpg_files_paths = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为.jpg（不区分大小写）
            if file.lower().endswith('.jpg'):
                # 将文件的绝对路径添加到列表中
                jpg_files_paths.append(os.path.join(root, file))

                # 返回包含所有.jpg文件绝对路径的列表
    return jpg_files_paths


def load_video():
    model, processor = load_qwen2_vl_model("/home/ubuntu/pre-trained/Qwen/Qwen2-VL-7B-Instruct")

    split_ocr_data = read_json("/opt/dlami/nvme/ocr_4o/ocr_process/video_ocr.json")

    for clip_ocr in tqdm(split_ocr_data):
        videos_dict = {}
        clip_list = []

        videos_dict["input_video"] = clip_ocr["input_video"]

        video_clips = clip_ocr["clips"]
        for video in video_clips:
            print(video["filename"])
            video_dict = {}
            video_dict["clip_number"] = video["clip_number"]
            video_dict["start_time"] = video["start_time"]
            video_dict["end_time"] = video["end_time"]
            video_dict["filename"] = video["filename"]
            video_dict["ocr"] = video["ocr"]

            """
            frame_path = "/opt/dlami/nvme/ocr_4o/ocr_process/frames/"
            file_path = os.path.basename(clip_ocr["input_video"]).split(".")[0]
            scene_path = os.path.basename(video_shot["filename"]).split(".")[0]
            frames_path = frame_path + file_path + "/" + scene_path

            frames_path_list = get_jpg_files_paths(frames_path)
            """

            # video_dict["ocr_caption"] = gen_ocr(model, processor, video_shot["ocr"], frames_path_list)
            video_dict["ocr_caption"] = gen_ocr(model, processor, video["filename"], video["ocr"])

            clip_list.append(video_dict)

        videos_dict["clips"] = clip_list

        write_json("/opt/dlami/nvme/ocr_4o/data/text/captions.json", videos_dict)


if __name__ == '__main__':
    load_video()