from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import cv2
from tqdm import tqdm
from scenedetect import VideoManager, SceneManager, StatsManager, ContentDetector
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def get_mp4_file_paths(folder_path):
    """获取文件夹中所有 .mp4 文件的绝对路径"""
    mp4_file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp4'):
            absolute_path = os.path.abspath(os.path.join(folder_path, file_name))
            mp4_file_paths.append(absolute_path)
    return mp4_file_paths


def detect_scenes_in_video(video_path, threshold=15, min_scene_length_sec=3):
    """使用 PySceneDetect 对视频进行场景切分"""
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    # 设置缩放比例提高处理速度
    video_manager.set_downscale_factor()
    video_manager.start()

    # 获取帧率和视频总时长
    video_fps = video_manager.get_framerate()
    total_seconds = video_manager.get_duration()[2].get_seconds()
    min_scene_length = int(min_scene_length_sec * video_fps)

    # 添加内容检测器
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_length))
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    ext = os.path.splitext(video_path)[1]
    file_save = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join("/opt/dlami/nvme/ocr_4o/data/video_shot/" + file_save, 'scenes')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(video_path)
    scenes_info = {"input_video": video_path, "clips": []}

    if not scene_list:
        scenes_info["clips"].append({
            "clip_number": 1,
            "start_time": 0,
            "end_time": total_seconds,
            "filename": video_path
        })
        return scenes_info

    for i, scene in enumerate(scene_list):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        start_time, end_time = scene[0].get_timecode(), scene[1].get_timecode()
        scene_filename = os.path.join(output_dir, f"scene_{i + 1}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(scene_filename, fourcc, video_fps, (
            int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                              )

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = video_capture.read()
            if not ret:
                break
            out.write(frame)
        out.release()

        scenes_info["clips"].append({
            "clip_number": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "filename": scene_filename
        })

    video_manager.release()
    video_capture.release()

    return scenes_info


def process_video_info_multithread(video_data_path, json_file_path):
    """使用多线程处理视频并保存分割信息到 JSON 文件"""
    # video_list = get_mp4_file_paths(video_data_path)
    video_list = video_data_path

    information_list = []

    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(detect_scenes_in_video, video_path, 15) for video_path in video_list]

        for future in tqdm(as_completed(futures), total=len(video_list)):
            try:
                information = future.result()
                information_list.append(information)
            except Exception as e:
                print(f"处理视频时出错：{e}")

    with open(json_file_path, 'w') as json_file:
        json.dump(information_list, json_file)

    print(f"数据已写入 {json_file_path}")


if __name__ == '__main__':
    # video_path = "/opt/dlami/nvme/ocr_4o/da/videomme/data"
    video_path = []

    with open('/opt/dlami/nvme/ocr_4o/data/text/4o_OCR_error.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    for da in data:
        video_path.append("/opt/dlami/nvme/ocr_4o/da/videomme/data" + "/" + da["videoID"] + ".mp4")

    json_path = "/opt/dlami/nvme/ocr_4o/data/text/video_shot_content.json"

    process_video_info_multithread(video_path, json_path)
