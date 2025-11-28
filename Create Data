import os
import pandas as pd
import cv2
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from tqdm import tqdm
VIDEO_FOLDER_PATH = r"Pharse2/video"
KEY_FRAMES_PATH = "keyframes"
MAP_KEYFRAMES_PATH = "map-keyframes"
OUTPUT_FOLDER = r"./"
THRESHOLD = 0.8
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_image_vector(image):
    """Trích xuất embedding ảnh từ DINOv2"""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Lấy vector đại diện: pooler_output nếu có, không thì lấy mean của last_hidden_state
    if hasattr(outputs, "pooler_output"):
        vec = outputs.pooler_output[0]
    else:
        vec = outputs.last_hidden_state.mean(dim=1)[0]

    return vec.detach().cpu().numpy()

def calculate_cosine(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  
    return dot_product / (norm1 * norm2)


def extract_key_frames(video_path, output_folder, threshold=THRESHOLD):
    video_name = os.path.basename(video_path).split(".")[0]
    video_output_folder = os.path.join(output_folder, KEY_FRAMES_PATH, video_name)
    map_folder = os.path.join(output_folder, MAP_KEYFRAMES_PATH)
    map_dir = os.path.join(map_folder, f"{video_name}.csv")
    if os.path.exists(map_dir):
        print(f"Đã tồn tại thư mục {map_dir}, bỏ qua.")
        return
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không mở được video {video_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    map_keyframes_data = []
    last_frame_vector = None
    last_frame_time = None   # <== lưu lại thời gian frame gần nhất
    saved_frame_count = 1
    current_frame_idx = 0

    # Create a progress bar to track frame processing
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames")
    
    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret:
            frames_to_skip = int(fps)
            current_frame_idx += frames_to_skip
            pbar.update(frames_to_skip)
            continue

        pst_time = current_frame_idx / fps   # thời gian hiện tại (giây)
        current_frame_vector = get_image_vector(frame)

        if last_frame_vector is None:
            is_accepted = True
        else:
            # Nếu cách frame trước >=10s thì auto nhận
            if pst_time - last_frame_time >= 10:
                is_accepted = True
            else:
                similarity = calculate_cosine(last_frame_vector, current_frame_vector)
                is_accepted = similarity < threshold

        if is_accepted:
            frame_filename = os.path.join(video_output_folder, f"{saved_frame_count:03d}.jpg")
            cv2.imwrite(frame_filename, frame)

            map_keyframes_data.append({
                "n": saved_frame_count,
                "pst_time": pst_time,
                "fps": fps,
                "frame_index": current_frame_idx,
            })

            last_frame_vector = current_frame_vector
            last_frame_time = pst_time   # <== cập nhật thời gian
            saved_frame_count += 1

        # skip 1s mỗi lần
        delay_seconds = 1
        frames_to_skip = int(delay_seconds * fps)
        frames_to_skip = frames_to_skip if frames_to_skip > 0 else int(fps)
        current_frame_idx += frames_to_skip
        pbar.update(frames_to_skip)
    
    pbar.close()

    cap.release()

    if saved_frame_count > 0:
        map_df = pd.DataFrame(map_keyframes_data)
        map_folder = os.path.join(output_folder, MAP_KEYFRAMES_PATH)
        os.makedirs(map_folder, exist_ok=True)
        map_df.to_csv(os.path.join(map_folder, f"{video_name}.csv"), index=False)

    print(f"Hoàn thành {video_name}: Trích xuất được {saved_frame_count} keyframes.")


def extract_from_videos(video_folder, output_folder, num_workers = 2):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]

    def process_video(video_file):
        video_path = os.path.join(video_folder, video_file)
        extract_key_frames(video_path, output_folder)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(process_video, video_files),
            total=len(video_files),
            desc="Processing videos",
            unit="video"
        ))

if __name__ == "__main__":
    
    extract_from_videos(VIDEO_FOLDER_PATH, OUTPUT_FOLDER)
