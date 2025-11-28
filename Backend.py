# eva02_retrieval.py (updated with TRAKE temporal retrieval)
import os
import re
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import torchvision
import torchvision.transforms as T
from collections import defaultdict
import csv
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch.nn.functional as F

# Configuration (can be overridden with environment variables)
EMBEDDING_DIR =  r"./eva02_large_patch14_clip_224.merged2b_s4b_b131k"
KEYFRAMES_DIR =  r"./keyframes"
DB_PATH =  r"faiss_db.pkl"
MODEL_NAME = "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
MODEL_PRETRAINED = "laion2b_s4b_b131k"
OUTPUT_FOLDER =  "./submission"
MAP_KEYFRAME_PATH =  r"./map-keyframes"
MEDIA_INFO_PATH =  r"./media-info"
DEFAULT_TOP_K = 100

GAP_KEYFRAME = 15 # Video thường 25 -> 30fps, để 15 cho chắc

# COCO categories (index 0 is reserved, so valid labels start at index 1)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Utilities
def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def _safe_basename(path: str) -> str:
    try:
        return os.path.basename(path)
    except Exception:
        return str(path)

def _parse_frame_number_from_filename(filename: str) -> Optional[int]:
    name, _ = os.path.splitext(filename)
    m = re.search(r'(\d{1,6})$', name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m2 = re.search(r'(\d+)', name)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None

def _detect_map_columns(df: pd.DataFrame) -> Tuple[str, str]:
    candidates_n = ["n", "frame_number", "frame_no", "frame"]
    candidates_idx = ["frame_idx", "frame_index", "index", "idx"]
    n_col = None; i_col = None
    for c in candidates_n:
        if c in df.columns:
            n_col = c; break
    for c in candidates_idx:
        if c in df.columns:
            i_col = c; break
    if n_col is None or i_col is None:
        raise KeyError(f"Mapping CSV missing expected columns. Found: {list(df.columns)}")
    return n_col, i_col

# Vector DB
class VectorDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.data = {
            'queries': [],
            'embeddings': [],
            'image_paths': [],
            'video_names': [],
            'frame_indices': [],
            'similarities': [],
            'timestamps': [],
            'metadata': {}
        }
        self.load_database()

    def load_database(self):
        try:
            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)

            # 🔑 Đảm bảo embeddings luôn là list
            if isinstance(self.data["embeddings"], np.ndarray):
                self.data["embeddings"] = list(self.data["embeddings"])

            if self.data["embeddings"]:
                all_embs = np.vstack(self.data["embeddings"]).astype(np.float32)
                d = all_embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(all_embs)
            else:
                self.index = None

            print(f"Loaded DB with {len(self.data['queries'])} queries")

        except Exception as e:
            print(f"Failed to load DB: {e}. Reinitializing...")
            self.data["embeddings"] = []   # reset về list

    def save_database(self):
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"💾 Saved DB -> {self.db_path}")
        except Exception as e:
            print(f"❌ DB save error: {e}")

    def add_query_results(self, query, results, embeddings):
        # Append the query string.
        self.data["queries"].append(query)
        # Append the embeddings (as 32-bit floats)
        self.data["embeddings"].append(embeddings.astype(np.float32))
        # Extract per-result information into lists.
        image_paths = [r["image_path"] for r in results]
        video_names = [r["video_name"] for r in results]
        frame_indices = [r["frame_idx"] for r in results]
        similarities = [r["similarity"] for r in results]
        # Append the results information.
        self.data["image_paths"].append(image_paths)
        self.data["video_names"].append(video_names)
        self.data["frame_indices"].append(frame_indices)
        self.data["similarities"].append(similarities)
        self.data["timestamps"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
     
        # Update FAISS index (using inner product, as before)
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)


    def get_all_queries(self) -> List[str]:
        return self.data['queries']

    def get_query_results(self, query_idx: int) -> Optional[Dict[str, Any]]:
        if 0 <= query_idx < len(self.data['queries']):
            return {
                'query': self.data['queries'][query_idx],
                'embeddings': self.data['embeddings'][query_idx],
                'image_paths': self.data['image_paths'][query_idx],
                'video_names': self.data['video_names'][query_idx],
                'frame_indices': self.data['frame_indices'][query_idx],
                'similarities': self.data['similarities'][query_idx],
                'timestamp': self.data['timestamps'][query_idx],
            }
        return None

    def get_combined_embeddings(self, query_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, List[Dict]]:
        if query_indices is None:
            query_indices = list(range(len(self.data['queries'])))
        combined_embeddings = []; combined_meta = []
        for idx in query_indices:
            if 0 <= idx < len(self.data['queries']):
                embs = self.data['embeddings'][idx]
                combined_embeddings.append(embs)
                for i in range(len(embs)):
                    combined_meta.append({
                        'query_idx': idx, 'query': self.data['queries'][idx],
                        'image_path': self.data['image_paths'][idx][i],
                        'video_name': self.data['video_names'][idx][i],
                        'frame_idx': self.data['frame_indices'][idx][i],
                        'similarity': self.data['similarities'][idx][i],
                        'timestamp': self.data['timestamps'][idx],
                    })
        if combined_embeddings:
            return np.concatenate(combined_embeddings, axis=0), combined_meta
        return np.array([]), []

    def export_to_numpy(self, output_path: str, query_indices: Optional[List[int]] = None):
        embs, meta = self.get_combined_embeddings(query_indices)
        if len(embs) == 0:
            print("❌ No embeddings to export"); return
        _ensure_dir(os.path.dirname(output_path))
        np.save(f"{output_path}_embeddings.npy", embs)
        with open(f"{output_path}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"📤 Exported: {len(embs)} embeddings")

    def show_stats(self):
        total_queries = len(self.data['queries'])
        total_images = sum(len(p) for p in self.data['image_paths'])
        print("\n📊 Vector DB Stats")
        print(f"  Total queries: {total_queries}")
        print(f"  Total images:  {total_images}")
        if total_queries:
            print("  Recent queries:")
            for i, (q, ts) in enumerate(zip(self.data['queries'][-5:], self.data['timestamps'][-5:])):
                print(f"   - [{total_queries-5+i}] {q} ({ts})")

# EVA02 Retrieval Engine with TRAKE
class EVA02ImageRetrieval:
    def __init__(self,
                 embedding_dir: str = EMBEDDING_DIR,
                 keyframes_dir: str = KEYFRAMES_DIR,
                 db_path: str = DB_PATH):

        self.embedding_dir = embedding_dir
        self.keyframes_dir = keyframes_dir

        # Vector DB
        self.vector_db = VectorDatabase(db_path)

        # Load model (CLIP-based)
        print("🔄 Loading EVA02 model... (this may take a while)")
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=MODEL_PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        # Load embeddings + mappings
        print("🔄 Loading embeddings...")
        self.embeddings, self.image_paths, self.video_names = self._load_all_embeddings()
        print(f"✅ Loaded {len(self.image_paths)} images from {len(set(self.video_names))} videos")

        # ------------------------------
        print("🔄 Loading Faster RCNN detection model...")
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detection_model.to(self.device).eval()
        self.detection_transform = T.Compose([T.ToTensor()])
        print("✅ Faster RCNN ready.")
        # ------------------------------
        
        self.processor_reranker = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.reranker = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(self.device)


        # ------------------------------
        print("🔄 Loading Faster RCNN detection model...")
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detection_model.to(self.device).eval()
        self.detection_transform = T.Compose([T.ToTensor()])
        print("✅ Faster RCNN ready.")
        # ------------------------------

    def _get_video_folder_path(self, video_name: str) -> str:
        return os.path.join(self.keyframes_dir, video_name)

    def _load_all_embeddings(self) -> Tuple[np.ndarray, List[str], List[str]]:
        all_embs = []; all_img_paths = []; all_vids = []
        if not os.path.exists(self.embedding_dir):
            print(f"⚠️ Embedding dir not found: {self.embedding_dir}"); return np.array([]), [], []
        npy_files = sorted([f for f in os.listdir(self.embedding_dir) if f.endswith(".npy")])
        for npy in npy_files:
            video_name = os.path.splitext(npy)[0]
            emb_path = os.path.join(self.embedding_dir, npy)
            try:
                embs = np.load(emb_path)
            except Exception as e:
                print(f"⚠️ Failed to load {emb_path}: {e}"); continue
            all_embs.append(embs)
            vid_folder = self._get_video_folder_path(video_name)
            if os.path.exists(vid_folder):
                frames = sorted([f for f in os.listdir(vid_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                for i, f in enumerate(frames):
                    if i < len(embs):
                        all_img_paths.append(os.path.join(vid_folder, f))
                        all_vids.append(video_name)
        if all_embs:
            return np.concatenate(all_embs, axis=0).astype(np.float32), all_img_paths, all_vids
        return np.array([]), [], []

    # Encoders
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            toks = self.tokenizer([text]).to(self.device)
            feats = self.model.encode_text(toks)
            feats = F.normalize(feats, dim=-1)
            return feats.cpu().numpy().astype(np.float32)

    def encode_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = self.preprocess_val(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(img)
            feats = F.normalize(feats, dim=-1)
            return feats.cpu().numpy().astype(np.float32)
    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Get embedding from EMBEDDING_DIR based on image path.
        Args:
            image_path (str): Path to the image file
        Returns:
            np.ndarray: Image embedding vector with shape (1, dim)
        """
        try:
            # Extract video name and frame number
            video_name = os.path.basename(os.path.dirname(image_path))
            frame_str = os.path.splitext(os.path.basename(image_path))[0]
            frame_idx = _parse_frame_number_from_filename(frame_str)

            if frame_idx is None:
                print(f"⚠️ Cannot parse frame number from: {image_path}")
                return None

            # Load embeddings for the video
            emb_path = os.path.join(self.embedding_dir, f"{video_name}.npy")
            if not os.path.exists(emb_path):
                print(f"⚠️ Embedding file not found: {emb_path}")
                return None

            # Load embeddings and get specific frame embedding
            embs = np.load(emb_path)
            if frame_idx >= len(embs):
                print(f"⚠️ Frame index {frame_idx} out of bounds for video {video_name}")
                return None

            # Return embedding as 2D array with shape (1, dim)
            return embs[frame_idx:frame_idx+1].astype(np.float32)

        except Exception as e:
            print(f"❌ Error getting embedding for {image_path}: {e}")
            return None
            
    @staticmethod
    def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    def _topk_from_sim(self, sims: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(sims)[::-1][:top_k]
        return idx, sims[idx]

    def _results_from_indices(self, indices: np.ndarray, scores: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        results = []; sel_embs = []
        for i, idx in enumerate(indices):
            img_path = self.image_paths[idx]
            video_name = self.video_names[idx]
            frame_str = os.path.splitext(os.path.basename(img_path))[0]
            media_info_file = os.path.join(MEDIA_INFO_PATH, f"{video_name}.json")
            if os.path.exists(media_info_file):
                try:
                    with open(media_info_file, "r", encoding="utf-8") as f:
                        media_info = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load media info for {video_name}: {e}")
                    media_info = {}
            
            
            results.append({
                "image_path": img_path, "video_name": video_name, "similarity": float(scores[i]), "frame_idx": frame_str, "watch_url": media_info.get("watch_url", "") if 'media_info' in locals() else ""
            })
            sel_embs.append(self.embeddings[idx])
        return results, np.asarray(sel_embs, dtype=np.float32)

    
    def detect_objects(self, image_path: str, threshold: float = 0.5) -> List[str]:
        """
        Run object detection on the given image and return a list of detected object names
        with confidence >= threshold.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []
        tensor_img = self.detection_transform(img).to(self.device)
        with torch.no_grad():
            preds = self.detection_model([tensor_img])
        pred_scores = preds[0].get("scores", [])
        pred_labels = preds[0].get("labels", [])
        detected = []
        for score, label in zip(pred_scores, pred_labels):
            if score >= threshold:
                label_str = COCO_INSTANCE_CATEGORY_NAMES[label]
                detected.append(label_str.lower())
        return detected    

    # ===== NEW: group results by video for easier viewing =====
    def group_results_by_video(
        self,
        results: List[Dict],
        sort_by: str = "frame",           # "frame" | "similarity"
        top_per_video: Optional[int] = 50,
        with_time: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Gom kết quả theo từng video.
        - sort_by="frame": sắp theo frame trong video (frame_idx_video tăng dần)
        - sort_by="similarity": sắp theo similarity giảm dần
        - top_per_video: nếu đặt số, chỉ lấy top N của mỗi video
        - with_time: nếu True, bổ sung time_str nếu có fps và mapping
        Trả về: dict { video_name: [list result dict đã bổ sung] }
        """
        from collections import defaultdict
        grouped: Dict[str, List[Dict]] = defaultdict(list)

        for r in results or []:
            v = r.get("video_name", "")
            item = dict(r)  # shallow copy để không đụng vào bản gốc

            # Đảm bảo có frame_idx_video (số frame theo video) & time_str
            if item.get("frame_idx_video") is None:
                # parse số n từ tên keyframe
                n = _parse_frame_number_from_filename(str(item.get("frame_idx", "")))
                if n is not None:
                    idx = self.load_frame_idx_from_map(v, n)
                    if idx is not None:
                        item["frame_idx_video"] = idx
                        if with_time:
                            fps = self.load_fps_from_map(v)
                            if fps:
                                item["time_str"] = self.frame_to_timestr(idx, fps)

            # Chuẩn hoá similarity về float
            if "similarity" in item and item["similarity"] is not None:
                try:
                    item["similarity"] = float(item["similarity"])
                except Exception:
                    pass

            grouped[v].append(item)

        # Sắp xếp và cắt top cho từng video
        for v, lst in grouped.items():
            if sort_by == "similarity":
                lst.sort(key=lambda x: (x.get("similarity") is None, -(x.get("similarity") or 0.0)))
            else:  # "frame"
                lst.sort(key=lambda x: (x.get("frame_idx_video") is None, x.get("frame_idx_video") or 0))
            if isinstance(top_per_video, int) and top_per_video > 0:
                grouped[v] = lst[:top_per_video]

        return dict(grouped)

    def print_grouped_results(self, grouped: Dict[str, List[Dict]], show: int = 10):
        """
        In gọn kết quả đã group để quan sát nhanh trên console.
        """
        for vid, items in grouped.items():
            print(f"\n▶ {vid}  (items: {len(items)})")
            for i, r in enumerate(items[:max(0, show)]):
                fidx = r.get("frame_idx_video", r.get("frame_idx"))
                sim  = r.get("similarity")
                tstr = r.get("time_str", "")
                name = _safe_basename(r.get("image_path", ""))
                sim_txt = f"{sim:.4f}" if isinstance(sim, (int, float)) else "NA"
                print(f"  - #{fidx}  sim={sim_txt}  {tstr}  {name}")


    def search_text_all(self, query: str, score: float = 0.2, save_to_db: bool = True) -> List[Dict]:
        if len(self.embeddings) == 0:
            print("❌ No embeddings loaded")
            return []
        text_emb = self.encode_text(query)
        sims = self.cosine_similarity_numpy(text_emb, self.embeddings)[0]
        idx = np.where(sims >= score)[0]
        scores = sims[idx]
        if len(scores) == 0:
            return []
        sorted_indices = np.argsort(scores)[::-1]
        idx = idx[sorted_indices]
        scores = scores[sorted_indices]
        results, sel_embs = self._results_from_indices(idx, scores)
        if save_to_db and results:
            self.vector_db.add_query_results(query, results, np.asarray(sel_embs, dtype=np.float32))
        return results

    def load_fps_from_map(self, video_name):
        # Giả sử file tên: L29_V021.csv
        map_file = os.path.join(MAP_KEYFRAME_PATH, f"{video_name}.csv")
        if not os.path.exists(map_file):
            return None
        
        with open(map_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return float(row["fps"])  # tất cả dòng đều có cùng fps
        return None

    def load_frame_idx_from_map(self, video_name, n):
        # Giả sử file tên: L29_V021.csv
        map_file = os.path.join(MAP_KEYFRAME_PATH, f"{video_name}.csv")
        if not os.path.exists(map_file):
            return None
        
        df = pd.read_csv(map_file)
        try:
            n_col, i_col = _detect_map_columns(df)
        except KeyError as e:
            print(f"Mapping CSV error for {video_name}: {e}")
            return None
        result = df.loc[df[n_col] == n, i_col]
        if not result.empty:
            return int(result.iloc[0])
        return None

    def frame_to_timestr(self, frame_idx, fps):
        sec = frame_idx / fps
        minutes = int(sec // 60)
        seconds = int(sec % 60)
        return f"{minutes}:{seconds:02d}"

    # Basic searches
    def search_text(self, query: str, top_k: int = DEFAULT_TOP_K, save_to_db: bool = True,
                    objects: str = "", threshold: float = 0.5) -> List[Dict]:
        if len(self.embeddings) == 0:
            print("❌ No embeddings loaded")
            return []
        text_emb = self.encode_text(query)
        sims = self.cosine_similarity_numpy(text_emb, self.embeddings)[0]
        idx, scores = self._topk_from_sim(sims, top_k)
        results, sel_embs = self._results_from_indices(idx, scores)

        if objects:
            desired = [o.strip().lower() for o in objects.split(",") if o.strip()]
            filtered_results = []
            filtered_embs = []
            for res, emb in zip(results, sel_embs):
                detected_objs = self.detect_objects(res["image_path"], abs(threshold))
                # Check if at least one desired object is present in detected objects.
                if threshold >= 0:
                    if any(obj in detected_objs for obj in desired):
                        filtered_results.append(res)
                        filtered_embs.append(emb)
                else:
                    if all(obj not in detected_objs for obj in desired):
                        filtered_results.append(res)
                        filtered_embs.append(emb)
            results = filtered_results
        else:
            filtered_embs = sel_embs
        if save_to_db and results:
            self.vector_db.add_query_results(query, results, np.asarray(filtered_embs, dtype=np.float32))

        for res in results:
            fps = self.load_fps_from_map(res["video_name"])
            if fps:
                res["fps"] = fps
                res["frame_idx_video"] = self.load_frame_idx_from_map(res['video_name'], int(res['frame_idx']))
                res["time_str"] = self.frame_to_timestr(res["frame_idx_video"], fps)

        return results

    def search_image(self, image_path: str, top_k: int = DEFAULT_TOP_K, save_to_db: bool = True) -> List[Dict]:
        if len(self.embeddings) == 0:
            print("❌ No embeddings loaded"); return []
        img_emb = self.encode_image(image_path)
        sims = self.cosine_similarity_numpy(img_emb, self.embeddings)[0]
        idx, scores = self._topk_from_sim(sims, top_k)
        query_tag = f"[IMAGE] {_safe_basename(image_path)}"
        results, sel_embs = self._results_from_indices(idx, scores)
        if save_to_db and results:
            self.vector_db.add_query_results(query_tag, results, sel_embs)
        return results

    # Get all frames of a video
    def get_frames_of_video(self, video_name: str) -> List[Dict]:
        video_folder = self._get_video_folder_path(video_name)
        if not os.path.exists(video_folder):
            return []
        frames = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        out = []
        for f in frames:
            out.append({
                "image_path": os.path.join(video_folder, f),
                "video_name": video_name,
                "frame_idx": os.path.splitext(f)[0],
                "similarity": None
            })
        return out

    def rerank(self, ev, list_rerank):
        Threshold = 0.4
        list_final = []
        for r in list_rerank:
            raw_image = Image.open(r['image_path']).convert('RGB')
            text = ev
            inputs = self.processor_reranker(raw_image, text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                itm_score = self.reranker(**inputs)[0]
                prob = F.softmax(itm_score, dim=-1)[0, 1]  # xác suất match
                if prob >= Threshold:
                    r['similarity'] = float(prob)
                    list_final.append(r)
        return list_final        
    # TRAKE: Temporal Retrieval and Alignment of Key Events
    def trake_closest(self, events: List[str], top_k: int = 200, candidates: int = 200) -> Optional[Dict[str, Any]]:
        if not events:
            return None
        # gather candidates per event
        all_event_candidates = []
        num = 0
        special_event = 0
        for ev in events:
            num += 1
            if ev[-1] == '*':
                special_event = num
                all_event_candidates.append(self.rerank(ev, self.search_text(ev[:-1], top_k=500, save_to_db=True)))
            else:
                all_event_candidates.append(self.search_text(ev, top_k=top_k, save_to_db=True))

        # Build set of candidate videos (videos that appear at least in one event top-k)
        candidate_videos = set()
        if num == 0:
            for r in all_event_candidates[0]:
                candidate_videos.add(r['video_name'])
        else:
            for r in all_event_candidates[special_event-1]:
                candidate_videos.add(r['video_name'])

        def parse_frame_int(s):
            try: return int(s)
            except:
                num = _parse_frame_number_from_filename(str(s))
                return num if num is not None else 0

        best_solution = None; best_score = -1e9

        # Try each candidate video
        list_solutions = []
            
        for vid in candidate_videos:
            # For each event, get sorted list of candidates for this video by similarity desc
            check_empty = False
            vid_candidates = []
            lag = 0
            for ev_res in all_event_candidates:
                num = 0
                lag += 1 
                for r in ev_res:
                    if r['video_name'] == vid:
                        fi = parse_frame_int(r['frame_idx'])
                        if fi is None:
                            continue
                        vid_candidates.append({'frame': fi, 'image_path': r['image_path'], 'sim': r['similarity'], 'frame_str': r['frame_idx'], 'lag': lag})
                        num += 1
                if num == 0:
                    check_empty = True
                    break
            if check_empty:
                continue
            vid_candidates.sort(key=lambda x: (x['frame'], -x['lag']))
            # filter same frame, keep highest similarity
            filtered = dict()
            for cand in vid_candidates:
                f = cand['frame']
                if filtered.get(f, None) is None or filtered[f]['sim'] < cand['sim']:
                    filtered[f] = cand
            vid_candidates = list(filtered.values())
            vid_candidates.sort(key=lambda x: x['frame'])

            # If any event has zero candidates in this video, skip
            D = dict()
            trace = defaultdict(dict)
            for cand in vid_candidates:
                lag = cand['lag']
                if lag == 1:
                    D[lag] = cand                
                else:
                    trace[lag-1][cand['frame']] = D.get(lag-1, None)
                    D[lag] = cand
            list_temp = []
            for cand in vid_candidates:
                lag = cand['lag']
                if lag == len(events) and trace[lag-1].get(cand['frame'], None) is not None:
                    sequence = [None] * len(events)
                    score = 0.0
                    while lag > 0 :
                        sequence[lag-1] = cand
                        score += cand['sim']
                        cand = trace[lag-1].get(cand['frame'], None)
                        if cand is None:
                            break
                        lag -= 1
                    list_temp.append((sequence, score))
            list_temp.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(5, len(list_temp))):
                list_solutions.append(list_temp[i])
            list_temp.clear()
        list_solutions.sort(key=lambda x: x[1], reverse=True)
        
        list_solutions = list_solutions[:min(candidates, len(list_solutions))]
        if len(list_solutions) == 0:
            print("❌ TRAKE: No valid solutions found")
            return None

        # prepare output
        get_ans = []
        for solution, score in list_solutions:
            ck = True
            for s in solution:
                if s is None:
                    ck = False
                    break
            if not ck:
                continue
            frame_ids = [s['frame_str'] for s in solution]
            video_name = solution[0]['image_path'].split(os.sep)[-2]  # parent folder name
            media_info_file = os.path.join(MEDIA_INFO_PATH, f"{video_name}.json")
            if os.path.exists(media_info_file):
                try:
                    with open(media_info_file, "r", encoding="utf-8") as f:
                        media_info = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load media info for {video_name}: {e}")
                    media_info = {}
            details = [{'event': events[i], 'frame': solution[i]['frame_str'], 'sim': solution[i]['sim'], 'path': solution[i]['image_path']} for i in range(len(events))]
            get_ans.append({'video_name': video_name, 'frame_ids': frame_ids, 'details': details, 'score': score, 'watch_url': media_info.get("watch_url", "") if 'media_info' in locals() else ""})
        # prepare output
        return get_ans
    
        
    def trake_highest(self, events: List[str], top_k: int = 200, candidates: int = 200, option: int =0) -> Optional[Dict[str, Any]]:
        if not events:
            return None
        # gather candidates per event
        all_event_candidates = []
        special_event = 0
        num = 0
        for ev in events:
            num += 1
            if option == 0:
                if ev[-1] == '*':
                    special_event = num
                    all_event_candidates.append(self.rerank(ev, self.search_text(ev[:-1], top_k=500, save_to_db=True)))
                else:                    
                    all_event_candidates.append(self.search_text(ev, top_k=top_k, save_to_db=True))
            else:
                all_event_candidates.append(self.search_text_all(ev, score=0.2, save_to_db=False))
        # Build set of candidate videos (videos that appear at least in one event top-k)
        candidate_videos = set()

        if num == 0:
            for r in all_event_candidates[0]:
                candidate_videos.add(r['video_name'])
        else:
            for res in all_event_candidates[special_event-1]:
                candidate_videos.add(res['video_name'])
        
        def parse_frame_int(s):
            try: return int(s)
            except:
                num = _parse_frame_number_from_filename(str(s))
                return num if num is not None else 0

        best_solution = None; best_score = -1e9

        # Try each candidate video
        check_empty = False
        list_solutions = []
            
        for vid in candidate_videos:
            # For each event, get sorted list of candidates for this video by similarity desc
            check_empty = False
            vid_candidates = []
            lag = 0
            for ev_res in all_event_candidates:
                num = 0
                lag += 1 
                for r in ev_res:
                    if r['video_name'] == vid:
                        fi = parse_frame_int(r['frame_idx'])
                        if fi is None:
                            continue
                        vid_candidates.append({'frame': fi, 'image_path': r['image_path'], 'sim': r['similarity'], 'frame_str': r['frame_idx'], 'lag': lag})
                        num += 1
                if num == 0:
                    check_empty = True
                    break
            if check_empty:
                continue
            vid_candidates.sort(key=lambda x: x['frame'])

            # If any event has zero candidates in this video, skip
            D = dict()
            trace = defaultdict(dict)
            score = defaultdict(float)
            num = 0
            i = 0
            while i < len(vid_candidates):
                cand = vid_candidates[i]
                lag = cand['lag']
                frame = cand['frame']

                # Gom nhóm cand có cùng frame liên tiếp
                group = [cand]
                j = i + 1
                while j < len(vid_candidates) and vid_candidates[j]['frame'] == frame:
                    group.append(vid_candidates[j])
                    j += 1

                previ_state = D.copy()
                previ_score = score.copy()
                for g in group:
                    lag = g['lag']
                    frame = g['frame']
                    if lag == 1:
                        if previ_score.get(lag, None) is None or previ_score[lag] < g['sim']:
                            D[lag] = g
                            score[lag] = g['sim']
                    else:
                        prev_cand = previ_state.get(lag-1, None)
                        if prev_cand is not None:
                            new_score = previ_score[lag-1] + g['sim']
                            if D.get(lag, None) is None or score[lag] < new_score:
                                D[lag] = g
                                score[lag] = new_score
                                trace[lag-1][frame] = prev_cand
                i = j
            list_temp = []
            
            for cand in vid_candidates:
                lag = cand['lag']
                if lag == len(events) and trace[lag-1].get(cand['frame'], None) is not None:
                    sequence = [None] * len(events)
                    score = 0.0
                    while lag > 0 :
                        sequence[lag-1] = cand
                        score += cand['sim']
                        cand = trace[lag-1].get(cand['frame'], None)
                        if cand is None:
                            break
                        lag -= 1
                    list_temp.append((sequence, score))
            list_temp.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(10, len(list_temp))):
                list_solutions.append(list_temp[i])
            list_temp.clear()
            D.clear()
            trace.clear()
        print(len(list_solutions))
        list_solutions.sort(key=lambda x: x[1], reverse=True)
        list_solutions = list_solutions[:min(candidates, len(list_solutions))]
        if len(list_solutions) == 0:
            print(" TRAKE: No valid solutions found")
            return None

        # prepare output
        get_ans = []
        for solution, score in list_solutions:
            ck = True
            for s in solution:
                if s is None:
                    ck = False
                    break
            if not ck:
                continue
            frame_ids = [s['frame_str'] for s in solution]
            video_name = solution[0]['image_path'].split(os.sep)[-2]  # parent folder name
            media_info_file = os.path.join(MEDIA_INFO_PATH, f"{video_name}.json")
            if os.path.exists(media_info_file):
                try:
                    with open(media_info_file, "r", encoding="utf-8") as f:
                        media_info = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load media info for {video_name}: {e}")
                    media_info = {}
            details = [{'event': events[i], 'frame': solution[i]['frame_str'], 'sim': solution[i]['sim'], 'path': solution[i]['image_path']} for i in range(len(events))]
            get_ans.append({'video_name': video_name, 'frame_ids': frame_ids, 'details': details, 'watch_url': media_info.get("watch_url", "") if 'media_info' in locals() else ""})
        # prepare output
        return get_ans
    
    # Export results (list of dicts) to submission CSV (using mapping CSVs)
    def export_results(self, results: List[Dict], output_name: str, const_value: Optional[int] = None):
        if not results:
            print("❌ No results to export"); return
        _ensure_dir(OUTPUT_FOLDER)
        out_path = os.path.join(OUTPUT_FOLDER, f"{output_name}.csv")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                vid, n = os.path.basename(os.path.dirname(r['image_path'])), _parse_frame_number_from_filename(os.path.basename(r['image_path']))
                if n is None:
                    print(f"⚠️ Skip cannot parse frame: {r['image_path']}"); continue
                map_csv = os.path.join(MAP_KEYFRAME_PATH, f"{vid}.csv")
                if not os.path.exists(map_csv):
                    print(f"⚠️ Mapping CSV not found for {vid}: {map_csv}"); continue
                try:
                    df = pd.read_csv(map_csv)
                    n_col, idx_col = _detect_map_columns(df)
                    sub = df.loc[df[n_col] == n, idx_col]
                    if sub.empty:
                        print(f"⚠️ frame index not found for {vid} n={n}"); continue
                    frame_index = int(sub.iloc[0])
                except Exception as e:
                    print(f"⚠️ Error reading mapping for {vid}: {e}"); continue

                if const_value is None:
                    f.write(f"{vid},{frame_index}\n")
                else:
                    f.write(f"{vid},{frame_index},{const_value}\n")
        print(f"📄 Exported submission -> {out_path}")

    def export_trake(self, trake_payload: List[Dict], output_name: str):
        print(" Exporting TRAKE results...")
        if not trake_payload:
            print("❌ No TRAKE results to export")
            return

        _ensure_dir(OUTPUT_FOLDER)
        out_path = os.path.join(OUTPUT_FOLDER, f"{output_name}.csv")

        with open(out_path, "w") as f:
            num = -1
            for r in trake_payload:
                vid = r.get("video_name")
                keyframes = r.get("frame_idx", [])
                size = r.get("event",3)
                if num == -1:
                    num = size
                
                if not vid or not keyframes:
                    print(f"Skip invalid entry: {r}")
                    continue

                map_csv = os.path.join(MAP_KEYFRAME_PATH, f"{vid}.csv")
                if not os.path.exists(map_csv):
                    print(f"Mapping CSV not found for {vid}: {map_csv}")
                    continue

    def export_trake(self, trake_payload: List[Dict], output_name: str):
        """
        Export TRAKE results thành submission CSV.
        Mỗi đáp án gồm `size` events -> gộp thành 1 dòng:
        vid, frame_1, frame_2, ..., frame_size
        """
        if not trake_payload:
            print("❌ No TRAKE results to export")
            return

        _ensure_dir(OUTPUT_FOLDER)
        out_path = os.path.join(OUTPUT_FOLDER, f"{output_name}.csv")

        with open(out_path, "w", encoding="utf-8") as f:
            num = -1   # số event cho mỗi đáp án
            buffer = []

            for r in trake_payload:
                vid = r.get("video_name")
                kf = r.get("frame_idx")        # chỉ 1 keyframe duy nhất
                size = r.get("event", 3)

                if num == -1:
                    num = size

                if not vid or kf is None:
                    print(f"⚠️ Skip invalid entry: {r}")
                    continue

                map_csv = os.path.join(MAP_KEYFRAME_PATH, f"{vid}.csv")
                if not os.path.exists(map_csv):
                    print(f"⚠️ Mapping CSV not found for {vid}: {map_csv}")
                    continue

                try:
                    df = pd.read_csv(map_csv)
                    n_col, idx_col = _detect_map_columns(df)

                    sub = df.loc[df[n_col] == int(kf), idx_col]
                    if sub.empty:
                        print(f"⚠️ keyframe {kf} not found in {vid}")
                        continue
                    frame_id = int(sub.iloc[0])

                    buffer.append((vid, frame_id))

                    # đủ một nhóm event -> flush ra file
                    if len(buffer) == num:
                        vid0 = buffer[0][0]
                        merged = [str(fid) for _, fid in buffer]
                        f.write(f"{vid0}," + ",".join(merged) + "\n")
                        buffer = []

                except Exception as e:
                    print(f"⚠️ Error reading mapping for {vid}: {e}")
                    continue

        print(f"📄 Exported TRAKE submission -> {out_path}")
    
    def export_submission(self, video_name, frame_id, gap, const_value=None, name=None):
        """
        Sinh file submission CSV từ video_name, frame_id, gap.
        Tạo 20 dòng theo mẫu: center, ±gap, ±2gap,...
        """
        if video_name is None or frame_id is None or gap is None:
            raise ValueError("video_name, frame_id, gap are required")

        frame_id = int(frame_id)
        gap = int(gap)
        if gap <= 0:
            raise ValueError("gap must be > 0")

        if name is None:
            name = f"submission_{int(time.time())}"

        out_dir = os.path.join("submission")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{name}.csv")

        rows = []
        offset = 0
        while len(rows) < 20:
            if offset == 0:
                rows.append([video_name, frame_id, const_value] if const_value is not None else [video_name, frame_id])
            else:
                # +offset
                plus_id = frame_id + offset
                rows.append([video_name, plus_id, const_value] if const_value is not None else [video_name, plus_id])
                # -offset
                minus_id = frame_id - offset
                if minus_id >= 0:
                    rows.append([video_name, minus_id, const_value] if const_value is not None else [video_name, minus_id])
            offset += gap

        rows = rows[:20]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # Conveniences
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_images": len(self.image_paths),
            "total_videos": len(set(self.video_names)),
            "embedding_shape": tuple(self.embeddings.shape) if isinstance(self.embeddings, np.ndarray) else None,
        }

    def get_stored_queries(self) -> List[str]:
        return self.vector_db.get_all_queries()

    def reload_query_results(self, query_idx: int) -> List[Dict]:
        data = self.vector_db.get_query_results(query_idx)
        if not data: return []
        results = []
        for i in range(len(data['image_paths'])):
            results.append({
                "image_path": data['image_paths'][i],
                "video_name": data['video_names'][i],
                "similarity": float(data['similarities'][i]),
                "frame_idx": data['frame_indices'][i],
            })
        return results

    def export_database(self, output_path: str = "./exported_vectors", query_indices: Optional[List[int]] = None):
        self.vector_db.export_to_numpy(output_path, query_indices)

    def show_database_stats(self):
        self.vector_db.show_stats()

def main():
    # get faiss db
    engine = EVA02ImageRetrieval()
    engine.show_database_stats()
    print(engine.get_stats())

if __name__ == "__main__":
    query = [
        "A player is shooting a basketball",
        "A player is dunking a basketball",
        "A player is dribbling a basketball",
    ]
    engine = EVA02ImageRetrieval()
    engine.show_database_stats()
    print(engine.get_stats())

    results = engine.trake_highest(query, top_k=200, candidates=200, option=0)
    print(results)

