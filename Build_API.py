import os
import time
import json
import csv
import numpy as np
from urllib.parse import unquote

from flask import Flask, request, jsonify, render_template, send_file, abort, send_from_directory
from flask_cors import CORS

# Import engine from your file
from eva02_retrieval_trake import EVA02ImageRetrieval, OUTPUT_FOLDER

# Config (nếu muốn override, hoặc đặt env vars)
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")

# Khởi tạo Flask app (templates folder expected at ./templates)
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Khởi tạo retriever
print("🔁 Initializing EVA02 retriever (this may take a while)...")
retriever = EVA02ImageRetrieval()
print("✅ Retriever ready.")

# Base dir allowed for serving images (keyframes dir from retriever)
IMG_ALLOWED_BASE = os.path.abspath(retriever.keyframes_dir) if getattr(retriever, "keyframes_dir", None) else os.path.abspath("./keyframes")

# Home page (render template)
@app.route("/")
def index():
    stats = retriever.get_stats()
    # stats may contain numpy tuple, convert to JSON-friendly
    return render_template("index.html", stats=stats)

# -----------------------
# API: Search text
# -----------------------
@app.route("/api/search_text", methods=["POST"])
def api_search_text():
    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 100) or 100)
    if not query:
        return jsonify({"error": "empty query"}), 400

    # New: extract object filter parameters
    objects = payload.get("objects", "")  # comma-separated objects; if empty, hold all
    try:
        threshold = float(payload.get("threshold", 0.5))
    except ValueError:
        threshold = 0.5

    start = time.time()
    results = retriever.search_text(
        query,
        top_k=top_k,
        save_to_db=True,
        objects=objects,
        threshold=threshold
    )
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results})

# -----------------------
# API: Search image by existing path
# -----------------------
@app.route("/api/search_image", methods=["POST"])
def api_search_image():
    payload = request.get_json(force=True, silent=True) or {}
    img_path = payload.get("image_path", "")
    top_k = int(payload.get("top_k", 100) or 100)
    if not img_path:
        return jsonify({"error": "image_path required"}), 400
    start = time.time()
    results = retriever.search_image(img_path, top_k=top_k, save_to_db=True)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results})

# -----------------------
# API: Upload image (multipart) then search
# -----------------------
UPLOAD_DIR = os.path.abspath("./_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
    # save
    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time()*1000)}_{safe_name}")
    f.save(save_path)
    top_k = int(request.form.get("top_k", 100) or 100)
    start = time.time()
    results = retriever.search_image(save_path, top_k=top_k, save_to_db=True)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results, "uploaded_path": save_path})

# -----------------------
# API: TRAKE (Temporal Retrieval) with option (default "Closest")
# -----------------------
@app.route("/api/trake", methods=["POST"])
def api_trake():
    payload = request.get_json(force=True, silent=True) or {}
    events = payload.get("events", [])
    top_k = int(payload.get("top_k", 200) or 200)
    candidates = int(payload.get("candidates", 200) or 200)
    # Use default option "Closest" if not provided by the front end.
    option = payload.get("option", "Closest")
    if not events:
        return jsonify({"error": "events required"}), 400

    start = time.time()
    if option == "Closest":
        results = retriever.trake_closest(events, top_k=top_k, candidates= candidates)
    elif option == "Highest":
        results = retriever.trake_highest(events, top_k=top_k, candidates=candidates)
    elif option == "All":
        results = retriever.trake_all(events, top_k=top_k, candidates=candidates)
    else:
        return jsonify({"error": "Invalid option"}), 400
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "result": results})

# -----------------------
# API: Stored queries / reload
# -----------------------
@app.route("/api/stored_queries", methods=["GET"])
def api_stored_queries():
    return jsonify(retriever.get_stored_queries())

@app.route("/api/reload_query/<int:idx>", methods=["GET"])
def api_reload_query(idx):
    res = retriever.reload_query_results(idx)
    return jsonify(res)

# -----------------------
# API: get_frames_of_video
# -----------------------
@app.route("/api/get_frames/<path:video_name>", methods=["GET"])
def api_get_frames(video_name):
    frames = retriever.get_frames_of_video(video_name)
    return jsonify(frames)

# -----------------------
# API: export_results
# Expect JSON:
# {
#   "results": [ { "image_path": "...", "video_name":"..", "frame_idx":"..", "similarity":.. }, ... ],
#   "name": "file_name_no_ext",
#   "const_value": optional integer or null
# }
# -----------------------
@app.route("/api/export_results", methods=["POST"])
def api_export_results():
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    name = payload.get("name", f"exported_{int(time.time())}")
    const_value = payload.get("const_value", None)
    # try convert const_value to int if provided and not blank
    if const_value in ("", None):
        const_value = None
    else:
        try:
            const_value = int(const_value)
        except:
            const_value = None
    retriever.export_results(results, name, const_value)
    csv_path = os.path.join("submission", f"{name}.csv")
    filename = f"{name}.csv"
    return jsonify({
        "status": "ok",
        "csv_path": filename,
        "download_url": f"/download/{filename}"
    })

# -----------------------
# API: export_trake_results
# Expect JSON:
# {
#   "results": [ { "image_path": "...", "video_name":"..", "frame_idx":"..", "similarity":.. }, ... ],
#   "name": "file_name_no_ext",
#   "const_value": optional integer or null
# }
# -----------------------

@app.route("/api/export_trake_results", methods=["POST"])
def export_trake_results():
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    name = payload.get("name", f"trake_{int(time.time())}")

    if not results:
        return jsonify({"status": "error", "msg": "No results provided"}), 400

    try:
        out_dir = os.path.join("submission")
        os.makedirs(out_dir, exist_ok=True)

        retriever.export_trake(results, name)
        csv_path = os.path.join(out_dir, f"{name}.csv")

        return jsonify({
                "status": "ok",
                "csv_path": filename,
                "download_url": f"/download/{filename}"
            })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route("/download/<path:filename>")
def download_file(filename):
    # filename chỉ là "dog-test.csv"
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

# -----------------------
# API: export_trake_results
# Expect JSON:
# {
#   "results": [ { "image_path": "...", "video_name":"..", "frame_idx":"..", "similarity":.. }, ... ],
#   "name": "file_name_no_ext",
#   "const_value": optional integer or null
# }
# ----------------------

# -----------------------
# Helper: Serve image files safely
# -----------------------
@app.route("/image")
def serve_image():
    # Expect query param 'path' = URL-encoded filesystem path (as retriever returns)
    raw = request.args.get("path", "")
    if not raw:
        return abort(400, "path param required")
    # decode
    decoded = unquote(raw)
    # normalize absolute
    abs_path = os.path.abspath(decoded)
    # Security: ensure abs_path is under permitted base (IMG_ALLOWED_BASE)
    try:
        common = os.path.commonpath([abs_path, IMG_ALLOWED_BASE])
    except ValueError:
        return abort(403)
    if common != IMG_ALLOWED_BASE and not abs_path.startswith(IMG_ALLOWED_BASE):
        # not inside allowed folder
        return abort(403)
    if not os.path.exists(abs_path):
        return abort(404)
    # send file
    return send_file(abs_path)

# -----------------------
# API: stats
# -----------------------
@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(retriever.get_stats())

# -----------------------
# API: Get frames range for a video (new feature)
# -----------------------
@app.route("/api/get_frames_range/<path:video_name>/<int:frame_id>/<int:range_val>", methods=["GET"])
def api_get_frames_range(video_name, frame_id, range_val):
    # Get all frames for the video (this function should return a list of dicts with "image_path" key)
    frames = retriever.get_frames_of_video(video_name)
    # Calculate boundaries ensuring we don't exceed available frame indices.
    start_idx = max(0, frame_id - range_val)
    end_idx = min(len(frames), frame_id + range_val + 1)
    frames_range = frames[start_idx:end_idx]
    return jsonify(frames_range)

# -----------------------
# API: Search text with image
# -----------------------
@app.route("/api/search_text_with_image", methods=["POST"])
def api_search_text_with_image():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    # Get image file
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
        
    # Get other parameters
    query = request.form.get("query", "")
    top_k = int(request.form.get("top_k", 100) or 100)
    objects = request.form.get("objects", "")
    threshold = float(request.form.get("threshold", 0.5))

    if not query:
        return jsonify({"error": "empty query"}), 400

    # Save uploaded image temporarily
    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time()*1000)}_{safe_name}")
    f.save(save_path)

    start = time.time()
    
    # First get text search results
    text_results = retriever.search_text(
        query,
        top_k=top_k * 2,  # Get more results for reranking
        save_to_db=False,
        objects=objects,
        threshold=threshold
    )

    if not text_results:
        os.remove(save_path)
        return jsonify({"error": "no text results found"}), 404

    # Extract DINO-v2 features for uploaded image
    query_features = retriever.encode_image(save_path)

    # Rerank using DINO-v2 similarities
    reranked_results = []
    for candidate in text_results:
        # Get candidate image path
        candidate_path = candidate["image_path"]
        
        # Calculate DINO-v2 similarity 
        candidate_features = retriever.get_embedding(candidate_path)
        if candidate_features is not None:
            # Convert to float32 for cosine similarity
            query_features_float = query_features.astype(np.float32)
            candidate_features_float = candidate_features.astype(np.float32)
            
            # Calculate cosine similarity
            dino_similarity = float(np.dot(query_features_float[0], candidate_features_float[0]) / 
                                 (np.linalg.norm(query_features_float[0]) * np.linalg.norm(candidate_features_float[0])))

            # Combine scores (0.6 text, 0.4 DINO-v2)
            combined_score = 0.6 * candidate["similarity"] + 0.4 * dino_similarity
            
            # Update result with combined score
            candidate_copy = candidate.copy()  # Create a copy to avoid modifying original
            candidate_copy["similarity"] = float(combined_score)  # Convert to native Python float
            reranked_results.append(candidate_copy)
    
    # Sort by combined score and take top_k
    reranked_results.sort(key=lambda x: x["similarity"], reverse=True)
    final_results = reranked_results[:top_k]

    elapsed = time.time() - start
    
    # Clean up uploaded file
    try:
        os.remove(save_path)
    except:
        pass

    return jsonify({
        "elapsed": float(elapsed),  # Convert to native Python float
        "results": final_results
    })

@app.route("/api/group_by_video", methods=["POST"])
def api_group_by_video():
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    sort_by = payload.get("sort_by", "frame")          # "frame" | "similarity"
    top_per_video = payload.get("top_per_video", None) # ví dụ 20
    if isinstance(top_per_video, str) and top_per_video.strip():
        try:
            top_per_video = int(top_per_video)
        except:
            top_per_video = None
    grouped = retriever.group_results_by_video(
        results, sort_by=sort_by, top_per_video=top_per_video, with_time=True
    )
    return jsonify(grouped)
    
@app.route("/api/export_submission", methods=["POST"])
def api_export_submission():
    payload = request.get_json(force=True, silent=True) or {}
    video_name = payload.get("video_name")
    frame_id = payload.get("frame_id")
    gap = payload.get("gap")
    const_value = payload.get("const_value")
    name = payload.get("name", f"submission_{int(time.time())}")

    if video_name is None or frame_id is None or gap is None:
        return jsonify({"status": "error", "msg": "video_name, frame_id, gap are required"}), 400

    try:
        csv_path = retriever.export_submission(video_name, frame_id, gap, const_value, name)
        return jsonify({"status": "ok", "csv_path": csv_path})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500
# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    print(f"🌐 Starting server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)  # Set debug=False in production
