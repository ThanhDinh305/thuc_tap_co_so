"""
app.py — Flask AI Service
Wraps the YOLOv8 fruit recognition model as a REST API.
Port: 5001
"""

import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_service import load_resources, predict_image

app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://localhost:5173"])

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "fruit-ai"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:
      - multipart/form-data with field 'image'
      - OR application/json with field 'image_base64' (base64 string)
    Returns JSON prediction result.
    """
    try:
        image_bytes = None

        # ── Case 1: File upload ──────────────────────────────────────
        if "image" in request.files:
            file = request.files["image"]
            ext  = Path(file.filename).suffix.lower() if file.filename else ".jpg"
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({"success": False, "message": f"Unsupported file type: {ext}"}), 400
            image_bytes = file.read()

        # ── Case 2: Base64 JSON ──────────────────────────────────────
        elif request.is_json:
            data    = request.get_json()
            b64data = data.get("image_base64", "")
            if not b64data:
                return jsonify({"success": False, "message": "No image data provided."}), 400
            import base64
            # Strip data URI prefix if present (data:image/jpeg;base64,...)
            if "," in b64data:
                b64data = b64data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64data)

        else:
            return jsonify({"success": False, "message": "No image provided. Use multipart or base64 JSON."}), 400

        result = predict_image(image_bytes)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 422
    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Internal server error during prediction."}), 500


if __name__ == "__main__":
    load_resources()
    port = int(os.environ.get("AI_PORT", 5001))
    print(f"[AI Service] Starting on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
