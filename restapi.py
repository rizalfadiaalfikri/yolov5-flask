"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image
import os

import torch
from flask import Flask, request, redirect

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640)
        results.save(save_dir="static/result")
        os.rename(os.path.join(f'static/result/','image0.jpg'), os.path.join(f'static/result/',image_file.filename))
        detail = results.pandas().xyxy[0].to_json(orient="records")
        data = {
        	"filename" : request.host_url +  "static/result/" + image_file.filename,
        	"detail" : detail
        }
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat