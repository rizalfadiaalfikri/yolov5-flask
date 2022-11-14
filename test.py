import argparse
import io
from PIL import Image

import torch
from flask import Flask, request, send_file
import mimetypes

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
        data = results.pandas().xyxy[0].to_json(orient="records")
        data[0]['filename'] = request.host_url +  "v1/files/" + [filename]
        return data

@app.route('/v1/files/<string:filename>', methods=["GET"])
def render_file(filename):
	if filename:
            item = os.path.join(f'nama_folder/', filename)
            if os.path.exists(item):
                return send_file(item, mimetype=mimetypes.MimeTypes().guess_type(filename)[0])
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
