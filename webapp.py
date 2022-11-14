"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import re

import torch
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/' 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        filename = file.filename
        
        split_filename =  filename.split("_")
        tanggal = split_filename[1]
        ext = tanggal.split(".")
        ext2 = ext[0]
        # print(ext2)
        # print(split_filename[0])
        # print(type(file))
        # print(filename)
        
        if not file:
            return

        # if file.filename == '': 
        #     # flash('No selected file') 
        #     return redirect(request.url) 
        # if file and allowed_file(file.filename): 
        #     # img_bytes = file.read()
        #     # img = Image.open(io.BytesIO(img_bytes))
        #     # results = model([img])
        #     # print(results)
        #     filename = secure_filename(file.filename) 
        #     # flash('file {} saved'.format(file.filename)) 
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'] + filename)) 
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])
        print(results.pandas().xyxy[0].value_counts('name'))
        results.save(save_dir="static/")

        results.render()  # updates results.imgs with boxes and labels
        return redirect("static/image0.jpg")

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat
