from flask import Flask, render_template, request, redirect, url_for
import time
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from classification import predict

def static_clear():
    for root, dirs, files in os.walk("static"):
        for f in files:
            for i in ["JPG", "jpeg", "jpg", "JPEG", "PNG", "png"]:
                if i in f.split("."):
                    try:
                        os.remove("static/{}".format(f))
                        print("removed {}".format(f))
                    except FileNotFoundError:
                        pass

rooms={'diningroom':'Dining Room','livingroom':'Living Room','kitchen':'Kitchen','bedroom':'Bed Room'}

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'JPG', 'JPEG','png','PNG'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    static_clear()
    return render_template("home.html")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('resultroom', filename=filename))
            else:
                return render_template("404.html")
        except Exception:
            return render_template("404.html")

@app.route('/resultroom/<filename>')
def resultroom(filename):
    dataset = 'allrooms'
    model = torch.load('_model_24.pt')
    result=predict(model,filename)
    for i in result:
        if result[i]==max(result.values()):
            res=rooms[i]
            percentage=round(max(result.values())*100,2)
            break
    return render_template("resultroom.html", name=res, filename=filename,percentage=percentage)

if __name__ == '__main__':
    app.run(debug=True)
