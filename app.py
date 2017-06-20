import base64
from statistics import mean
import os
import re
import threading
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


def threshold(array):
    for i, val in enumerate(array):
        if val > 127:
            array[i] = 1
        else:
            array[i] = 0


@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    img_data = re.sub(r'^data:image/.+;base64,', '', request.form['img'])
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    iar = np.array(img)
    np.delete(iar, [3], axis=1)
    balanced = []
    for row in iar:
        for pix in row:
            balanced.append(mean(pix))

    threshold(balanced)
    nbalanced = np.array(balanced)


    return jsonify({'status': 'ok'})


if __name__ == "__main__":
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', '8000')),
            debug=os.environ.get('DEBUG', False))
