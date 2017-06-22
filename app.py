import base64
from statistics import mean
import os
import re
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from sklearn.externals import joblib
from utils import threshold


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


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
    clf = joblib.load('svc.pkl')
    results = clf.predict([balanced])
    return jsonify({'status': 'ok',
                    'svc': int(results[0])})


if __name__ == "__main__":
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', '8000')),
            debug=os.environ.get('DEBUG', False))
