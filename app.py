import base64
from statistics import mean
import os
import re
import io

from flask import Flask, render_template, request, jsonify
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.externals import joblib
from bokeh.embed import components
from bokeh.charts import Bar, Histogram
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
    results = clf.predict([nbalanced])
    df = pd.DataFrame(np.random.randn(6,4), index=[10,20,30,40,50,60], columns=list('ABCD'))
    # p = Bar(df, 'cyl', values='B', title="SVC probability")
    p = Histogram(df['A'])
    plot_script, plot_div = components(p)

    return jsonify({'status': 'ok',
                    'svc': int(results[0]),
                    'bokeh_js': plot_script,
                    'bokeh_div': plot_div})


if __name__ == "__main__":
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', '8000')),
            debug=os.environ.get('DEBUG', False))
