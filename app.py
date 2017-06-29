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

from get_data import CLASSIFIERS_DIR, DATA_DIR, get_mnist, prep_model


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    img_data = re.sub(r'^data:image/.+;base64,', '', request.form['img'])
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))

    img.save('client_digit.png')

    iar = np.array(img)
    np.delete(iar, [3], axis=1)
    balanced = []
    for row in iar:
        for pix in row:
            balanced.append(mean(pix))

    threshold(balanced)
    nbalanced = np.array(balanced)
    clf = joblib.load(os.path.join(CLASSIFIERS_DIR, 'svc.pkl'))
    results = clf.predict([nbalanced])
    # Render chart
    d = {'values': clf.predict_proba([nbalanced])[0]}
    df = pd.DataFrame(d)
    p = Bar(df, values='values', label='index', title="SVC Probability", legend=False, toolbar_location=None)

    plot_script, plot_div = components(p)

    return jsonify({'status': 'ok',
                    'svc': int(results[0]),
                    'bokeh_js': plot_script,
                    'bokeh_div': plot_div})


if __name__ == "__main__":
    # Check if we have trained models.
    if not os.path.exists(CLASSIFIERS_DIR):
        if not os.path.exists(DATA_DIR):
            get_mnist()
        prep_model()

    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', '8000')),
            debug=os.environ.get('DEBUG', False))
