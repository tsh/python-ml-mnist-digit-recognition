import os
import base64
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    print (request.form)
    image_data = re.sub(r'^data:image/.+;base64,', '', request.form['img'])
    print (image_data)
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', '8000')),
            debug=os.environ.get('DEBUG', False))
