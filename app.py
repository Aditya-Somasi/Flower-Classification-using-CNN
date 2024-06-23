import sys
import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Set default encoding to UTF-8
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__, static_url_path='/static')
model = load_model('model/flower_classification_model.h5')
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_flower_class(model, img_path):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
    return class_names[class_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename.encode('utf-8').decode('utf-8'))
            file.save(file_path)
            prediction = predict_flower_class(model, file_path)
            return render_template('index.html', prediction=prediction, image_path=file_path)
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
