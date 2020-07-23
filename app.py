import os
from PIL import Image
from flask import Flask, request, Response, render_template
import cv2
import io
from tflite_model import *
from camera import VideoCamera
from camera import model_hand_lm
import json
app = Flask(__name__)

#init tflite model
model_hand = Model("hand_landmark.tflite")
in_shape = model_hand.getInputShape()
h_hand = in_shape[1]
w_hand = in_shape[2]

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response



@app.route('/')
def index():
    return Response('ETRI Hand Tracking Test 2020.07.14 #1')
    #return render_template('index.html')


@app.route('/image', methods=['POST'])
def image():
    try:

        image_file = request.files['image'].read()  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        #if threshold is None:
        #  threshold = 0.5
        #else:
        #  threshold = float(threshold)

        img = np.array(Image.open(io.BytesIO(image_file)))
        #img = cv2.imread(image_file)
        img_reszd = cv2.resize(img, (w_hand, h_hand))
        img_pre = ((img_reszd - 127.5) /  127.5).astype('float32')
        print(img_pre.shape) 
        output_tensors = model_hand.runModel(img_pre[:,:,0:3])
        output_json = hand_json(output_tensors, [img.shape[0],img.shape[1]], [h_hand, w_hand])
        return output_json

    except Exception as e:
        print('POST /image error: %e' % e)
        return e



if __name__ == '__main__':
	# without SSL
     app.run(host='0.0.0.0', port=8080)
    # app.run(debug=True)
	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
