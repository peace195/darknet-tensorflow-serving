import requests
import cv2
import numpy as np
import sys
from darkflow.net.yolo import predict
from darkflow.net.build import TFNet
import json

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/darkflow:predict'
# The image URL is the location of the image we should send to the server
IMAGE_DIR = 'test/training/images/1.jpg'
options = {"model": 'cfg/tiny-yolo-test.cfg', "threshold": 0.1}
tfnet = TFNet(options)

def main():
    # Compose a JSON Predict request (send JPEG image in base64).
    im = cv2.imread(IMAGE_DIR)
    imsz = cv2.resize(im, (416, 416))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    dim = np.shape(imsz)
    predict_request = '{"signature_name":"predict", "instances" : [{"input": %s}]}' % imsz.tolist()
    # Send few requests to warm-up the model.
    response = requests.post(SERVER_URL, data=predict_request)
    json_response = json.loads(response.text)
    net_out = np.squeeze(np.array(json_response['predictions'], dtype='float32'))
    boxes = tfnet.framework.findboxes(net_out)
    h = dim[0]
    w = dim[1]
    threshold = tfnet.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = tfnet.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })

    for prediction in boxesInfo:
        print(prediction)


if __name__ == '__main__':
    main()