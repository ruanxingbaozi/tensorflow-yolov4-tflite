import json
from datetime import datetime

import numpy as np
import requests
from PIL import Image
import argparse

def read_tensor_from_image_file3(filename, input_height=299,
                                 input_width=299,
                                 input_mean=127.5,
                                 input_std=127.5):
    import cv2
    img = Image.open(filename)
    img = np.array(img, np.float32)
    resized = cv2.resize(img, (input_height, input_width), interpolation=cv2.INTER_LINEAR)
    # img = (resized - np.array([input_mean])) / np.array([input_std])
    return resized


def load_labels(label_file):
    label = []
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            label.append(line.strip())
    return label


def http_inception_v3_client(target, model_name, label_file, img):
    url = target + "/v1/models/" + model_name + ":predict"
    image = read_tensor_from_image_file3(img)
    start = datetime.now()
    request = {
        "signature_name": 'predict_images',
        "instances": image.reshape([1, 299, 299, 3]).tolist()
    }
    response = requests.post(url, data=json.dumps(request))
    results = response.json()
    top_k = np.array(results['predictions'][0]).argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i] + ": " + str(results['predictions'][0][i]))
    deltaTime = (datetime.now() - start).total_seconds()
    print("Time:", deltaTime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="host:port")
    parser.add_argument("--model_name", default="inception_v3", help="host:port")
    parser.add_argument("--label_file", help="inception_v3 label file")
    parser.add_argument("--img", required=True, help="img file")
    args = parser.parse_args()

    http_inception_v3_client(args.host, args.model_name, args.label_file, args.img)
    # http_inception_v3_client("http://inceptionv3demo.user.fenghub.com:30080", "inception_v3", "./labels.txt",
    #                          "/Users/jinxiang/Downloads/dataset-imagenet/ILSVRC2012_val_00000904.JPEG")
    # python ./inceptionv3-client-demo.py --host=http://inceptionv3demo.user.fenghub.com:30080 \
    # --model_name=inception_v3 --label_file=./labels.txt --img=./ILSVRC2012_val_00000904.JPEG