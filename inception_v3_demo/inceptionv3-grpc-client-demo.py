import argparse
from datetime import datetime

import grpc
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def read_tensor_from_image_file3(filename, input_height=299,
                                 input_width=299,
                                 input_mean=127.5,
                                 input_std=127.5):
    import cv2
    img = Image.open(filename)
    img = np.array(img, np.float32)
    resized = cv2.resize(img, (input_height, input_width), interpolation=cv2.INTER_LINEAR)
    # img = (resized - np.array([input_mean])) / np.array([input_std]).astype(np.float)
    return resized


def load_labels(label_file):
    label = []
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            label.append(line.strip())
    return label


def grpc_inception_v3_client(target, model_name, label_file, img):
    image = read_tensor_from_image_file3(img)
    start = datetime.now()
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(tf.make_tensor_proto(image, shape=[1, 299, 299, 3]))
    response = stub.Predict(request, 5.0)  # 5 seconds
    channel.close()
    result = [tf.make_ndarray(response.outputs[a]) for a in list(response.outputs)]
    results = np.array(result).ravel()
    top_k = np.array(results).argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i] + ": " + str(results[i]))
    deltaTime = (datetime.now() - start).total_seconds()
    print("Time:", deltaTime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="host:port")
    parser.add_argument("--model_name", default="inception_v3", help="host:port")
    parser.add_argument("--label_file", help="inception_v3 label file")
    parser.add_argument("--img", required=True, help="img file")
    args = parser.parse_args()

    grpc_inception_v3_client(args.host, args.model_name, args.label_file, args.img)
    # grpc_inception_v3_client("inceptionv3grpc.user.fenghub.com:30081", "inception_v3", "./labels.txt",
    #                          "/Users/jinxiang/Downloads/dataset-imagenet/ILSVRC2012_val_00000904.JPEG")
    # python ./inceptionv3-client-demo.py --host=http://inceptionv3demo.user.fenghub.com:30080 \
    # --model_name=inception_v3 --label_file=./labels.txt --img=./ILSVRC2012_val_00000904.JPEG
