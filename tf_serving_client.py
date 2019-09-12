#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import time
tt = time.time()

from matplotlib.pyplot import imread
from grpc.beta import implementations
from protos.tensorflow.core.framework import tensor_pb2  
from protos.tensorflow.core.framework import tensor_shape_pb2  
from protos.tensorflow.core.framework import types_pb2 

from protos.tensorflow_serving.apis import predict_pb2  
from protos.tensorflow_serving.apis import prediction_service_pb2

CLASSES = ['Grape_Chlorosis', 'Corn_Southern rust', 'Wheat_Healthy', 'Wheat_Black chaff',
           'Grape_Healthy', 'Corn_Downy mildew', 'Corn_Northern leaf blight',
           'Wheat_Brown rust', 'Grape_Black rot', 'Corn_Healthy', 'Corn_Eyespot',
           'Wheat_Yellow rust', 'Grape_Esca', 'Grape_Powdery mildew', 'Wheat_Powdery mildew']

parser = argparse.ArgumentParser(description='incetion grpc client flags.')
parser.add_argument('--host', default='0.0.0.0', help='pdd serving host')
parser.add_argument('--port', default='9000', help='pdd serving port')
parser.add_argument('--image', help='path to JPEG image file')
FLAGS = parser.parse_args()


def main():  
    # create prediction service client stub
    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    # create request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'pdd_model'
    request.model_spec.signature_name = 'serving_default'
    
    # read image into numpy array
    img = imread(FLAGS.image).astype(np.float32) / 255.
    
    # convert to tensor proto and make request
    # shape is in NHWC (num_samples x height x width x channels) format
    # ensure NHWC shape and build tensor proto
    tensor_shape = [1]+list(img.shape)  
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_shape]  
    tensor_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)  
    tensor = tensor_pb2.TensorProto(
                  dtype=types_pb2.DT_FLOAT,
                  tensor_shape=tensor_shape,
                  float_val=list(img.reshape(-1)))
    # 'input_image' - name of the input layer
    request.inputs['input_image'].CopyFrom(tensor) 
    # second parameter - timeout
    resp = stub.Predict(request, 30.0)
    
    logits = resp.outputs['prediction/Softmax:0'].float_val
    print("All predictions:", logits)
    prediction = int(np.argmax(logits))
    
    print("Predicted class: %s" % CLASSES[prediction])
    print("Confidence: %.4f" % logits[prediction])
    print('total time: {}s'.format(time.time() - tt))
    

if __name__ == '__main__':
    main()