# jetson-nano-obj-detect

Repository to illustrate two things:

1. How to convert an existing Keras model to NVIDIAs TensorRT
1. Use this model to detect objects in realtime with the Raspberry Pi Camera module (v2) 

## Convert a Keras model to TentorTR

The Google Colab Notebook ```keras_model_to_tensorrt.ipynb``` contains some code to get an existing Keras model from Keras [ModelZoo](https://modelzoo.co/framework/keras), store it locally and convert it as TensorRT model and download it.

## Use model for realtime object detection

The ```rpi_cam_obj_detect.py``` example use an object detection model (namely MobileNetV2) as TensorRT and try to detect objects over a Raspberry Pi Camera module (v2) in realtime




