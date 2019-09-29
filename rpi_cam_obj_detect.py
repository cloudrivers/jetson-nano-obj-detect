import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

IM_WIDTH = 1280
IM_HEIGHT = 720

def gstreamer_pipeline(
        capture_width=IM_WIDTH, 
        capture_height=IM_HEIGHT, 
        display_width=IM_WIDTH, 
        display_height=IM_HEIGHT, 
        framerate=60, 
        flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

MODEL_NAME = 'mobilenetv2'

NUM_CLASSES = 90
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'mobilenetv2-model-trt.pb')
OUTPUT_NAMES = ['Logits/Softmax']
INPUT_NAMES = ['input_1']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

detection_graph = get_frozen_graph(PATH_TO_CKPT)

# Create session and load graph
tf_config = tf.compat.v1.ConfigProto() 
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(detection_graph, name='')

for node in detection_graph.node:
    print(node.name)
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

input_tensor_name = INPUT_NAMES[0] + ":0"
output_tensor_name = OUTPUT_NAMES[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(input_tensor_name, output_tensor_name))

input_tensor = tf_sess.graph.get_tensor_by_name(input_tensor_name)
output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

WIN_NAME = 'Jetson Nano Object Detection'

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

frameCount = 0

while cv2.getWindowProperty(WIN_NAME,0) >= 0:

    t1 = cv2.getTickCount()
    ret_val, frame = cap.read();
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame.setflags(write=1)
    frame_expanded = np.expand_dims(resized_frame, axis=0)
    frame_preprocessed = preprocess_input(frame_expanded)
    
    preds = tf_sess.run(
        output_tensor,
        feed_dict={input_tensor: frame_preprocessed})


    print(preds)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.imshow(WIN_NAME, frame)
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    frameCount+=1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

