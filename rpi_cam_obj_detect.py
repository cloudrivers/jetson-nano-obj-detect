import os
import cv2
import numpy as np
import tensorflow as tf

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
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'mobilenetv2-model.pb')

detection_graph = tf.compat.v1.Graph()

with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    TFSess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

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
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)
    
    (boxes, scores, classes, num) = TFSess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    print(boxes)
    print(np.squeeze(boxes))
    print(classes)
    print(num)
    print(scores)


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

