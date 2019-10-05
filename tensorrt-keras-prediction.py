import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def get_frozen_graph(graph_file):
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

#img_path = './images/kafee-croissant-small.jpg'
img_path = './images/flugzeug.jpg'
trt_graph = get_frozen_graph('./models/mobilenetv2-model-trt-v2.pb')
output_names = ['Logits/Softmax']
input_names = ['input_1']

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.compat.v1.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

img = image.load_img(img_path, target_size=image_size[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feed_dict = {
    input_tensor_name: x
}
preds = tf_sess.run(output_tensor, feed_dict)

print('Predicted:', decode_predictions(preds, top=3)[0])
