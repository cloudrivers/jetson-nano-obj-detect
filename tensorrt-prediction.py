import tensorflow as tf
import cv2

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

pb_fname = "./models/mobilenetv2-model-trt.pb"
#pb_fname = "./models/saved_model.pb"

trt_graph = get_frozen_graph(pb_fname)

input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.compat.v1.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

IMAGE_PATH = "./images/kaffee-croissant-small.jpg"
filenames = [IMAGE_PATH]
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image = tf.image.decode_jpeg(value, channels=3)
image2 = tf.image.resize(image, (300, 300))

image_resized = tf.bitcast(tf.cast(image2, dtype=tf.int8), tf.uint8)

scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
    tf_input: image_resized[None, ...]
})
boxes = boxes[0]  # index by 0 to remove batch dimension
scores = scores[0]
classes = classes[0]
num_detections = int(num_detections[0])

# Boxes unit in pixels (image coordinates).
boxes_pixels = []
for i in range(num_detections):
    # scale box to image coordinates
    box = boxes[i] * np.array([image_resized.shape[0],
                               image_resized.shape[1], image_resized.shape[0], image_resized.shape[1]])
    box = np.round(box).astype(int)
    boxes_pixels.append(box)
boxes_pixels = np.array(boxes_pixels)

# Remove overlapping boxes with non-max suppression, return picked indexes.
pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)


for i in pick:
    box = boxes_pixels[i]
    box = np.round(box).astype(int)
    # Draw bounding box.
    image_out = cv2.rectangle(
        image_resized, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
    label = "{}:{:.2f}".format(int(classes[i]), scores[i])
    # Draw label (class index and probability).
    draw_label(image_out, (box[1], box[0]), label)

# Save and display the labeled image.
save_image(image_out[:, :, ::-1])

