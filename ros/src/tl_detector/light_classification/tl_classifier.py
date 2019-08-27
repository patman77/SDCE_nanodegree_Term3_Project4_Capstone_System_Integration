from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import os
import cv2


class TLClassifier(object):
    def __init__(self):
        # Variables
        # TODO: Move this parameter to the config_string
        MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
        PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
        # self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        # self.NUM_CLASSES = 90

        # Load frozen TF model to memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load label map
        # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for self.detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_np = np.array(image).astype(np.uint8)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                for bbox, score, clas in zip(boxes[0], scores[0], classes[0]):

                    if (score > 0.3) and (clas == 10):
                        print(clas, bbox, score)

                        ytl = int(bbox[0] * image_np.shape[0])
                        xtl = int(bbox[1] * image_np.shape[1])
                        ybr = int(bbox[2] * image_np.shape[0])
                        xbr = int(bbox[3] * image_np.shape[1])

                        cv2.rectangle(image_np, (xtl, ytl), (xbr, ybr), (0,255,0), 3)
                    
                        txt = '%.2f'%score
                        cv2.putText(image_np, txt,(xtl, ytl - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 3)

        return TrafficLight.UNKNOWN, image_np
