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


        # Load frozen TF model to memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Definite input and output Tensors for self.detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        tl_state = TrafficLight.UNKNOWN

        # with self.detection_graph.as_default():
            # with tf.Session(graph=self.detection_graph) as sess:


        image_np = np.array(image).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Filter for robust tl_classification when there are multiple of them
        tl_states = []

        for bbox, score, clas in zip(boxes[0], scores[0], classes[0]):

            if (score > 0.3) and (clas == 10):
                print(clas, bbox, score)

                ytl = int(bbox[0] * image_np.shape[0])
                xtl = int(bbox[1] * image_np.shape[1])
                ybr = int(bbox[2] * image_np.shape[0])
                xbr = int(bbox[3] * image_np.shape[1])

                # Classify the color of the traffic light
                # Crop the tl bbox
                # TODO: Add aspect ratio check
                tl_img = image_np[ytl:ybr, xtl:xbr]
                # Crop margins
                offset = int(tl_img.shape[1]/4)
                cr_img = tl_img[offset:-offset, offset:-offset]

                # Convert to HSV and extract Value part from the image
                cr_v_img = cv2.cvtColor(cr_img, cv2.COLOR_RGB2HSV)[:,:,2]

                # Finding mean intensities of each section
                section_h = int(cr_img.shape[0]/3)
                sections = np.hstack((np.mean(cr_v_img[:section_h]), 
                                      np.mean(cr_v_img[section_h:2*section_h]), 
                                      np.mean(cr_v_img[2*section_h:])))
                tl_st = np.argmax(sections)
                tl_states.append(tl_st)

                # Draw debug information on the frame
                cv2.rectangle(image_np, (xtl, ytl), (xbr, ybr), (0,255,0), 3)
            
                txt = '%d: %.2f'%(tl_st, score)
                cv2.putText(image_np, txt,(xtl, ytl - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 3)

        if len(set(tl_states)) == 1:
            tl_state = tl_states[0]

        return tl_state, image_np
