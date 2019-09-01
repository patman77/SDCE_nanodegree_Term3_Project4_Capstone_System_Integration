from styx_msgs.msg import TrafficLight
import rospy

import tensorflow as tf
import numpy as np
import os
import cv2


class TLClassifier(object):
    def __init__(self, model_name):
        # Variables

        PATH_TO_CKPT = os.path.join(model_name, 'frozen_inference_graph.pb')
        self.tl_colors = ['Red', 'Yellow', 'Green', '-', 'Undefined']
        self.tl_colorCodes = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (0, 0, 0), (200, 200, 200)]


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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Variables for frames skipping when running on a CPU
        self.on_gpu = tf.test.is_gpu_available(cuda_only=True)
        self.skip_frame = False
        self.last_state = TrafficLight.UNKNOWN
        self.last_image_np = np.zeros(1)

    def get_classification(self, image, roi):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            image (cv::Mat): image containing debug detection output

        """
        tl_state = TrafficLight.UNKNOWN

        # Input image preprocessing
        image_np = np.array(image).astype(np.uint8)
        ymin = int(roi[0] * image_np.shape[0])
        xmin = int(roi[1] * image_np.shape[1])
        ymax = int(roi[2] * image_np.shape[0])
        xmax = int(roi[3] * image_np.shape[1])

        image_cropped = image_np[ymin:ymax, xmin:xmax]

        # Frames skipping when running on a CPU
        if not self.on_gpu and self.skip_frame:
            self.skip_frame = not self.skip_frame
            return self.last_state, self.last_image_np

        # Expand dimensions since the model expects images 
        # to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_cropped, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, 
             self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Filter for robust tl_classification when there are multiple of them
        tl_states = []

        for bbox, score, clas in zip(boxes[0], scores[0], classes[0]):

            if (score > 0.3) and (clas == 10) and \
                (0.07/(roi[2]-roi[0]) < (bbox[2] - bbox[0]) < 0.5/(roi[2]-roi[0])):

                ytl = int(bbox[0] * image_cropped.shape[0])
                xtl = int(bbox[1] * image_cropped.shape[1])
                ybr = int(bbox[2] * image_cropped.shape[0])
                xbr = int(bbox[3] * image_cropped.shape[1])

                ### Classify the color of the traffic light
                # Crop the tl bbox
                tl_img = image_cropped[ytl:ybr, xtl:xbr]
                # Crop margins
                offset = int(tl_img.shape[1]/4)
                cr_img = tl_img[offset:-offset, offset:-offset]

                # Aspect ratio check
                asp_rat = cr_img.shape[0] / cr_img.shape[1]

                if 1.5 < asp_rat < 5:
                    # Convert to HSV and extract Value part from the image
                    if cv2.__version__ < '3.0.0':
                        cr_v_img = cv2.cvtColor(cr_img, cv2.cv.CV_BGR2HSV)[:,:,2]
                    else:
                        cr_v_img = cv2.cvtColor(cr_img, cv2.COLOR_BGR2HSV)[:,:,2]

                    # Finding mean intensities of each section
                    section_h = int(cr_img.shape[0]/3)
                    sections = np.hstack((np.mean(cr_v_img[:section_h]), 
                                          np.mean(cr_v_img[section_h:2*section_h]), 
                                          np.mean(cr_v_img[2*section_h:])))
                    tl_st = np.argmax(sections)
                    tl_states.append(tl_st)

                    # Draw debug information on the frame
                    try:
                        cv2.rectangle(image_np, (xmin+xtl, ymin+ytl), 
                                      (xmin+xbr, ymin+ybr), 
                                      self.tl_colorCodes[tl_st], 3)
                    except:
                        pass
                    
                    txt = '%s: %.2f'%(self.tl_colors[tl_st][0], score)
                    bot_pos = ymin+ytl-10 if ymin+ytl-10 > 30 else ymin+ybr+25
                    left_pos = xmin+xtl if xmin+xtl > 0 else 0
                    try:
                        cv2.putText(image_np, txt, (left_pos, bot_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                    self.tl_colorCodes[tl_st], 2)
                    except:
                        pass
                else:
                    tl_st = TrafficLight.UNKNOWN

                # debug
                rospy.logdebug("%s: %.3f, bbox: %s"%(self.tl_colors[tl_st], score, bbox))

        if len(set(tl_states)) == 1:
            tl_state = tl_states[0]

        try:
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), 
                          self.tl_colorCodes[tl_state], 15)
        except:
            pass

        # Update variables for frames skipping when running on a CPU
        if not self.on_gpu:
            self.last_state = tl_state
            self.skip_frame = not self.skip_frame
            self.last_image_np = image_np

        return tl_state, image_np
