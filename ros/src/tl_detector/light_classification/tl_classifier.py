from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime


class TLClassifier(object):
    def __init__(self, is_sim):
        
        if is_sim:
            PATH_TO_GRAPH = r'light_classification/model/ssd_sim/frozen_inference_graph.pb'
            
        self.graph = tf.Graph()
        # traffic light classification score
        self.threshold = 0.5
        
        with self.graph.as_default():
            read_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                read_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
                
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
            
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        with self.graph.as_default():
            img_expanded = np.expand_dims(image,axis=0)
            start_time = datetime.datetime.now()
            (boxes, scores, classes, num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                  feed_dict = {self.image_tensor:img_expanded})
            end_time = datetime.datetime.now()
            calculation_time = end_time - start_time
            print("Calculation time :", calculation_time.total_seconds())
            
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        
        # it should be traffic light only when getting higher score than threshold
        if scores[0] > self.threshold:
            if classes[0] == 1:
                print("Traffic light is GREEN")
                return TrafficLight.GREEN
            elif classes[0] == 2:
                print("Traffic light is RED")
                return TrafficLight.GREEN
            elif classes[0] == 3:
                print("Traffic light is YELLOW")
                return TrafficLight.YELLOW
        
        return TrafficLight.UNKNOWN
