import cv2
import numpy as np
import rospy
import tensorflow as tf
from utils import label_map_util
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        model_path = rospy.get_param("/traffic_light_model")
        self.path_to_graph = r'models/rcnn/frozen_inference_graph.pb'
        self.graph = self.load_graph(self.path_to_graph)
        self.sess = tf.Session()

        with self.graph.as_default():
            with tf.Session(graph=self.graph) as self.sess:
                self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
                self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        # Load a sample image.
        image_np = np.expand_dims(np.asarray(image), 0)
        result = TrafficLight.UNKNOWN

        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detect_boxes, self.detect_scores, self.detect_classes],
                                                 feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        i = 0
        result_str = 'unknown'
        score = 0
        if boxes.shape[i] > 0:
            if classes[i] == 1:  # 'Green':
                result = TrafficLight.GREEN
                result_str = 'Green'
            elif classes[i] == 2:  # 'Red':
                result = TrafficLight.RED
                result_str = 'Red'
            elif classes[i] == 3:  # 'Yellow':
                result = TrafficLight.YELLOW
                result_str = 'Yellow'

        rospy.debug('classification result {}, score {:.3f}'.format(result_str, score))

        return result
