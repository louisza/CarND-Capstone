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
        PATH_TO_LABELS = r'data/udacity_label_map.pbtxt'
        NUM_CLASSES = 4
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction




        return {
            0: TrafficLight.GREEN,
            1: TrafficLight.RED,
            2: TrafficLight.UNKNOWN,
            3: TrafficLight.YELLOW,
        }.get(self.predict(image), TrafficLight.UNKNOWN)
