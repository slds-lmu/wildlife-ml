"""
Object detection via Microsoft`s Megadetector.

Code for compatibility has been adopted from 'https://github.com/microsoft/CameraTraps'.
"""
import os
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import trange

from wildlifeml.utils.io import load_image, save_as_json
from wildlifeml.utils.misc import (
    download_file,
    list_image_paths,
    truncate_float,
)


class MegaDetector:
    """MegaDetector for object detecting."""

    def __init__(
        self,
        batch_size: int = 2,
        confidence_threshold: float = 0.1,
        model_path: str = 'models/megadetector.pb',
        url: Optional[str] = None,
    ) -> None:
        """Initialize a MegaDetector object."""
        if not os.path.exists(model_path):
            print('Model at location "{}" does not exist.'.format(model_path))
            MegaDetector.download_model(model_path, url)

        self.graph = MegaDetectorGraph(model_path)
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def download_model(target_path: str, url: Optional[str] = None) -> None:
        """
        Download the megadetector model.

        If no url is passed, the officially supplied link from the megadetector
        repository is chosen.
        """
        if url is None:
            url = (
                'https://lilablobssc.blob.core.windows.net/models'
                '/camera_traps/megadetector/megadetector_v2.pb'
            )

        print('Starting download from "{}"'.format(url))
        download_file(url, target_path)
        print('Downloading the model was successful.')

    def predict_directory(
        self, directory: str, save_file: bool = True, output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Predict bounding boxes for a directory.

        The directory is traversed recursively.
        Note: A batch size > 1 only works if all images in the directory have the
        same shape.
        """
        file_paths = list_image_paths(directory)
        print('Found {} image files in "{}"'.format(len(file_paths), directory))

        # Traverse through list according to batch size.
        print('Converting directory ...')
        output_list = []
        for i in trange(0, len(file_paths), self.batch_size):
            batch_files = file_paths[i : i + self.batch_size]

            # Load images as array
            imgs = np.stack([np.asarray(load_image(path)) for path in batch_files])
            # Predict bounding boxes
            batch_result = self.predict(imgs)
            # Update with file paths
            [r.update({'file': f}) for r, f in zip(batch_result, batch_files)]
            # Add batch results to main output
            output_list.extend(batch_result)

        # Save output as JSON file
        if save_file:
            if output_file is None:
                output_file = directory + '_megadetector.json'
            save_as_json(output_list, output_file)

        return output_list

    def predict(self, imgs: np.ndarray) -> List[Dict]:
        """Predict bounding boxes for a numpy array."""
        # Obtain predictions for full batch
        b_box, b_score, b_class = self.graph(imgs)

        output_list = []

        # Construct result dictionary for all samples in the batch.
        for boxes, scores, classes in zip(b_box, b_score, b_class):

            # Loop over every single bounding box in every image
            detections_cur_image = []
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):

                if s > self.confidence_threshold:
                    detection_entry = {
                        'category': int(c),
                        'conf': truncate_float(float(s), precision=4),
                        'bbox': MegaDetector._convert_coords(b),
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = truncate_float(float(s), precision=4)

            output_list.append(
                {
                    'max_detection_conf': max_detection_conf,
                    'detections': detections_cur_image,
                }
            )

        return output_list

    @staticmethod
    def _convert_coords(coords: np.ndarray) -> Tuple[float, ...]:
        """
        Convert coordinate representation.

        Convert coordinates from the model's output format [y1, x1, y2, x2] to the
        format [x1, y1, width, height]. All coordinates (including model outputs)
        are normalized in the range [0, 1].

        :param coords: np.array of predicted bounding box coordinates
        with shape [y1, x1, y2, x2]
        :return:
        """
        width = coords[3] - coords[1]
        height = coords[2] - coords[0]
        new_coords = [coords[1], coords[0], width, height]

        # Truncate floats for JSON output.
        return tuple([truncate_float(d, 4) for d in new_coords])


class MegaDetectorGraph:
    """Class for the computational graph of the Megadetector."""

    def __init__(self, graph_path: str) -> None:
        """Initialize a MegaDetectorGraph object."""
        graph = MegaDetectorGraph._load_model(graph_path)
        self.tf_session = tf.Session(graph=graph)

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def _load_model(graph_path: str) -> tf.Graph:
        """Load a TF Megadetector graph from a .pb file."""
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def run(self, imgs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass through the model.

        :param imgs: Numpy array of shape (B x H x W x C)
        :return: Tuple of tensors: bbox, scores and classes
        """
        return self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: imgs},
        )

    def __call__(self, imgs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass through the model.

        :param imgs: Numpy array of shape (B x H x W x C)
        :return: Tuple of tensors: bbox, scores and classes
        """
        return self.run(imgs)
