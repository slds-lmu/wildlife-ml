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
    list_files,
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
    ) -> Dict:
        """
        Predict bounding boxes for a directory.

        The directory is NOT traversed recursively. All images should be in one flat
        directory.
        Note: A batch size > 1 only works if all images in the directory have the
        same shape.
        """
        file_names = list_files(directory)
        file_paths = [os.path.abspath(os.path.join(directory, f)) for f in file_names]
        print('Found {} image files in "{}"'.format(len(file_paths), directory))

        # Traverse through list according to batch size.
        print('Predicting bounding boxes ...')
        output_dict = {}
        cnt_empty = 0
        cnt_corrupt = 0
        for i in trange(0, len(file_paths), self.batch_size):
            batch_files = file_paths[i : i + self.batch_size]

            # Load images as array
            img_list = []
            for path in batch_files:
                try:
                    img_arr = np.asarray(load_image(path))
                    img_list.append(img_arr)
                except (IOError, OSError) as e:
                    cnt_corrupt += 1
                    print(
                        'Failed to load file with path "{}". It will be skipped.\n'
                        'Error: {}'.format(path, str(e))
                    )

            # Skip to next batch if no image could be loaded
            if len(img_list) == 0:
                continue

            # Combine list into full array
            imgs = np.stack(img_list)

            # Predict bounding boxes
            batch_result = self.predict(imgs)

            # Add an entry for every bounding box found with own key
            for detections, f_path in zip(batch_result.values(), batch_files):
                key_stem = os.path.split(f_path)[1]
                for i, detection in enumerate(detections, start=1):
                    detection.update({'file': f_path})
                    output_dict.update({key_stem + '_' + str(i).zfill(3): detection})

                if len(detections) == 0:
                    output_dict.update(
                        {
                            key_stem
                            + '_'
                            + str(1).zfill(3): {'category': int(-1), 'file': f_path}
                        }
                    )
                    cnt_empty += 1

        share_empty = cnt_empty / (len(file_names) - cnt_corrupt) * 100
        print(
            f'Processing finished. Found bounding boxes for {1-share_empty} percent of '
            f'images at threshold {self.confidence_threshold}.'
        )

        # Save output as JSON file
        if save_file:
            if output_file is None:
                output_file = directory + '_megadetector.json'
            save_as_json(output_dict, output_file)
            print('Results were saved in "{}"'.format(output_file))

        return output_dict

    def predict(self, imgs: np.ndarray) -> Dict[int, List[Dict]]:
        """Predict bounding boxes for a numpy array."""
        # Obtain predictions for full batch
        b_box, b_score, b_class = self.graph(imgs)

        output_dict: Dict[int, List[Dict]] = {}

        # Construct result dictionary for all samples in the batch.
        for i, (boxes, scores, classes) in enumerate(zip(b_box, b_score, b_class)):

            # Loop over every single bounding box in every image
            detections_cur_image = []
            for b, s, c in zip(boxes, scores, classes):

                if s > self.confidence_threshold:
                    detection_entry = {
                        'category': int(c),
                        'conf': truncate_float(float(s), precision=4),
                        'bbox': MegaDetector._convert_coords(b),
                    }

                    detections_cur_image.append(detection_entry)

            output_dict.update({i: detections_cur_image})

        return output_dict

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
