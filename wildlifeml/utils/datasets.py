"""Utilities related to WildlifeDatasets."""
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
)

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from wildlifeml.utils.io import load_json
from wildlifeml.utils.misc import flatten_list

BBOX_SUFFIX_LEN: Final[int] = 4


def render_bbox(
    img: Image,
    x_coords: List[Tuple[int, int]],
    y_coords: List[Tuple[int, int]],
    outline: str = 'red',
    border_width: int = 10,
) -> Image:
    """Render a bounding box into a PIL Image."""
    img_draw = ImageDraw.Draw(img)
    for x, y in zip(x_coords, y_coords):
        img_draw.rectangle(
            xy=((x[0], y[0]), (x[1], y[1])),
            outline=outline,
            width=border_width,
        )
    return img


def map_img_to_bboxes(img_key: str, detector_dict: Dict) -> List[str]:
    """Find all bbox-level keys for img key."""
    return [k for k in detector_dict.keys() if img_key in k]


def map_bbox_to_img(bbox_key: str) -> str:
    """Find img key for bbox-level key."""
    return bbox_key[: len(bbox_key) - BBOX_SUFFIX_LEN]


# --------------------------------------------------------------------------------------


def do_stratified_splitting(
    detector_dict: Dict,
    img_keys: List[str],
    splits: Tuple[float, float, float],
    meta_dict: Optional[Dict] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[Any], ...]:
    """Perform stratified holdout splitting."""
    keys_array = np.array(img_keys)
    if meta_dict is not None:
        strat_dict = get_strat_dict(meta_dict)
        strat_var_array = np.array([strat_dict[k] for k in img_keys])
    else:
        strat_dict = {}
        strat_var_array = np.empty(len(keys_array))

    # Split intro train and test keys
    sss_tt = StratifiedShuffleSplit(
        n_splits=1,
        train_size=splits[0] + splits[1],
        test_size=splits[2],
        random_state=random_state,
    )
    print('---> Performing stratified split')
    idx_train, idx_test = next(iter(sss_tt.split(keys_array, strat_var_array)))
    keys_train = keys_array[idx_train].tolist()
    keys_test = keys_array[idx_test].tolist()
    keys_val = []

    if splits[1] > 0:
        # Split train again into train and val
        keys_array = np.array(keys_train)
        if len(strat_dict) > 0:
            strat_dict_train = {k: v for k, v in strat_dict.items() if k in keys_train}
            strat_var_array = np.array([strat_dict_train[k] for k in keys_train])
        else:
            strat_var_array = np.empty(len(keys_array))
        sss_tv = StratifiedShuffleSplit(
            n_splits=1,
            train_size=splits[0],
            test_size=splits[1],
            random_state=random_state,
        )
        idx_train, idx_val = next(iter(sss_tv.split(keys_array, strat_var_array)))
        keys_train = keys_array[idx_train].tolist()
        keys_val = keys_array[idx_val].tolist()

    # Get keys on bbox level
    print('---> Mapping image keys to bbox keys')
    keys_train = [map_img_to_bboxes(k, detector_dict) for k in keys_train]
    keys_val = [map_img_to_bboxes(k, detector_dict) for k in keys_val]
    keys_test = [map_img_to_bboxes(k, detector_dict) for k in keys_test]

    return flatten_list(keys_train), flatten_list(keys_val), flatten_list(keys_test)


def do_stratified_cv(
    detector_dict: Dict,
    img_keys: List[str],
    folds: Optional[int],
    meta_dict: Optional[Dict],
    random_state: Optional[int] = None,
) -> Tuple[List[Any], ...]:
    """Perform stratified cross-validation."""
    keys_array = np.array(img_keys)
    if meta_dict is not None:
        strat_dict = get_strat_dict(meta_dict)
        strat_var_array = np.array([strat_dict[k] for k in img_keys])
    else:
        strat_var_array = np.empty(len(keys_array))

    if folds is None:
        raise ValueError('Please provide number of folds in cross-validation.')

    # Get k train-test splits
    skf = StratifiedKFold(n_splits=folds, random_state=random_state)
    print('---> Performing stratified splits')
    idx_train = [list(i) for i, _ in skf.split(keys_array, strat_var_array)]
    idx_test = [list(j) for _, j in skf.split(keys_array, strat_var_array)]
    keys_train, keys_test = [], []
    print('---> Mapping image keys to bbox keys')
    for i, _ in enumerate(idx_train):
        slice_keys = keys_array[idx_train[i]].tolist()
        keys_new = [map_img_to_bboxes(k, detector_dict) for k in slice_keys]
        keys_train.append(flatten_list(keys_new))
    for i, _ in enumerate(idx_test):
        slice_keys = keys_array[idx_test[i]].tolist()
        keys_new = [map_img_to_bboxes(k, detector_dict) for k in slice_keys]
        keys_test.append(flatten_list(keys_new))

    return keys_train, keys_test


def get_strat_dict(meta_dict: Dict[str, Dict]) -> Dict[str, str]:
    """Create stratifying variable for dataset splitting."""
    if len(meta_dict) == 0:
        return {}

    else:
        lengths = [len(v) for v in meta_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                'All variables provided for stratification must have the same number '
                'of elements.'
            )
        strat_dict = {
            k: '_'.join([str(v) for v in meta_dict[k].values()])
            for k in meta_dict.keys()
        }
        return strat_dict


# --------------------------------------------------------------------------------------


def separate_empties(
    detector_file_path: str,
    conf_threshold: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Separate images into empty and non-empty instances according to Megadetector."""
    detector_dict = load_json(detector_file_path)
    keys_empty = [
        k for k in detector_dict.keys() if detector_dict[k].get('category') == -1
    ]
    if conf_threshold is not None:
        keys_empty.append(
            [
                k
                for k in detector_dict.keys()
                if detector_dict[k].get('conf') < conf_threshold
            ]
        )
    keys_nonempty = list(set(detector_dict.keys()) - set(keys_empty))
    return keys_empty, keys_nonempty


# --------------------------------------------------------------------------------------


def map_preds_to_img(
    preds_bboxes: Dict[str, float],
    detector_dict: Dict,
) -> Dict[str, int]:
    """Map predictions on bbox level back to img level."""
    keys_imgs = list(set([map_bbox_to_img(k) for k in preds_bboxes.keys()]))
    preds_imgs = {}

    for key in keys_imgs:
        # Find all bbox predictions for img
        keys_k = map_img_to_bboxes(key, detector_dict)
        preds_k = [preds_bboxes[k] for k in keys_k]
        weights_k = [detector_dict[k].get('conf') for k in keys_k]
        weights_k_normalized = [i / sum(weights_k) for i in weights_k]
        pred = sum([i * j for i, j in zip(weights_k_normalized, preds_k)])
        preds_imgs.update({key: pred})

    return preds_imgs
