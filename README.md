# Welcome to **Wildlife ML**!

Do you have a wildlife camera trap image database and are finding it challenging
to apply artificial intelligence systems due to the lack of labeled images?
Look no further than our new GitHub repository, which provides a software package
designed specifically to help ecologists to leverage the potential of active learning
and deep learning for image classification.

Our approach is two-fold, combining improved strategies for object detection and
image classification with an active learning system that allows for
more efficient training of deep learning models.
Plus, our software package is designed to be user-friendly,
making it accessible to researchers without specific programming skills.
With our proposed two-stage framework, we show that implementing active learning can
lead to improved predictive performance and more efficient use of pre-labeled data.
Join us in making ecological practice more accessible and effective!

## What can `wildlife-ml` do?

As previously mentioned, our package is based on a two-stage approach.
First, we leverage the predictive power of the amazing
[Megadetector](https://github.com/microsoft/CameraTraps) (**MD**) to pre-select images
that contain wildlife.
Additionally, MD provides bounding boxes of the detected wildlife, which allows the
algorithm of our next stage to directly focus on the element in question.
This greatly reduces the difficulty of the problem and as a result boosts predictive
performance of the classificator network.

Apart from our classification pipeline, we offer a simple interface for active
learning.

## Examples

We offer introductory minimal example scripts, which showcase our powerful package:

**EXAMPLE SCRIPTS COMING SOON**

*Please see our
[experiments repository](https://github.com/slds-lmu/wildlife-experiments/) in the
meantime :).*

## Citation

By using this repo, please cite our paper [Automated wildlife image classification: An active learning tool for ecological applications](https://arxiv.org/abs/2303.15823), here is a bibtex entry for it:

```
@misc{bothmann2023automated,
      title={Automated wildlife image classification: An active learning tool for ecological applications},
      author={Ludwig Bothmann and Lisa Wimmer and Omid Charrakh and Tobias Weber and Hendrik Edelhoff and Wibke Peters and Hien Nguyen and Caryl Benjamin and Annette Menzel},
      year={2023},
      eprint={2303.15823},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi={https://doi.org/10.48550/arXiv.2303.15823}
}
```

## Workflow

### 01: Folder structure

An optimal folder structure for working with our package is like:

```
├── labels.csv
├── images_megadetector.json
└── images
    ├── img_id1.xx
    ├── ...
    └── img_idn.xx
```

where `labels.csv` is a headless csv that contains entries as:

```
img_id1.xx,<int_label>
...
img_idn.xx,<int_label>
```

If your dataset follows the structure of separating classes by directories, we also
offer a `DatasetConverter` for converting the file structure to the one above.

`.xx` symbolizes that it doesn't matter what format
the image files have as long as the format is compatible with `PIL`.

###  02: Prediction of Bounding Boxes

#### Running the Megadetector

This package includes an implementation of the
[Megadetector](https://github.com/microsoft/CameraTraps) for detecting objects in
camera traps.
It is accessible as a python object:

```python
from wildlifeml import MegaDetector

md = MegaDetector(batch_size=32, confidence_threshold=0.1)
md.predict_directory(directory='<directory>', output_file='<output_file>')
```

`<directory>` should be a directory that contains images, where bounding boxes
should be predicted. More options and arguments are available. The results are saved
in a `.json` file.

We also offer a CLI option:

```shell
wildlifeml-get-bbox -d <directory>
```

Access help over the `--help` flag.

#### The MD index file

By default, the MD saves its results in `images_megadetector.json`.
The file could look something like:

```
{
    "img_id1.xx_001": {
        "category": 1,
        "conf": 0.9999,
        "bbox": [
            0.3052,
            0.6357,
            0.08082,
            0.1481
        ],
        "file": "path_to_img_id1.xx"
    },
    ....
}
```

Each detected bounding box receives one entry in the `.json` file.
The keys in MD file are composed as `<img_path>_<idx_of_bbox>`. This implies that an
image can have multiple bounding boxes.

Each entry has a `category`. `1` implies a bounding box with wildlife. `-1` indicates
that the megadetector has not found a bounding box.

#### Mapping bounding boxes to images

The above structure has one distinct weakness: We don't know how many bounding boxes
an image have or what keys of the MD file point to each image.

This can be mitigated by the `BBoxMapper`, which computes a dictionary that is
structured as follows:

```
{
    "img_id1.xx": [
        "img_id1.xx_001",
        "img_id1.xx_002",
        ........
        "img_id1.xx_nnn"
    ],
    .......
}
```

The `BBoxMapper` requires a path to the MD file.

```python
from wildlifeml.data import BBoxMapper
mapper = BBoxMapper(detector_file_path='<path_to_images_megadetector.json>')
key_map = mapper.get_keymap()
```

### 03: Creating a Wildlife Dataset

#### Initializing a Dataset

Our dataset builds on the Keras `Sequence` utility and thus supports multi-threaded
loading during training. For defining the content of the dataset, a list of keys
pointing to the respective images shall be provided.

Initializing our `WildlifeDataset` requires providing a list of strings `keys`.
This corresponds to a list that contains the identifiers provided by the MD file, e.g:

```python
train_keys = ['img_id1.xx_001', 'img_id1.xx_002', 'img_id2.xx_001']
```

Using a key based initialization, one can easily derive cross validation splits,
test sets or other variants based on one MD file.

Other required parameters are:

- `image_dir`: The image directory
- `detector_file_path`: Path to the MD file
- `batch_size`: Batch size
- `bbox_map`: Result obtained from `BBoxMapper`

For training the `label_file_path` is also required.
All in all, an instantiation of `WildlifeDataset` is done like:

```python
from wildlifeml.data import WildlifeDataset

wd = WildlifeDataset(
    keys=train_keys,
    image_dir='<path_to_image_dir>',
    detector_file_path='<path_to_images_megadetector.json>',
    batch_size=8,
    bbox_map=key_map,
    label_file_path='<path_to_labels.csv>',
)
```

`do_cropping` is by default set to `True` and leads to the image being directly
cropped to their respective bounding boxes when being loaded in the data process.

### 04: Training a classifier

`wildlife-ml` contains the `WildlifeTrainer` that is an interface for directly training
a classifier on the MD processed wildlife data.
The trainer first loads a pretrained `model_backbone` from the Keras hub and then
conducts training in two stages. First, the backbone itself is frozen and only a linear
head on top of the backbone is trainable (`transfer`-phase).
Second, parts or the full `model_backbone` is unfrozen and more parts of the network
can be adapted (`finetune`-phase).

The trainer has following mandatory parameters:

- `batch_size`: Batch size
- `loss_func`: TF or Keras loss function
- `num_classes`: The number of classes in the dataset
- `transfer_epochs`: How many epochs to train in the `transfer`-phase.
- `finetune_epochs`: How many epochs to train in the `finetune`-phase.
- `transfer_optimizer`: Keras optimizer to use in the `transfer`-phase.
- `finetune_optimizer`: Keras optimizer to use in the `finetune`-phase.
- `finetune_layers`: Number of layers to unfreeze for the `finetune`-phase.

Thus, a minimal training initialization is for example:

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from wildlifeml.training.trainer import WildlifeTrainer

trainer = WildlifeTrainer(
    batch_size=8,
    loss_func=SparseCategoricalCrossentropy(),
    num_classes=10,
    transfer_epochs=10,
    finetune_epochs=10,
    transfer_optimizer=Adam(),
    finetune_optimizer=Adam(),
    finetune_layers=3
)
```

The training itself is triggered over the `fit` function and providing a training and
validation `WildlifeDataset`.

```python
trainer.fit(train_dataset, val_dataset)
```

### 05: Evaluating a model

Due to our cascaded MD approach. The evaluation of the model requires a bit more
care than a usual Keras model.
Additionally, we make the assumption that an image does only carry one label.
As the MD can provide multiple bounding boxes, the final decision is made by a
confidence-weighted vote.
This is contained in the  `Evaluator`, which computes metrics by incorporating the
decision of the MD and respecting the multiple-boxes-per-image dilemma.

Required parameters for the evaluator are:

- `label_file_path`: Path pointing to the labels
- `detector_file_path`: Path to the MD file
- `dataset`: `WildlifeDataset` to evaluate
- `num_classes`: Number of classes in the dataset.

If the `Evaluator` is supposed to discard boxes that do not meet the MD confidence
threshold as empty, it is advisable to also set the `conf_threshold` parameter.
Per default, the `Evaluator` assumes the empty class to be encoded with the largest
number with class labels starting from 0 (e.g., class 9 for a total of 10 classes).
Specify otherwise if this is not the case in your dataset.

An example is:

```python
from wildlifeml.training.evaluator import Evaluator

evaluator = Evaluator(
    detector_file_path='<path_to_images_megadetector.json>',
    label_file_path='<path_to_labels.csv>',
    dataset=training_dataset,
    num_classes=10,
    conf_threshold=0.1,
    empty_class_id=99
)
```

For a trained model, which is contained in a `WildlifeTrainer`, the accuracy, precision,
recall and f1 score is computed as:

```python
evaluator.evaluate(trainer)
metrics = evaluator.compute_metrics()
```

If you wish to extract the predictions and ground-truth labels for all individual
observations, use `evaluator.get_details()`.

### 06: Active Learning

Apart from fitting a model in a fully supervised way, we offer an active learning
pipeline.
Here, no labels are available from the start. The program proposes a selection of
images, which would contribute the most to improvement of the model.
These images should be labelled by a human.
For an elaborated introduction to the areas of active learning, we recommend to
read our paper.

Active learning in `wildlife-ml` is realized over the `ActiveLearner` object.
Minimally the `ActiveLearner` needs:

- `trainer`: `WildlifeTrainer` as model container and handler.
- `pool_dataset`: `WildlifeDataset` without labels
- `label_file_path`: Path pointing to the file where labels should be saved
- `label_file_path`: Path pointing to the file where labels should be saved
- `conf_threshold`: Minimal confidence for bounding boxes in the MD file
- `empty_class_id`: idx of the class that symbolizes an empty image

In case a test set exists, it is recommended to fill the `test_dataset` argument
as well.
An example is:

```python
from wildlifeml import ActiveLearner

a_learner = ActiveLearner(
    trainer=trainer,
    pool_dataset=train_dataset,
    label_file_path='<path_to_labels.csv>',
    conf_threshold=0.1,
    empty_class_id=10,
    test_dataset=test_dataset
)
```

Execution of the active learning loop is triggered over:

```python
a_learner.run()
```

After `.run()` has finished, the directory `active-wildlife` will contain a directory
`images` and a file `active_labels.csv`.
The task of the user is now to derive a label for the images in the `images` directory
and enter the corresponding label into the `csv` file.
After the user has finished labelling the images, the `csv` file should be saved
and the above `run` function should be invoked again.
The manually derived labels are appended to the training process and incorporated into
the full procedure.
Now, the program will propose another round of images to label.
The user should abort the active learning procedure if he/she/they are satisfied with
the model performance of the test set.
