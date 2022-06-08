# Welcome to **Wildlife ML**!

# Workflow
##  01: Prediction of Bounding Boxes

This package includes an implementation of the
[Megadetector](https://github.com/microsoft/CameraTraps) for detecting objects in
camera traps.
It is accessible as a python object:

```python
from wildlifeml import MegaDetector

md = MegaDetector()
md.predict_directory(directory=<directory>, output_file=<output_file>)
```
`<directory>` should be a directory that contains images, where bounding boxes
should be predicted. More options and arguments are available. The results are saved
in `.json` file.

We also offer a CLI option:

```shell
wildlifeml-get-bbox -d <directory>
```

Access help over the `--help` flag.

## 02: Creating a Wildlife Dataset

### Folder structure

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

`images_megadetector.json` is the output from the `MegaDetector`. Images are contained
in one flattened directory.

If your dataset follows the structure of separating classes by directories, we also
offer a `DatasetConverter` for converting the file structure to the one above.

### Splitting the dataset

For splitting the dataset into train and test, one can call `do_train_split`.
The result is a tuple of keys, that corresponds to ids that should be in the train and
test set respectively.

```python
from wildlifeml.data import do_train_split

train_keys, test_keys = do_train_split(
    label_file_path=<path_to_labels.csv>,
    detector_file_path=<path_to_images_megadetector.json>,
    splits=(0.7, 0.3)
)
```

### Initializing a Dataset

Our dataset builds on the Keras `Sequence` utility and thus supports multi-threaded
loading during training. For defining the content of the dataset, a list of keys
pointing to the respective images shall be provided. This is e.g. the output of
`do_train_split`. Furthermore, `shuffle` can be toggled and `batch_size` defines the
number of samples on each call of the dataset.


```python
from wildlifeml.data import WildlifeDataset

wd = WildlifeDataset(
    keys=train_keys,
    label_file_path=<path_to_labels.csv>,
    detector_file_path=<path_to_images_megadetector.json>,
    batch_size=8
)
```

The dataset object also crops the images according to their bounding boxes.
