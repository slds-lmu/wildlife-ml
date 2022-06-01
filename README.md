# Welcome to **Wildlife ML**!

# Prediction of Bound Boxes

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
