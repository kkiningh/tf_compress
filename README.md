Tensoflow Compress
===

General compression library for tensorflow models.

## Installation

The easiest way to install is using pip.

```python
# sudo apt-get install bazel git python-pip
git clone https://github.com/kkiningh/tf_compress.git
cd tf_compress
pip install --user .
```

You can also build a python package from source.
This requires [Bazel](https://bazel.build/).

```python
# sudo apt-get install bazel git python-pip
git clone https://github.com/kkiningh/tf_compress.git
cd tf_compress
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```

## Usage

Access the library using

```python
import tensorflow_compress as tfc
```

## Acknowledgements
This library was inspired by [tf.contrib.model_pruning](https://github.com/tensorflow/tensorflow/tree/r1.8/tensorflow/contrib/model_pruning).
