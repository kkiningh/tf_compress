"""Tests for sparse layers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_compress as tfc

class SparseLayer(tf.test.TestCase):
    def testSparseLayer(self):
        with self.test_session() as sess:
            pass


if __name__ == '__main__':
  tf.test.main()
