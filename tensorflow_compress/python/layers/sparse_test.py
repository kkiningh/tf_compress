"""Tests for sparse layers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

from tensorflow_compress.python.layers import sparse


class SparseLayer(test.TestCase):
    def testSparseLayer(self):
        with self.test_session() as sess:
            x = random_ops.random_uniform((5, 2), seed=1)
            y = sparse.masked_dense(
                x, 10, kernel_initializer=init_ops.zeros_initializer(),
                activation=nn_ops.relu, name='my_masked_dense')
            self.assertListEqual([5, 10], y.get_shape().as_list())
            self.assertAllEqual([0.0], sess.run(y))


if __name__ == '__main__':
    test.main()
