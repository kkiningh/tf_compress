"""Masked layer implementations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

MASK_COLLECTION = 'masks'
THRESHOLD_COLLECTION = 'thresholds'
MASKED_WEIGHT_COLLECTION = 'masked_weights'
WEIGHT_COLLECTION = 'kernel'
MASKED_WEIGHT_NAME = 'weights/masked_weight'


class MaskedDense(base.Layer):
    """Dense layer with masked weights

    Functionally similar to tf.layers.Dense, but with masked weights

    Arguments:
        units: Integer or Long, dimensionality of the output space
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MaskedDense, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})

        self.kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        self.mask = self.add_variable(
            'mask',
            shape=[input_shape[-1].value, self.units],
            initializer=init_ops.ones_initializer(),
            trainable=False,
            dtype=self.dtype)

        self.threshold = self.add_variable(
            'threshold',
            shape=[],
            initializer=init_ops.zeros_initializer(),
            trainable=False,
            dtype=self.dtype)

        # Add masked_weights in the weights namescope so as to make it easier
        # for the quantization library to add quant ops.
        self.masked_kernel = math_ops.multiply(
            self.mask, self.kernel, name=MASKED_WEIGHT_NAME)

        ops.add_to_collection(MASK_COLLECTION, self.mask)
        ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_kernel)
        ops.add_to_collection(THRESHOLD_COLLECTION, self.threshold)
        ops.add_to_collection(WEIGHT_COLLECTION, self.kernel)

        if self.use_bias:
            self.bias = self.add_variable(
                'bias',
                shape=[self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        if len(output_shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, self.masked_kernel,
                                   [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.masked_kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


def masked_dense(inputs,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 reuse=None):
    """Functional interface for MaskedDense class.

    Returns:
        Output tensor the same shape as `inputs` except the last dimension is of
        size `units`.
    """
    layer = MaskedDense(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse)
    return layer.apply(inputs)
