# https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow

import keras.backend as K
import tensorflow as tf
def weighted_binary_crossentropy(pos_weight=1.):
    """ Returns a Weighted binary cross entropy loss function between an output tensor and a target tensor.
    # Arguments
        pos_weight: A coefficient to use on the positive examples.
    # Returns
        A loss function supposed to be used in model.compile().
        This same loss function weighted_binary_crossentropy() (note with the parentheses) should be
        passed to tf.keras.models.load_model when loading the network, e.g. as in convertH5toPB.py:
                custom_objects={"_weighted_binary_crossentropy": weighted_binary_crossentropy()}
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    """

    def _to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)

    def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))
        target = tf.dtypes.cast(target, tf.float32)
        return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=pos_weight)

    def _weighted_binary_crossentropy(y_true, y_pred):
        """ This function is the actual loss function that model.fit uses.
        :param y_true: the ground truth expected outputs in batch
        :param y_pred: the predictions from network
        :returns: the mean over the batch
        """
        return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)

    return _weighted_binary_crossentropy # note that calling weighted_binary_crossentropy() returns the *function* _weighted_binary_crossentropy
