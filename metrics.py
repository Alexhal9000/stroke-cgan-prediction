import sys

# third party
import tensorflow.keras as keras
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import Callback
import tensorflow as tf


class Dice(object):
    """
    Slightly adapted version of the dice loss used by Dalca et al. in their neuron libray:

    Dalca AV, Guttag J, Sabuncu MR
    Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
    CVPR 2018
    License: GPLv3

    Dice of two Tensors.

    Tensors should either be:
    - probabilitic for each label
        i.e. [batch_size, *vol_size, nb_labels], where vol_size is the size of the volume (n-dims)
        e.g. for a 2D vol, y has 4 dimensions, where each entry is a prob for that voxel
    - max_label
        i.e. [batch_size, *vol_size], where vol_size is the size of the volume (n-dims).
        e.g. for a 2D vol, y has 3 dimensions, where each entry is the max label of that voxel

    Variables:
        nb_labels: optional numpy array of shape (L,) where L is the number of labels
            if not provided, all non-background (0) labels are computed and averaged
        weights: optional numpy array of shape (L,) giving relative weights of each label
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft

    Usage:
        diceloss = metrics.dice(weights=[1, 2, 3])
        model.compile(diceloss, ...)

    Test:
        import keras.utils as nd_utils
        reload(nrn_metrics)
        weights = [0.1, 0.2, 0.3, 0.4, 0.5]
        nb_labels = len(weights)
        vol_size = [10, 20]
        batch_size = 7

        dice_loss = metrics.Dice(nb_labels=nb_labels).loss
        dice = metrics.Dice(nb_labels=nb_labels).dice
        dice_wloss = metrics.Dice(nb_labels=nb_labels, weights=weights).loss

        # vectors
        lab_size = [batch_size, *vol_size]
        r = nd_utils.to_categorical(np.random.randint(0, nb_labels, lab_size), nb_labels)
        vec_1 = np.reshape(r, [*lab_size, nb_labels])
        r = nd_utils.to_categorical(np.random.randint(0, nb_labels, lab_size), nb_labels)
        vec_2 = np.reshape(r, [*lab_size, nb_labels])

        # get some standard vectors
        tf_vec_1 = tf.constant(vec_1, dtype=tf.float32)
        tf_vec_2 = tf.constant(vec_2, dtype=tf.float32)

        # compute some metrics
        res = [f(tf_vec_1, tf_vec_2) for f in [dice, dice_loss, dice_wloss]]
        res_same = [f(tf_vec_1, tf_vec_1) for f in [dice, dice_loss, dice_wloss]]

        # tf run
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(res)
            sess.run(res_same)
            print(res[2].eval())
            print(res_same[2].eval())
    """

    def __init__(self, nb_labels,
                 weights=None,
                 input_type='prob',
                 dice_type='soft',
                 approx_hard_max=True,
                 vox_weights=None,
                 crop_indices=None,
                 area_reg=0.1):  # regularization for bottom of Dice coeff
        """
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft
        approx_hard_max - see note below

        Note: for hard dice, we grab the most likely label and then compute a
        one-hot encoding for each voxel with respect to possible labels. To grab the most
        likely labels, argmax() can be used, but only when Dice is used as a metric
        For a Dice *loss*, argmax is not differentiable, and so we can't use it
        Instead, we approximate the prob->one_hot translation when approx_hard_max is True.
        """

        self.nb_labels = nb_labels
        self.weights = None if weights is None else K.variable(weights)
        self.vox_weights = None if vox_weights is None else K.variable(vox_weights)
        self.input_type = input_type
        self.dice_type = dice_type
        self.approx_hard_max = approx_hard_max
        self.area_reg = area_reg
        self.crop_indices = crop_indices

    def dice(self, y_true, y_pred):
        """
        compute dice for given Tensors

        """
        
        if self.input_type == 'prob':
            # We assume that y_true is probabilistic, but just in case:
            y_true /= K.sum(y_true, axis=-1, keepdims=True)
            y_true = K.clip(y_true, K.epsilon(), 1)

            # make sure pred is a probability
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1)
        

        # Prepare the volumes to operate on
        # If we're doing 'hard' Dice, then we will prepare one-hot-based matrices of size
        # [batch_size, nb_voxels, nb_labels], where for each voxel in each batch entry,
        # the entries are either 0 or 1
        if self.dice_type == 'hard':

            # if given predicted probability, transform to "hard max""
            if self.input_type == 'prob':
                if self.approx_hard_max:
                    y_pred_op = _hard_max(y_pred, axis=-1)
                    y_true_op = _hard_max(y_true, axis=-1)
                else:
                    y_pred_op = _label_to_one_hot(K.argmax(y_pred, axis=-1), self.nb_labels)
                    y_true_op = _label_to_one_hot(K.argmax(y_true, axis=-1), self.nb_labels)

            # if given predicted label, transform to one hot notation
            else:
                assert self.input_type == 'max_label'
                y_pred_op = _label_to_one_hot(y_pred, self.nb_labels)
                y_true_op = _label_to_one_hot(y_true, self.nb_labels)

        # If we're doing soft Dice, require prob output, and the data already is as we need it
        # [batch_size, nb_voxels, nb_labels]
        else:
            assert self.input_type == 'prob', "cannot do soft dice with max_label input"
            y_pred_op = y_pred
            y_true_op = y_true

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        sum_dim = np.arange(1, K.ndim(y_true_op) - 1)
        top = 2 * K.sum(y_true_op * y_pred_op, sum_dim)
        bottom = K.sum(K.square(y_true_op), sum_dim) + K.sum(K.square(y_pred_op), sum_dim)
        # make sure we have no 0s on the bottom. K.epsilon()
        bottom = K.maximum(bottom, self.area_reg)
        return top / (bottom+0.0000001)
        

    def mean_dice(self, y_true, y_pred):
        """ weighted mean dice across all patches and labels """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_metric *= self.weights
        if self.vox_weights is not None:
            dice_metric *= self.vox_weights

        # return one minus mean dice as loss
        mean_dice_metric = K.mean(dice_metric)
        return mean_dice_metric

    def loss(self, y_true, y_pred):
        """ the loss. Assumes y_pred is prob (in [0,1] and sum_row = 1) """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # loss
        dice_loss = 1 - dice_metric

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_loss *= self.weights

        # return one minus mean dice as loss
        mean_dice_loss = K.mean(dice_loss)
        return mean_dice_loss



    def precision(self, y_true, y_pred):
        """
        compute dice for given Tensors

        """

        if self.input_type == 'prob':
            # We assume that y_true is probabilistic, but just in case:
            y_true /= K.sum(y_true, axis=-1, keepdims=True)
            y_true = K.clip(y_true, K.epsilon(), 1)

            # make sure pred is a probability
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1)

        # Prepare the volumes to operate on
        # If we're doing 'hard' Dice, then we will prepare one-hot-based matrices of size
        # [batch_size, nb_voxels, nb_labels], where for each voxel in each batch entry,
        # the entries are either 0 or 1
        if self.dice_type == 'hard':

            # if given predicted probability, transform to "hard max""
            if self.input_type == 'prob':
                if self.approx_hard_max:
                    y_pred_op = _hard_max(y_pred, axis=-1)
                    y_true_op = _hard_max(y_true, axis=-1)
                else:
                    y_pred_op = _label_to_one_hot(K.argmax(y_pred, axis=-1), self.nb_labels)
                    y_true_op = _label_to_one_hot(K.argmax(y_true, axis=-1), self.nb_labels)

            # if given predicted label, transform to one hot notation
            else:
                assert self.input_type == 'max_label'
                y_pred_op = _label_to_one_hot(y_pred, self.nb_labels)
                y_true_op = _label_to_one_hot(y_true, self.nb_labels)

        # If we're doing soft Dice, require prob output, and the data already is as we need it
        # [batch_size, nb_voxels, nb_labels]
        else:
            assert self.input_type == 'prob', "cannot do soft dice with max_label input"
            y_pred_op = y_pred
            y_true_op = y_true

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        sum_dim = np.arange(1, K.ndim(y_true_op) - 1)
        top = K.sum(y_true_op * y_pred_op, sum_dim)
        bottom = K.sum(K.abs(y_pred_op), sum_dim)
        # make sure we have no 0s on the bottom. K.epsilon()
        bottom = K.maximum(bottom, self.area_reg)
        return K.mean(top / bottom)

