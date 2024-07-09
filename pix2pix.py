# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pix2pix.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
from DepthwiseConv3D import DepthwiseConv3D

import tensorflow as tf
from tensorflow.keras.layers import *

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 400, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_string('path', None, 'Path to the data folder')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')

IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE


def load(image_file):
  """Loads the image and generates input and target image.

  Args:
    image_file: .jpeg file

  Returns:
    Input image, target image
  """
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function
def random_jitter(input_image, real_image):
  """Random jittering.

  Resizes to 286 x 286 and then randomly crops to IMG_HEIGHT x IMG_WIDTH.

  Args:
    input_image: Input Image
    real_image: Real Image

  Returns:
    Input Image, real image
  """
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def create_dataset(path_to_train_images, path_to_test_images, buffer_size,
                   batch_size):
  """Creates a tf.data Dataset.

  Args:
    path_to_train_images: Path to train images folder.
    path_to_test_images: Path to test images folder.
    buffer_size: Shuffle buffer size.
    batch_size: Batch size

  Returns:
    train dataset, test dataset
  """
  train_dataset = tf.data.Dataset.list_files(path_to_train_images)
  train_dataset = train_dataset.shuffle(buffer_size)
  train_dataset = train_dataset.map(
      load_image_train, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.list_files(path_to_test_images)
  test_dataset = test_dataset.map(
      load_image_test, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size)

  return train_dataset, test_dataset


class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def downsample(filters, size, strid=2, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.

  Conv2D => Batchnorm => LeakyRelu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer

  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strid, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result




def downsample3D(filters, size, strid=2, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.

  Conv2D => Batchnorm => LeakyRelu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer

  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, size, strides=strid, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample3D(filters, size, strid=2, norm_type='batchnorm', apply_dropout=False):

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3DTranspose(filters, size, strides=strid,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  #result.add(DepthwiseConv3D((3, 3, 3), strides=(1, 1, 1), 
    #padding='same', depth_multiplier=1))

  return result





def upsample(filters, size, strid=2, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=strid,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  #result.add(tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), 
    #padding='same', depth_multiplier=1))

  return result



def seg_unet_3d(output_channels, input_channels, norm_type='batchnorm'):

    # encoder
    input_img = tf.keras.layers.Input(shape=[None, None, None, input_channels])
    fac=1
    """
    conv1 = Conv3D(32*fac, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv3D(32*fac, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = InstanceNormalization()(conv1) 
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    """
    
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = InstanceNormalization()(conv2) 
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    #pool2 = Dropout(0.8)(pool2)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = InstanceNormalization()(conv3) 
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    #pool3 = Dropout(0.2)(pool3)
    
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = InstanceNormalization()(conv4) 
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    #pool4 = Dropout(0.2)(pool4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.5)(conv5)

    # decoder
    up6 = UpSampling3D((2, 2, 2))(conv5)
    merge6 = concatenate([up6, conv4], axis=4)
    merge6 = InstanceNormalization()(merge6) 
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(0.5)(conv6)

    up7 = UpSampling3D((2, 2, 2))(conv6)
    merge7 = concatenate([up7, conv3], axis=4)
    merge7 = InstanceNormalization()(merge7) 
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(0.5)(conv7)
    
    up8 = UpSampling3D((2, 2, 2))(conv7)
    merge8 = concatenate([up8, conv2], axis=4)
    merge8 = InstanceNormalization()(merge8) 
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(0.5)(conv8)

    """
    up9 = UpSampling3D((2, 2, 2))(conv8)
    merge9 = concatenate([up9, conv1], axis=4)
    merge9 = InstanceNormalization()(merge9) 
    conv9 = Conv3D(32*fac, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv3D(32*fac, (3, 3, 3), activation='relu', padding='same')(conv9)
    """

    decoded = Conv3D(output_channels, (3, 3, 3), activation='softmax', padding='same')(conv8)
    return tf.keras.Model(inputs=input_img, outputs=decoded)


def seg_unet_2d(output_channels, input_channels, norm_type='batchnorm'):

    # encoder
    input_img = tf.keras.layers.Input(shape=[None, None, input_channels])
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = InstanceNormalization()(conv1) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = InstanceNormalization()(conv2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = InstanceNormalization()(conv3) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = InstanceNormalization()(conv4) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.5)(conv5)

    # decoder
    up6 = UpSampling2D((2, 2))(conv5)
    merge6 = concatenate([up6, conv4], axis=3)
    merge6 = InstanceNormalization()(merge6) 
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(0.5)(conv6)

    up7 = UpSampling2D((2, 2))(conv6)
    merge7 = concatenate([up7, conv3], axis=3)
    merge7 = InstanceNormalization()(merge7) 
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(0.5)(conv7)
    
    up8 = UpSampling2D((2, 2))(conv7)
    merge8 = concatenate([up8, conv2], axis=3)
    merge8 = InstanceNormalization()(merge8) 
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(0.5)(conv8)

    up9 = UpSampling2D((2, 2))(conv8)
    merge9 = concatenate([up9, conv1], axis=3)
    merge9 = InstanceNormalization()(merge9) 
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    decoded = Conv2D(output_channels, (3, 3), activation='softmax', padding='same')(conv9)
    return tf.keras.Model(inputs=input_img, outputs=decoded)


def seg_unet_3d_resnet(output_channels, input_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

  Returns:
    Generator model
  """
  """
  down_stack = [
      downsample3D(64, 7, 1, norm_type, apply_norm=False),  # (bs, 192, 192, 64)
      downsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 128)
      downsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 256)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 24, 24, 512)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 12, 12, 512)
  ]
  """

  up_stack = [
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 12, 12, 1024)
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample3D(512, 3, 2, norm_type),
      upsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 512)
      upsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 256)
      upsample3D(64, 3, 2, norm_type),  # (bs, 192, 192, 128)
  ]

  pre_last = upsample3D(64, 3, 2, norm_type)

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3D(
      output_channels, 7, strides=1,
      padding='same', kernel_initializer=initializer,
      activation='softmax')  # (bs, 256, 256, 3)


  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, None, input_channels])
  x = inputs

  # Downsampling through the model
  skips = []
  """
  nnn = 0
  for down in down_stack:
    x = down(x)
    if nnn>0:
      skips.append(x)
    nnn = nnn+1
  """

  x = downsample3D(64, 7, 1, norm_type, apply_norm=False)(x)
  #x = MaxPooling3D(pool_size=(2, 2, 2))(x)
  

  # ResNet
  lay = [3, 3, 5, 2]
  for nnn in range(4):
    
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    skips.append(x)
    daskip = x
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(64*(nnn+1), 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.5)(x) 
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(64*(nnn+1), 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = concat([x, daskip])


    for nn in range(lay[nnn]):
      daskip = x
      initializer = tf.random_normal_initializer(0., 0.02)
      x = tf.keras.layers.Conv3D(64*(nnn+1), 3, strides=1, padding='same', kernel_initializer=initializer)(x)
      x = InstanceNormalization()(x)
      x = tf.keras.layers.ReLU()(x)
      x = Dropout(0.5)(x) 
      initializer = tf.random_normal_initializer(0., 0.02)
      x = tf.keras.layers.Conv3D(64*(nnn+1), 3, strides=1, padding='same', kernel_initializer=initializer)(x)
      x = InstanceNormalization()(x)
      x = concat([x, daskip])

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(pre_last(x))

  return tf.keras.Model(inputs=inputs, outputs=x)



def unet_generator_3d(output_channels, input_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

  Returns:
    Generator model
  """

  down_stack = [
      downsample3D(64, 4, 2, norm_type),  # (bs, 192, 192, 64)
      downsample3D(128, 4, 2, norm_type),  # (bs, 96, 96, 128)
      downsample3D(256, 4, 2, norm_type),  # (bs, 48, 48, 256)
      downsample3D(256, 4, 2, norm_type),  # (bs, 24, 24, 512)
      downsample3D(256, 4, 2, norm_type),  # (bs, 12, 12, 512)
  ]

  up_stack = [
      upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 12, 12, 1024)
      upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 48, 48, 512)
      upsample3D(128, 4, 2, norm_type),  # (bs, 96, 96, 256)
      upsample3D(64, 4, 2, norm_type),  # (bs, 192, 192, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3DTranspose(
      output_channels, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)


  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, None, input_channels])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  #mid = tf.keras.Model(inputs=inputs, outputs=x)

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def unet_generator_3d_resnet(output_channels, input_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

  Returns:
    Generator model
  """

  down_stack = [
      downsample3D(64, 7, 1, norm_type, apply_norm=False),  # (bs, 192, 192, 64)
      downsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 128)
      downsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 256)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 24, 24, 512)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 12, 12, 512)
  ]

  up_stack = [
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 12, 12, 1024)
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 512)
      upsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 256)
      #upsample3D(64, 4, 2, norm_type),  # (bs, 192, 192, 128)
  ]

  pre_last = upsample3D(64, 3, 2, norm_type)

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3D(
      output_channels, 7, strides=1,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, None, input_channels])
  x = inputs

  # Downsampling through the model
  skips = []
  nnn = 0
  for down in down_stack:
    x = down(x)
    if nnn>0:
      skips.append(x)
    nnn = nnn+1

  skips = reversed(skips[:-1])

  # ResNet
  for nn in range(9):
    daskip = x
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(256, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.5)(x) 
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(256, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = concat([x, daskip])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(pre_last(x))

  return tf.keras.Model(inputs=inputs, outputs=x)


def unet_generator_3d_resnet_dual(output_channels, input_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

  Returns:
    Generator model
  """

  down_stack = [
      downsample3D(64, 7, 1, norm_type, apply_norm=False),  # (bs, 192, 192, 64)
      downsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 128)
      downsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 256)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 24, 24, 512)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 12, 12, 512)
  ]

  up_stack = [
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 12, 12, 1024)
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 512)
      upsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 256)
      #upsample3D(64, 4, 2, norm_type),  # (bs, 192, 192, 128)
  ]

  pre_last = upsample3D(64, 3, 2, norm_type)

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3D(
      output_channels, 7, strides=1,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, None, input_channels])
  x = inputs

  # Downsampling through the model
  skips = []
  nnn = 0
  for down in down_stack:
    x = down(x)
    if nnn>0:
      skips.append(x)
    nnn = nnn+1

  skips = reversed(skips[:-1])

  # ResNet
  for nn in range(9):
    daskip = x
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(256, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.5)(x) 
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(256, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = concat([x, daskip])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(pre_last(x))

  x = tf.concat([x[:, :, :, :, 0, tf.newaxis] ,((x[:, :, :, :, 1, tf.newaxis] + 1)/2)], 4)

  return tf.keras.Model(inputs=inputs, outputs=x)




def unet_generator_3d_resnet_lesion(output_channels, input_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

  Returns:
    Generator model
  """

  down_stack = [
      downsample3D(64, 7, 1, norm_type, apply_norm=False),  # (bs, 192, 192, 64)
      downsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 128)
      downsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 256)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 24, 24, 512)
      #downsample3D(256, 4, 2, norm_type),  # (bs, 12, 12, 512)
  ]

  up_stack = [
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 12, 12, 1024)
      #upsample3D(256, 4, 2, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample3D(256, 3, 2, norm_type),  # (bs, 48, 48, 512)
      upsample3D(128, 3, 2, norm_type),  # (bs, 96, 96, 256)
      #upsample3D(64, 4, 2, norm_type),  # (bs, 192, 192, 128)
  ]

  pre_last = upsample3D(64, 3, 2, norm_type)

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3D(
      output_channels, 7, strides=1,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, None, input_channels])
  x = inputs

  # Downsampling through the model
  skips = []
  nnn = 0
  for down in down_stack:
    x = down(x)
    if nnn>0:
      skips.append(x)
    nnn = nnn+1

  skips = reversed(skips[:-1])

  # ResNet
  for nn in range(9):
    daskip = x
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(256, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.5)(x) 
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv3D(256, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = concat([x, daskip])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(pre_last(x))

  x = (x[:, :, :, :, 0, tf.newaxis] + 1)/2

  return tf.keras.Model(inputs=inputs, outputs=x)




def unet_generator(output_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

  Returns:
    Generator model
  """

  down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),  # (bs, 192, 192, 64)
      downsample(128, 4, norm_type),  # (bs, 96, 96, 128)
      downsample(256, 4, norm_type),  # (bs, 48, 48, 256)
      downsample(512, 4, norm_type),  # (bs, 24, 24, 512)
      downsample(512, 4, norm_type),  # (bs, 12, 12, 512)
      downsample(512, 4, norm_type),  # (bs, 6, 6, 512)
      #downsample(512, 4, norm_type),  # (bs, 3, 3, 512)
  ]

  up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 6, 6, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 12, 12, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample(256, 4, norm_type),  # (bs, 48, 48, 512)
      upsample(128, 4, norm_type),  # (bs, 96, 96, 256)
      upsample(64, 4, norm_type),  # (bs, 192, 192, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)



  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, 1])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=True):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, 2, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, 2, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, 2, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)




def discriminator_3d(input_channels, norm_type='batchnorm', target=False):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

  Returns:
    Discriminator model
  """
  stride_new = (1, 2, 2)
  kernel_new = (1, 4, 4)

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, None, input_channels], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[None, None, None, input_channels], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down0 = downsample3D(64, kernel_new, stride_new, norm_type, apply_norm=False)(x) 
  down1 = downsample3D(128, kernel_new, stride_new, norm_type)(down0) 
  down2 = downsample3D(256, kernel_new, stride_new, norm_type)(down1)

  #extra
  #down4 = downsample3D(256, kernel_new, stride_new, norm_type)(down2) 
  #down5 = downsample3D(256, kernel_new, stride_new, norm_type)(down4)


  conv = tf.keras.layers.Conv3D(
      512, 4, strides=1, padding='same', kernel_initializer=initializer,
      use_bias=False)(down2)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  last = tf.keras.layers.Conv3D(
      1, 4, strides=1, padding='same',
      kernel_initializer=initializer)(leaky_relu)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)



def encoder(output_channels, norm_type='batchnorm'):

  my_ker_3d = (3, 3, 2)
  my_strid_3d = (1, 1, 2)

  down_stack = [

      downsample3D(64, my_ker_3d, my_strid_3d, norm_type, apply_norm=False),  # 32 -> 16
      downsample3D(128, my_ker_3d, my_strid_3d, norm_type),  # 16 -> 8
      downsample3D(256, my_ker_3d, my_strid_3d, norm_type),  # 8 -> 4
      #downsample3D(256, my_ker_3d, my_strid_3d, norm_type),  # 4 -> 2

  ]


  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3D(
      1, (1, 1, 1), strides=(1, 1, 1),
      padding='same', kernel_initializer=initializer,
      activation='tanh') 


  inputs = tf.keras.layers.Input(shape=[None, None, None, 1])
  x = inputs

  # Downsampling through the model
  cc = tf.math.reduce_sum(x, 3, keepdims=True)

  for down in down_stack:
    x = down(x)

  x = last(x)
  #x = tf.concat([x, cc], 3)

  return tf.keras.Model(inputs=inputs, outputs=x)



def decoder(output_channels, input_channels, norm_type='batchnorm'):

  my_ker_3d = (3, 3, 2)
  my_strid_3d = (1, 1, 2)

  up_stack = [
      upsample3D(256, my_ker_3d, my_strid_3d, norm_type, apply_dropout=True),  # (bs, 24, 24, 1024)
      upsample3D(128, my_ker_3d, my_strid_3d, norm_type),  # (bs, 96, 96, 256)
      upsample3D(64, my_ker_3d, my_strid_3d, norm_type),  # (bs, 192, 192, 128)
  ]
  

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3DTranspose(
      1, (1, 1, 1), strides=(1, 1, 1),
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)



  inputs = tf.keras.layers.Input(shape=[None, None, input_channels, 1])
  x = inputs

  # Upsampling through the model

  for up in up_stack:#down_stack:
    x = up(x)


  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



def discriminator_shallow(norm_type='batchnorm', target=True):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 2], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[None, None, 2], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  #down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(128, 4, norm_type)(down1)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)


def get_checkpoint_prefix():
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

  return checkpoint_prefix


class Pix2pix(object):
  """Pix2pix class.

  Args:
    epochs: Number of epochs.
    enable_function: If true, train step is decorated with tf.function.
    buffer_size: Shuffle buffer size..
    batch_size: Batch size.
  """

  def __init__(self, epochs, enable_function):
    self.epochs = epochs
    self.enable_function = enable_function
    self.lambda_value = 100
    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.generator = unet_generator(output_channels=1)
    self.discriminator = discriminator()
    self.checkpoint = tf.train.Checkpoint(
        generator_optimizer=self.generator_optimizer,
        discriminator_optimizer=self.discriminator_optimizer,
        generator=self.generator,
        discriminator=self.discriminator)

  def discriminator_loss(self, disc_real_output, disc_generated_output):
    real_loss = self.loss_object(
        tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = self.loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  def generator_loss(self, disc_generated_output, gen_output, target):
    gan_loss = self.loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (self.lambda_value * l1_loss)
    return total_gen_loss

  def train_step(self, input_image, target_image):
    """One train step over the generator and discriminator model.

    Args:
      input_image: Input Image.
      target_image: Target image.

    Returns:
      generator loss, discriminator loss.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = self.generator(input_image, training=True)

      disc_real_output = self.discriminator(
          [input_image, target_image], training=True)
      disc_generated_output = self.discriminator(
          [input_image, gen_output], training=True)

      gen_loss = self.generator_loss(
          disc_generated_output, gen_output, target_image)
      disc_loss = self.discriminator_loss(
          disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_loss, self.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        disc_loss, self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(
        generator_gradients, self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(
        discriminator_gradients, self.discriminator.trainable_variables))

    return gen_loss, disc_loss

  def train(self, dataset, checkpoint_pr):
    """Train the GAN for x number of epochs.

    Args:
      dataset: train dataset.
      checkpoint_pr: prefix in which the checkpoints are stored.

    Returns:
      Time for each epoch.
    """
    time_list = []
    if self.enable_function:
      self.train_step = tf.function(self.train_step)

    for epoch in range(self.epochs):
      start_time = time.time()
      for input_image, target_image in dataset:
        gen_loss, disc_loss = self.train_step(input_image, target_image)

      wall_time_sec = time.time() - start_time
      time_list.append(wall_time_sec)

      # saving (checkpoint) the model every 20 epochs
      if (epoch + 1) % 20 == 0:
        self.checkpoint.save(file_prefix=checkpoint_pr)

      template = 'Epoch {}, Generator loss {}, Discriminator Loss {}'
      print (template.format(epoch, gen_loss, disc_loss))

    return time_list


def run_main(argv):
  del argv
  kwargs = {'epochs': FLAGS.epochs, 'enable_function': FLAGS.enable_function,
            'path': FLAGS.path, 'buffer_size': FLAGS.buffer_size,
            'batch_size': FLAGS.batch_size}
  main(**kwargs)


def main(epochs, enable_function, path, buffer_size, batch_size):
  path_to_folder = path

  pix2pix_object = Pix2pix(epochs, enable_function)

  train_dataset, _ = create_dataset(
      os.path.join(path_to_folder, 'train/*.jpg'),
      os.path.join(path_to_folder, 'test/*.jpg'),
      buffer_size, batch_size)
  checkpoint_pr = get_checkpoint_prefix()
  print ('Training ...')
  return pix2pix_object.train(train_dataset, checkpoint_pr)


if __name__ == '__main__':
  app.run(run_main)
