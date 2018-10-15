import tensorflow as tf
import numpy as np

# training input dataset and parsing

def _parser(protobuf, params):
  features = {
    'filename': tf.FixedLenFeature([], tf.string, default_value=""),
    'image': tf.FixedLenFeature([], tf.string, default_value="")
  }
  
  dimension = [params['target_dim'], params['target_dim'], 3]

  parsed_features = tf.parse_single_example(protobuf, features)
  image = tf.decode_raw(parsed_features["image"], tf.float64)
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, dimension)
  
  return {'image':image}

def input_fn(filenames, batch_size=32, params={}):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x:_parser(x, params))
  dataset = dataset.batch(batch_size)

  return dataset
