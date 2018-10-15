import tensorflow as tf
import numpy as np

# training input dataset and parsing

def _parser(protobuf, params):
  features = {
      'label': tf.FixedLenFeature([], tf.int64, default_value=0),
      'image': tf.FixedLenFeature([], tf.string, default_value="")
  }

  dimension = [params['target_dim'], params['target_dim'], 3]
  
  parsed_features = tf.parse_single_example(protobuf, features)
  image = tf.decode_raw(parsed_features["image"], tf.float64)
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, dimension)
  label = tf.cast(parsed_features["label"], tf.int32)
  
  return {'image':image}, label

def input_fn(filenames, batch_size=32, num_epochs=1000, params={}):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x:_parser(x, params))
  dataset = dataset.shuffle(1024)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)
  
  return dataset
