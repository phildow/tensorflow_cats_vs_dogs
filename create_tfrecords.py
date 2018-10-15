import argparse
import glob
import os

import tensorflow as tf
import numpy as np
import scipy.misc

CAT_LABEL = 0
DOG_LABEL = 1

# Convert data to features

def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Create the tf records file

def write_tfrecords(input_images, label, filename, predict_set):
  # open the TFRecords file
  writer = tf.python_io.TFRecordWriter(filename)

  for image_filename in input_images:
    image = scipy.misc.imread(image_filename).astype(np.float64).reshape(-1)
    image = np.divide(image, 255.)
    image = tf.compat.as_bytes(image.tostring())
    
    if predict_set == False:
      feature = {
        'label': _int64_feature(label),
        'image': _bytes_feature(image)
      }
    else:
      feature = {
        'filename': _bytes_feature(tf.compat.as_bytes(image_filename, encoding='utf-8')),
        'image': _bytes_feature(image)
      }

    # create a protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # serialize to string and write on the file
    writer.write(example.SerializeToString())

  writer.close()

def create_tfrecords(input_dir, output_dir, predict_set=False):
  cats = glob.glob( os.path.join(os.path.join(input_dir, 'cats'), '*.jpg') )
  dogs = glob.glob( os.path.join(os.path.join(input_dir, 'dogs'), '*.jpg') )

  cats_output = os.path.join(os.path.join(output_dir, 'cats.tfrecords'))
  dogs_output = os.path.join(os.path.join(output_dir, 'dogs.tfrecords'))

  print(f'Writing {len(cats)} cats to {cats_output}')
  print(f'Writing {len(dogs)} dogs to {dogs_output}')

  write_tfrecords(cats, CAT_LABEL, cats_output, predict_set)
  write_tfrecords(dogs, DOG_LABEL, dogs_output, predict_set)

# Running the script

def generate_arg_parser():
  parser = argparse.ArgumentParser(description="Convert MNIST digits CSV data to the TFRecord format")

  parser.add_argument(
    '--input-dir',
    type=str,
    help='A directory of cats and dogs folders to create tf records from',
    required=True
  )

  parser.add_argument(
    '--output-dir',
    type=str,
    help='The directory the TFRecord files will be written to. A cats.tfrecords and dogs.tfrecords will be written.',
    required=True
  )

  parser.add_argument(
    '--predict-set',
    type=bool,
    help='True if this set will be used for prediction, False otherwise',
    default=False
  )

  return parser

if __name__ == '__main__':
  args = generate_arg_parser().parse_args()
  create_tfrecords(args.input_dir, args.output_dir, args.predict_set)