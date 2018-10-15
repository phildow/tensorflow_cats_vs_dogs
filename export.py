import tensorflow as tf
import numpy as np
import argparse

# local
from models import model

# saving the model

def serving_input_receiver_fn(params):
  dimension = [None, params['target_dim'], params['target_dim'], 3]

  inputs = {
    'image': tf.placeholder(tf.float32, dimension, name='image'),
  }

  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def save_model(model_dir, output_dir, dims):
  estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)
  input_params = {'target_dim': dims}

  estimator.export_saved_model(output_dir, lambda:serving_input_receiver_fn(input_params))

# running the script

def generate_arg_parser():
  parser = argparse.ArgumentParser(description="Train a basic DNN on the MNIST digits dataset")

  parser.add_argument(
    '--model-dir',
    type=str,
    help='The model directory where model outputs reside'
  )

  parser.add_argument(
    '--output-dir',
    type=str,
    help='The directory to which the exported model will be saved',
    required=True
  )

  parser.add_argument(
    '--dims',
    type=int,
    help='The image dimensions along a single axis',
    required=True
  )

  parser.add_argument(
    '--verbose',
    action='store_true',
    help="Sets logging verbose threshold to info"
  )

  return parser

if __name__ == '__main__':
  parser = generate_arg_parser()
  args = parser.parse_args()

  if args.verbose:
    print('Running in verbose mode')
    tf.logging.set_verbosity(tf.logging.INFO)

  save_model(args.model_dir, args.output_dir, args.dims)