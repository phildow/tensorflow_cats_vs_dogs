import tensorflow as tf
import numpy as np
import argparse

# local
from input_functions import training_input
from input_functions import evaluation_input
from models import model

# model training

def train_model(train_filenames, validate_filenames, batch_size, num_epochs, dims, model_dir):
  estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)
  input_params = {'target_dim': dims}

  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda:training_input.input_fn(train_filenames, batch_size, num_epochs, input_params),
    max_steps=num_epochs)

  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda:evaluation_input.input_fn(validate_filenames, batch_size, input_params),
    steps=100)

  results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  print(results)

# Running the script

def generate_arg_parser():
  parser = argparse.ArgumentParser(description="Train a basic DNN on the MNIST digits dataset")

  parser.add_argument(
    '--train-input',
    type=str,
    help='The TFRecord file or files to train on',
    required=True
  )

  parser.add_argument(
    '--validate-input',
    type=str,
    help='The TFRecord file or files to validate on',
    required=True
  )

  parser.add_argument(
    '--batch',
    type=int,
    help='The batch size to train on',
    required=True
  )

  parser.add_argument(
    '--epochs',
    type=int,
    help='The number of epochs to train for',
    required=True
  )

  parser.add_argument(
    '--dims',
    type=int,
    help='The image dimensions along a single axis',
    required=True
  )

  parser.add_argument(
    '--model-dir',
    type=str,
    help='The model directory where model outputs reside, or a tmp directory if not specified'
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

  train_filenames = [s.strip() for s in args.train_input.split(',')]
  validate_filenames = [s.strip() for s in args.validate_input.split(',')]

  train_model(
    train_filenames=train_filenames,
    validate_filenames=validate_filenames,
    batch_size=args.batch, 
    num_epochs=args.epochs, 
    dims=args.dims,
    model_dir=args.model_dir)