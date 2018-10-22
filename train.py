import tensorflow as tf
import numpy as np
import argparse

# local
from input_functions import training_input
from input_functions import evaluation_input
from models import model

# model training

def train_model(train_filenames, validate_filenames, batch_size, num_epochs, model_dir, params):
  """
  Trains a model on a set of training records and validates it against a set of validation records

  Args:

  1. train_filenames: a list of TF Record filepaths for validation
  2. validate_filenames: a list of TF Record filepaths for validation
  3. batch_size: the batch size to use for training
  4. num_epochs: the number of epochs to use for training
  5. model_dir: the directory where model state (checkpoints) should be saved during training
  6. params: a dictionary of model parameters:
      - target_dim: the size of an image along a single dimension, e.g. 128, 224, etc
  """
  
  estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)
  input_params = params

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
  parser = argparse.ArgumentParser(description="Train a mobilenet classifier on the kaggle dogs vs cats dataset")

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

  # parameters

  train_filenames = [s.strip() for s in args.train_input.split(',')]
  validate_filenames = [s.strip() for s in args.validate_input.split(',')]

  params = {
    'target_dim': args.dims
  }

  train_model(
    train_filenames=train_filenames,
    validate_filenames=validate_filenames,
    batch_size=args.batch, 
    num_epochs=args.epochs, 
    model_dir=args.model_dir,
    params=params)