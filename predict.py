import argparse

import tensorflow as tf
import numpy as np
import scipy.misc

# local
from input_functions import prediction_input
from models import model

# single prediction

def predict_single(model_input, model_dir, params=None):
  predict_fn = tf.contrib.predictor.from_saved_model(model_dir)
  
  shape = (-1, params['target_dim'], params['target_dim'], 3)

  image = scipy.misc.imread(model_input).astype(np.float64).reshape(-1)
  image = np.divide(image, 255.)
  image = image.reshape(shape)

  predictions = predict_fn({
    'image': image
  })

  if tf.logging.get_verbosity() == tf.logging.INFO:
    print(predictions)
  else:
    print('CAT' if predictions['class'] == 0 else 'DOG')

# tf records prediction

def predict_tfrecords(predict_filenames, batch_size, model_dir, params=None):
  estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)
  results = estimator.predict(input_fn=lambda:prediction_input.input_fn(predict_filenames, batch_size, params))

  for pred_dict in results:
    #template = ('\nPrediction is "{}" ({:.3f})')
    class_id = pred_dict['class']
    #probs = pred_dict['probabilities'][class_id]
    #print(template.format(class_id, probs))
    print(class_id)

# Running the script

def generate_arg_parser():
  parser = argparse.ArgumentParser(description='''
    Make a prediction. 
    Specify either both --predict-tfrecords and model-dir to make bulk predictions from tf records on a trained model
    or --predict-single and --saved-model to make a prediction from a single input using the exported model.
    ''')

  parser.add_argument(
    '--predict-tfrecords',
    type=str,
    help='The TFRecord file to run predictions on.'
  )

  parser.add_argument(
    '--predict-single',
    type=str,
    help='A filename or other data that the model will run prediction on.'
  )

  parser.add_argument(
    '--model-dir',
    type=str,
    help='The model directory where model outputs reside'
  )

  parser.add_argument(
    '--saved-model',
    type=str,
    help='The model directory where a saved model reside. Pass the folder that is created by export.py'
  )

  parser.add_argument(
    '--batch',
    type=int,
    help='The batch size to predict. Pass in 1 if you use --predict-single',
    required=True
  )

  parser.add_argument(
    '--verbose',
    action='store_true',
    help="Sets logging verbose threshold to info"
  )

  # custom parser arguments

  parser.add_argument(
    '--dims',
    type=int,
    help='The image dimensions along a single axis',
    required=True
  )

  return parser

if __name__ == '__main__':
  parser = generate_arg_parser()
  args = parser.parse_args()

  # verbosity

  if args.verbose:
    print('Running in verbose mode')
    tf.logging.set_verbosity(tf.logging.INFO)

  # parameters

  params = {
    'target_dim': args.dims
  }

  # single or tf record prediction

  if args.predict_single is not None:
    predict_single(args.predict_single, args.saved_model, params)
  else:
    predict_tfrecords(
      predict_filenames=[args.predict_tfrecords], 
      batch_size=args.batch,
      model_dir=args.model_dir,
      params=params)

