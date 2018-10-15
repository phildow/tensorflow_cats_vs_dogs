import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def model_fn(features, labels, mode, params):
  
  MOBILENET = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'

  # build model layers

  module = hub.Module(MOBILENET)
  feature_vector = module(features["image"])

  logits = tf.layers.dense(feature_vector, 1, name='logit')
  probabilities = tf.nn.sigmoid(logits, name='sigmoid')

  # prepare predictions

  predictions = {
    'probability': probabilities,
    'class': tf.to_int32(probabilities > 0.5)
  }
  prediction_output = tf.estimator.export.PredictOutput({
    'probability': probabilities,
    'class': tf.to_int32(probabilities > 0.5)
  })

  # return an estimator spec for prediction before computing a loss

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode, 
      predictions=predictions,
      export_outputs={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output
      })

  # calculate loss

  labels = tf.reshape(labels, [-1,1])
  labels = tf.cast(labels, tf.float32)

  loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=labels,
    logits=logits
  )

  # calculate accuracy metric

  accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["class"], name='accuracy')

  if mode == tf.estimator.ModeKeys.TRAIN:

    # generate some summary info

    tf.summary.scalar('average_loss', loss)
    tf.summary.scalar('accuracy', accuracy[1])

    # prepare an optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())

    # return an estimator spec

    return tf.estimator.EstimatorSpec(
      mode=mode, 
      loss=loss, 
      train_op=train_op)
  
  if mode == tf.estimator.ModeKeys.EVAL:

    # add evaluation metrics
    
    eval_metric_ops = {
      "accuracy": accuracy
    }

    # return an estimator spec

    return tf.estimator.EstimatorSpec(
      mode=mode, 
      loss=loss, 
      eval_metric_ops=eval_metric_ops)