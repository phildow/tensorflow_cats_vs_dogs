import numpy as np
import argparse
import glob
import os

from PIL import Image

# resize the images

def _resize_image(img, target_dim):
  """
  Center crops and resizes an image to target_dim
  """

  width, height = img.size
  
  if width < target_dim or height < target_dim:
    return None

  if height < width:
    crop_dim = height 
  else:
    crop_dim = width

  left = width/2 - crop_dim/2
  upper = height/2 - crop_dim/2
  right = left + crop_dim
  lower = upper + crop_dim

  crop = (left, upper, right, lower)

  img = img.crop(crop)
  img = img.resize((target_dim,target_dim))

  return img

def _resize_batch(filenames, output_dir, target_dim):
  """
  Resizes a set of filenames into the output_dir, appending the basename of the input
  """

  for filename in filenames:
    img = Image.open(filename)
    img = _resize_image(img, target_dim)

    if img is None:
      continue

    output_filename = os.path.join(output_dir, os.path.basename(filename))
    img.save(output_filename, 'JPEG', quality=80)

def resize_images(input_dir, output_dir, target_dim):

  # input directories
  train_input = os.path.join(input_dir, 'train')
  validation_input = os.path.join(input_dir, 'validation')
  test_input = os.path.join(input_dir, 'test')

  # output directories
  train_output = os.path.join(output_dir, 'train')
  validation_output = os.path.join(output_dir, 'validation')
  test_output = os.path.join(output_dir, 'test')

  # create root output directories
  if not os.path.isdir(train_output):
    os.mkdir(train_output)
  if not os.path.isdir(validation_output):
    os.mkdir(validation_output)
  if not os.path.isdir(test_output):
    os.mkdir(test_output)

  # resize training images
  
  # cats
  train_input_cats = glob.glob( os.path.join(os.path.join(train_input, 'cats'), '*.jpg') )
  train_output_cats = os.path.join(os.path.join(train_output, 'cats'))
  
  if not os.path.isdir(train_output_cats):
    os.mkdir(train_output_cats)
  
  _resize_batch(train_input_cats, train_output_cats, target_dim)

  # dogs
  train_input_dogs = glob.glob( os.path.join(os.path.join(train_input, 'dogs'), '*.jpg') )
  train_output_dogs = os.path.join(os.path.join(train_output, 'dogs'))
  
  if not os.path.isdir(train_output_dogs):
    os.mkdir(train_output_dogs)
  
  _resize_batch(train_input_dogs, train_output_dogs, target_dim)

  # resize validation images

  # cats
  validation_input_cats = glob.glob( os.path.join(os.path.join(validation_input, 'cats'), '*.jpg') )
  validation_output_cats = os.path.join(os.path.join(validation_output, 'cats'))
  
  if not os.path.isdir(validation_output_cats):
    os.mkdir(validation_output_cats)
  
  _resize_batch(validation_input_cats, validation_output_cats, target_dim)

  # dogs
  validation_input_dogs = glob.glob( os.path.join(os.path.join(validation_input, 'dogs'), '*.jpg') )
  validation_output_dogs = os.path.join(os.path.join(validation_output, 'dogs'))
  
  if not os.path.isdir(validation_output_dogs):
    os.mkdir(validation_output_dogs)
  
  _resize_batch(validation_input_dogs, validation_output_dogs, target_dim)

  # resize test images

  # cats
  test_input_cats = glob.glob( os.path.join(os.path.join(test_input, 'cats'), '*.jpg') )
  test_output_cats = os.path.join(os.path.join(test_output, 'cats'))
  
  if not os.path.isdir(test_output_cats):
    os.mkdir(test_output_cats)
  
  _resize_batch(test_input_cats, test_output_cats, target_dim)

  # dogs
  test_input_dogs = glob.glob( os.path.join(os.path.join(test_input, 'dogs'), '*.jpg') )
  test_output_dogs = os.path.join(os.path.join(test_output, 'dogs'))
  
  if not os.path.isdir(test_output_dogs):
    os.mkdir(test_output_dogs)
  
  _resize_batch(test_input_dogs, test_output_dogs, target_dim)

  return

# running the script

def generate_arg_parser():
  parser = argparse.ArgumentParser(description="Resize the provided images to a standard size")

  parser.add_argument(
    '--input-dir',
    type=str,
    help='The directory where the input images reside'
  )

  parser.add_argument(
    '--output-dir',
    type=str,
    help='The directory where the resized images will be saved',
    required=True
  )

  parser.add_argument(
    '--target-dim',
    type=int,
    help='The target dimension along a size axels. Resized images will be a square in this dimension. Images with an axis smaller than the target dimension will be ignored',
    required=True
  )

  return parser

if __name__ == '__main__':
  parser = generate_arg_parser()
  args = parser.parse_args()

  resize_images(args.input_dir, args.output_dir, args.target_dim)