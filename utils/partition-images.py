import argparse
import shutil
import os

CLASS_COUNT = 12500

def partition_dataset(original_dir, target_dir, partition):

  training_count, validation_count, test_count = partition[0], partition[1], partition[2]
  
  train_start = 0
  train_end = training_count
  
  validation_start = training_count
  validation_end = training_count + validation_count

  test_start = training_count + validation_count
  test_end = training_count + validation_count + test_count

  # create target/train, target/validation, target/test dirs

  train_dir = os.path.join(target_dir, 'train')
  if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

  validation_dir = os.path.join(target_dir, 'validation')
  if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

  test_dir = os.path.join(target_dir, 'test')
  if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

  # create target/train/cats, target/train/dogs dirs

  train_cats_dir = os.path.join(train_dir, 'cats')
  if not os.path.isdir(train_cats_dir):
    os.mkdir(train_cats_dir)

  train_dogs_dir = os.path.join(train_dir, 'dogs')
  if not os.path.isdir(train_dogs_dir):
    os.mkdir(train_dogs_dir)

  # create target/validation/cats, target/validation/dogs dirs

  validation_cats_dir = os.path.join(validation_dir, 'cats')
  if not os.path.isdir(validation_cats_dir):
    os.mkdir(validation_cats_dir)

  validation_dogs_dir = os.path.join(validation_dir, 'dogs')
  if not os.path.isdir(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

  # create target/test/cats, target/test/dogs dirs

  test_cats_dir = os.path.join(test_dir, 'cats')
  if not os.path.isdir(test_cats_dir):
    os.mkdir(test_cats_dir)

  test_dogs_dir = os.path.join(test_dir, 'dogs')
  if not os.path.isdir(test_dogs_dir):
    os.mkdir(test_dogs_dir)

  # partition cats and copy into train, validation, and test dirs

  fnames = ['cat.{}.jpg'.format(i) for i in range(train_start, train_end)]
  for fname in fnames:
    src = os.path.join(original_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

  fnames = ['cat.{}.jpg'.format(i) for i in range(validation_start, validation_end)]
  for fname in fnames:
    src = os.path.join(original_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
      
  fnames = ['cat.{}.jpg'.format(i) for i in range(test_start, test_end)]
  for fname in fnames:
    src = os.path.join(original_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
  # partition dogs and copy into train, validation, and test dirs 
      
  fnames = ['dog.{}.jpg'.format(i) for i in range(train_start, train_end)]
  for fname in fnames:
    src = os.path.join(original_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

  fnames = ['dog.{}.jpg'.format(i) for i in range(validation_start, validation_end)]
  for fname in fnames:
    src = os.path.join(original_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
      
  fnames = ['dog.{}.jpg'.format(i) for i in range(test_start, test_end)]
  for fname in fnames:
    src = os.path.join(original_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)   

# running the script

def generate_arg_parser():
  parser = argparse.ArgumentParser(description="Resize the provided images to a standard size")

  parser.add_argument(
    '--input-dir',
    type=str,
    help='The directory where the original training images reside'
  )

  parser.add_argument(
    '--output-dir',
    type=str,
    help='The directory where the train, validation, and test sets will be created',
    required=True
  )

  parser.add_argument(
    '--count',
    type=int,
    help='The number of each class of images to take. There are 12,500 each of cat and dog images, so some number less than or equal to that.',
    required=True
  )

  parser.add_argument(
    '--partition',
    type=str,
    help='Forward slash separated ratio of train,validation,test distribution, numbers should sum to 100, e.g. 80/10/10',
    required=True
  )

  return parser

if __name__ == '__main__':
  parser = generate_arg_parser()
  args = parser.parse_args()

  count = args.count
  partition = [int(x.strip())/100 for x in args.partition.split('/')]
  
  assert(count <= CLASS_COUNT)
  assert(len(partition) == 3)
  assert(sum(partition) == 1)

  partition_count = [int(count*p) for p in partition]

  print(f'partitioning dataset into {partition_count[0]} training / {partition_count[1]} validation / {partition_count[2]} test examples')

  partition_dataset(args.input_dir, args.output_dir, partition_count)