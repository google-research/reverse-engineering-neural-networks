"""Scripts to generate vocabulary files for different datasets."""

import argparse
import os

import tensorflow_datasets as tfds
import renn
import tqdm

parser = argparse.ArgumentParser(description='Generate vocab files.')
parser.add_argument('dataset', type=str)
parser.add_argument('--save_dir', type=str, default='/tmp')
parser.add_argument('--data_dir', type=str, default='/tmp')
parser.add_argument('--vocab_size', type=int, default=2**15)


def load_dataset(name, data_dir=None):
  """Loads dataset as a generator that yields text documents."""

  if name == 'ag_news':
    dset = tfds.load('ag_news_subset', 'train')
    dset = extract(dset, 'description')
    return dset.as_numpy_iterator()

  elif name == 'goemotions':
    dset = tfds.load('goemotions', 'train')
    dset = extract(dset, 'comment_text')
    return dset.as_numpy_iterator()

  elif name == 'imdb':
    dset = tfds.load('imdb_reviews', 'unsupervised')
    dset = extract(dset, 'text')
    return dset.as_numpy_iterator()

  elif name == 'snli':
    dset = tfds.load('snli', 'train')
    dset = extract(dset, 'premise')
    return dset.as_numpy_iterator()

  elif name == 'yelp':
    fname = os.path.join(data_dir, 'yelp/train.csv')
    parser = renn.data.data_utils.PARSERS[name]
    generator = renn.data.data_utils.readfile(fname, parser)
    return map(lambda d: d['text'], generator)

  elif name == 'dbpedia':
    fname = os.path.join(data_dir, 'dbpedia/train.csv')
    parser = renn.data.data_utils.PARSERS[name]
    generator = renn.data.data_utils.readfile(fname, parser)
    return map(lambda d: d['text'], generator)

  elif name == 'amazon':
    fname = os.path.join(data_dir, 'amazon/train.csv')
    parser = renn.data.data_utils.PARSERS[name]
    generator = renn.data.data_utils.readfile(fname, parser)
    return map(lambda d: d['text'], generator)

  else:
    raise ValueError(f'Invalid dataset {name}.')


def save_vocab(filename, vocab):
  with open(filename, 'w') as f:
    for v in vocab:
      f.write(v + '\n')


def extract(dset, key):
  return dset.map(lambda d: d[key].decode())


def main():
  args = parser.parse_args()

  # Build vocab.
  print(f'Building vocab for {args.dataset}')
  iterator = load_dataset(args.dataset, os.path.expanduser(args.data_dir))
  vocab = renn.data.build_vocab(tqdm.tqdm(iterator),
                                args.vocab_size,
                                split_fun=lambda d: d.lower().split())

  # Save to file.
  filename = os.path.join(args.save_dir, args.dataset + '.vocab')
  print(f'Saving vocab to: {filename}')
  save_vocab(filename, vocab)


if __name__ == '__main__':
  main()
