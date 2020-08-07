"""Scripts to generate vocabulary files for different datasets."""

import argparse
import os

import tensorflow_datasets as tfds
import renn


parser = argparse.ArgumentParser(description='Generate vocab files.')
parser.add_argument('dataset', type=str)
parser.add_argument('--savedir', type=str, default='/tmp')
parser.add_argument('--vocab_size', type=int, default=2 ** 15)


def imdb_vocab(filename, vocab_size):
  dset = tfds.load('imdb_reviews', split='unsupervised')

  # Build generator that iterates over the corpus.
  corpus_generator = tfds.as_numpy(dset.map(lambda d: d['text']))

  # Build vocabulary.
  vocab = renn.data.build_vocab(corpus_generator,
                                vocab_size,
                                split_fun=lambda d: d.decode().lower().split())

  # Save to file.
  with open(filename, 'w') as f:
    for v in vocab:
      f.write(v + '\n')


def main():
  args = parser.parse_args()

  # Where to save.
  filename = os.path.join(args.savedir, args.dataset + '.vocab')
  print(f'Saving vocab to: {filename}')

  if args.dataset == 'imdb':
    imdb_vocab(filename, args.vocab_size)

  else:
    raise ValueError(f'Invalid dataset {args.dataset}')


if __name__ == '__main__':
  main()
