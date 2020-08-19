"""Scripts to generate vocabulary files for different datasets."""

import argparse
import os

import tensorflow_datasets as tfds
import renn

parser = argparse.ArgumentParser(description='Generate vocab files.')
parser.add_argument('dataset', type=str)
parser.add_argument('--savedir', type=str, default='/tmp')
parser.add_argument('--vocab_size', type=int, default=2**15)

DATASETS = {
    # dataset name, split, and dictionary key that contains text
    'ag_news': ('ag_news_subset', 'train', 'description'),
    'goemotions': ('goemotions', 'train', 'comment_text'),
    'imdb': ('imdb_reviews', 'unsupervised', 'text'),
    'snli': ('snli', 'train', 'premise'),
}


def save_vocab(filename, vocab):
  with open(filename, 'w') as f:
    for v in vocab:
      f.write(v + '\n')


def build_vocab(name, split, key, vocab_size):
  dset = tfds.load(name, split=split)

  # Extract text.
  dset = tfds.as_numpy(dset.map(lambda d: d[key]))

  # Build vocabulary.
  return renn.data.build_vocab(iter(dset),
                               vocab_size,
                               split_fun=lambda d: d.decode().lower().split())


def main():
  args = parser.parse_args()

  if args.dataset in DATASETS:
    dset, split, key = DATASETS[args.dataset]

  else:
    raise ValueError(f'Invalid dataset {args.dataset}')

  # Build vocab.
  print(f'Building vocab for {args.dataset}')
  vocab = build_vocab(dset, split, key, args.vocab_size)

  # Save to file.
  filename = os.path.join(args.savedir, args.dataset + '.vocab')
  print(f'Saving vocab to: {filename}')
  save_vocab(filename, vocab)


if __name__ == '__main__':
  main()
