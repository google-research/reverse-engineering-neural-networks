""" Data utils """
import csv
import tensorflow as tf


def readfile(filename, parse_row):
  """
  Reads a csv file containing labeled data, where
  the function parse_row() extracts a score and
  text from the labeled data
  """

  with open(filename, 'r') as f:
    for row in csv.reader(f):

      score, text = parse_row(row)

      yield {'text': text, 'label': score}


def column_parser(text_column):
  """
  Returns a parser which parses a row of a csv file
  containing labeled data, extracting the label
  and the text

  This parser assumes the label is the zeroth element
  of the row, and the text is the 'text_column' element
  """

  def f(row):
    return int(row[0]), row[text_column]

  return f


def sentiment_relabel(num_classes):
  """
  Returns a function which relabels (initially five-class)
  sentiment labels for subclassing the Yelp and Amazon
  datasets.
  """

  LABEL_TYPE = tf.int64

  TENSORS = {
      -1: tf.constant(-1, dtype=LABEL_TYPE),
      0: tf.constant(0, dtype=LABEL_TYPE),
      1: tf.constant(1, dtype=LABEL_TYPE),
      2: tf.constant(2, dtype=LABEL_TYPE),
      3: tf.constant(3, dtype=LABEL_TYPE),
      4: tf.constant(4, dtype=LABEL_TYPE),
      5: tf.constant(5, dtype=LABEL_TYPE)
  }

  equal = lambda x, y: tf.cast(tf.equal(x, y), LABEL_TYPE)
  greater = lambda x, y: tf.cast(tf.greater(x, y), LABEL_TYPE)
  less = lambda x, y: tf.cast(tf.less(x, y), LABEL_TYPE)

  if num_classes in [1, 2]:
    return lambda x: greater(x, 3) + equal(x, 3) * TENSORS[-1]
  elif num_classes == 3:
    return lambda x: equal(x, 1) * TENSORS[0] + equal(x, 2) * TENSORS[
        -1] + equal(x, 3) * TENSORS[1] + equal(x, 4) * TENSORS[-1] + equal(
            x, 5) * TENSORS[2]
  elif num_classes == 5:
    return lambda x: tf.subtract(x, 1)
  else:
    raise ValueError('Sentiment subclasses must be 1,2,3,5')


PARSERS = {
    'yelp': column_parser(1),
    'dbpedia': column_parser(2),
    'amazon': column_parser(2)
}
