""" Data utils """
import csv


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


PARSERS = {'yelp': column_parser(1), 'dbpedia': column_parser(2)}
