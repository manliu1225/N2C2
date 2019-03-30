import codecs
import json
import logging
import os
import random
import re
from warnings import warn

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from stemming.porter2 import stem as porter_stem

from tabulate import tabulate
import xml.etree.cElementTree as ET
import glob
from collections import defaultdict
import nltk


def get_data(dir):
  filenames = glob.glob(os.path.join(dir, '*.xml'))
  print(filenames)

  data = defaultdict(list)
  data["TEXT_li"] = [ET.ElementTree(file=file).find("TEXT").text for file in filenames]
  # for e in ET.ElementTree(file=file).findall(r'TAGS/*'): data['{}_li'.format(e.tag)] = [(os.path.basename(file), ET.ElementTree(file=file).find('TAGS/{}'.format(e.tag)).attrib.values()[0]) for file in filenames]
  return data

def clean_data(e):
  return re.sub(r'\n+|\t+|\*+|_+|\s+|-+', r' ', e)

def tokenize_data(e):
  return ' '.join(nltk.word_tokenize(e))
   
