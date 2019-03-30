from .models.classifiers import BaseClassifer, CliClassifier
from . import load_instance
from argparse import ArgumentParser
import logging
import glob
import numpy as np
from  sklearn.metrics import f1_score, make_scorer
from tabulate import tabulate
import os
import sys
import random
from sklearn.model_selection import KFold, cross_validate
from .track1_eval import f1_score_measure

logging.basicConfig(format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = ArgumentParser(description='test')
parser.add_argument('--y', type=lambda x: glob.glob(os.path.join(x, '*.txt')), metavar='files', default=None, help='List of files to use as label data. ')
args = parser.parse_args()

SUPP_FILE = '/Users/liuman/Documents/n2c2/features/dietsupp.txt'
CAD_FILE = '/Users/liuman/Documents/n2c2/features/CAD_list.txt'
AB_FILE = '/Users/liuman/Documents/n2c2/features/Abdomen_list.txt'
DIA_FILE = '/Users/liuman/Documents/n2c2/features/diabete_complications.txt'
TEXT_FILE = '/Users/liuman/Documents/n2c2/data/total_TEXT_li.txt'
FASTTEXT_DIR='/Users/liuman/Documents/n2c2/features/total_fasttext'
NER_DIR = '/Users/liuman/Documents/n2c2/features/total_ner_features'
# ORIGINAL_DIR = '/Users/liuman/Documents/n2c2/data/original'
tag_list = ['ABDOMINAL',
 'ADVANCED-CAD',
 'ALCOHOL-ABUSE',
 'ASP-FOR-MI',
 'CREATININE',
 'DIETSUPP-2MOS',
 'DRUG-ABUSE',
 'ENGLISH',
 'HBA1C',
 'KETO-1YR',
 'MAJOR-DIABETES',
 'MAKES-DECI',
 'MI-6MOS',
 ]

###offcial result###
random.seed(5)
# train_labels = [os.path.basename(f) for f in glob.glob(TRAIN_DIR)]
# test_labels = [os.path.basename(f) for f in glob.glob(TEST_DIR)]
# X_train = range(len(train_labels))
# X_test = range(len(train_labels), len(train_labels)+len(test_labels)-1)
# y_all_train = [load_instance(x) for x in args.y_train] # list of list
# y_all_test = [load_instance(x) for x in args.y_test] # list of list

X_train = random.sample(range(202),202)
X_test = range(202,288)
# labels = glob.glob(os.path.join(ORIGINAL_DIR, '*.xml'))
# print([os.path.basename(labels[i]) for i in X_test])
logger.info('load instance successfully! number of training data is {}'.format(len(X_train)))
logger.info('load instance successfully! number of test data is {}'.format(len(X_test)))
y_all = [load_instance(x) for x in args.y]
y_train_all = [[y[index] for index in X_train] for y in y_all]


cliclf = CliClassifier(NER_DIR, FASTTEXT_DIR, CAD_FILE, SUPP_FILE, AB_FILE, DIA_FILE, TEXT_FILE)
cliclf.fit(X_train, y_train_all)
y_pred = cliclf.predict(X_test)
logger.info('Results of fitting model #:\n' + tabulate([[tag, f1_score(y_train_all[i], y_train_all[i], average='micro')] for i, tag in enumerate(tag_list)], headers=['Label', 'Score'], tablefmt='psql'))

# [[(tag, value), (tag, value)...],[...], ...]
y_test_transformer = [zip(tag_list, e) for e in zip(*y_train_all)]
y_pred_transformer = [zip(tag_list, e) for e in zip(*y_train_all)]
f1_score_measure(y_test_transformer,y_pred_transformer ,1)

print("#"*30)

'''
cat fasttext/ABDOMINAL_li.txt test_features/fasttext/ABDOMINAL_li_fasttexttxt > total_fasttext/ABDOMINAL_li.txt
cat fasttext/ADVANCED-CAD_li.txt test_features/fasttext/ADVANCED-CAD_li_fasttexttxt > total_fasttext/ADVANCED-CAD_li.txt
cat fasttext/ALCOHOL-ABUSE_li.txt test_features/fasttext/ALCOHOL-ABUSE_li_fasttexttxt > total_fasttext/ALCOHOL-ABUSE_li.txt
cat fasttext/ASP-FOR-MI_li.txt test_features/fasttext/ASP-FOR-MI_li_fasttexttxt > total_fasttext/ASP-FOR-MI_li.txt
cat fasttext/CREATININE_li.txt test_features/fasttext/CREATININE_li_fasttexttxt > total_fasttext/CREATININE_li.txt
cat fasttext/DIETSUPP-2MOS_li.txt test_features/fasttext/DIETSUPP-2MOS_li_fasttexttxt > total_fasttext/DIETSUPP-2MOS_li.txt
cat fasttext/DRUG-ABUSE_li.txt test_features/fasttext/DRUG-ABUSE_li_fasttexttxt > total_fasttext/DRUG-ABUSE_li.txt
cat fasttext/ENGLISH_li.txt test_features/fasttext/ENGLISH_li_fasttexttxt > total_fasttext/ENGLISH_li.txt
cat fasttext/HBA1C_li.txt test_features/fasttext/HBA1C_li_fasttexttxt > total_fasttext/HBA1C_li.txt
cat fasttext/KETO-1YR_li.txt test_features/fasttext/KETO-1YR_li_fasttexttxt > total_fasttext/KETO-1YR_li.txt
cat fasttext/MAJOR-DIABETES_li.txt test_features/fasttext/MAJOR-DIABETES_li_fasttexttxt > total_fasttext/MAJOR-DIABETES_li.txt
cat fasttext/MAKES-DECISIONS_li.txt test_features/fasttext/MAKES-DECISIONS_li_fasttexttxt > total_fasttext/MAKES-DECISIONS_li.txt
cat fasttext/MI-6MOS_li.txt test_features/fasttext/MI-6MOS_li_fasttexttxt > total_fasttext/MI-6MOS_li.txt

'''



