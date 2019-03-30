import os
import glob
from argparse import ArgumentParser
from tabulate import tabulate

parser = ArgumentParser(description='test')
parser.add_argument('--y', type=lambda x: glob.glob(os.path.join(x, '*.txt')), metavar='files', default=None, help='List of files to use as label data. ')
args = parser.parse_args()

def numberOfLabels(file):
    filename = os.path.basename(file)
    with open(file, 'r') as inputf:
        li = [e.strip() for e in inputf.readlines()]
        count = {filename: {'M':float(li.count('M'))/len(li), 'N':float(li.count('N'))/len(li)}}
    return count

count_all = {}
for file in args.y: count_all.update(numberOfLabels(file))
print('Statistics of labels #:\n' + tabulate([[k, v.values()[0], v.values()[1]] for k, v in count_all.items()], headers=['Label', 'met', 'not met'], tablefmt='psql'))

'''
Statistics of labels #:
+------------------------+-------+-----------+
| Label                  |   met |   not met |
|------------------------+-------+-----------|
| MI-6MOS_li.txt         |    18 |       184 |
| KETO-1YR_li.txt        |     1 |       201 |
| DRUG-ABUSE_li.txt      |    12 |       190 |
| ENGLISH_li.txt         |   192 |        10 |
| ADVANCED-CAD_li.txt    |   125 |        77 |
| ALCOHOL-ABUSE_li.txt   |     7 |       195 |
| DIETSUPP-2MOS_li.txt   |   105 |        97 |
| MAKES-DECISIONS_li.txt |   194 |         8 |
| ASP-FOR-MI_li.txt      |   162 |        40 |
| ABDOMINAL_li.txt       |    77 |       125 |
| MAJOR-DIABETES_li.txt  |   113 |        89 |
| HBA1C_li.txt           |    67 |       135 |
| CREATININE_li.txt      |    82 |       120 |
+------------------------+-------+-----------+

Statistics of labels #:
+------------------------+-----------+-----------+
| Label                  |       met |   not met |
|------------------------+-----------+-----------|
| MI-6MOS_li.txt         | 0.0891089 |  0.910891 |
| KETO-1YR_li.txt        | 0.0049505 |  0.99505  |
| DRUG-ABUSE_li.txt      | 0.0594059 |  0.940594 |
| ENGLISH_li.txt         | 0.950495  |  0.049505 |
| ADVANCED-CAD_li.txt    | 0.618812  |  0.381188 |
| ALCOHOL-ABUSE_li.txt   | 0.0346535 |  0.965347 |
| DIETSUPP-2MOS_li.txt   | 0.519802  |  0.480198 |
| MAKES-DECISIONS_li.txt | 0.960396  |  0.039604 |
| ASP-FOR-MI_li.txt      | 0.80198   |  0.19802  |
| ABDOMINAL_li.txt       | 0.381188  |  0.618812 |
| MAJOR-DIABETES_li.txt  | 0.559406  |  0.440594 |
| HBA1C_li.txt           | 0.331683  |  0.668317 |
| CREATININE_li.txt      | 0.405941  |  0.594059 |
+------------------------+-----------+-----------+
'''
"""
test:
Statistics of labels #:
+--------------------------+----------+-----------+
| Label                    |      met |   not met |
|--------------------------+----------+-----------|
| ENGLISH.pred.txt         | 1        |  0        |
| CREATININE.pred.txt      | 0.162791 |  0.837209 |
| MAJOR-DIABETES.pred.txt  | 0.593023 |  0.406977 |
| MI-6MOS.pred.txt         | 0        |  1        |
| KETO-1YR.pred.txt        | 0        |  1        |
| HBA1C.pred.txt           | 0.209302 |  0.790698 |
| DIETSUPP-2MOS.pred.txt   | 0.465116 |  0.534884 |
| ABDOMINAL.pred.txt       | 0.151163 |  0.848837 |
| MAKES-DECISIONS.pred.txt | 1        |  0        |
| DRUG-ABUSE.pred.txt      | 0        |  1        |
| ASP-FOR-MI.pred.txt      | 1        |  0        |
| ADVANCED-CAD.pred.txt    | 0.662791 |  0.337209 |
| ALCOHOL-ABUSE.pred.txt   | 0        |  1        |
+--------------------------+----------+-----------+

###