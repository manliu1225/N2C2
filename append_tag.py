import xml.etree.cElementTree as ET
from argparse import ArgumentParser
import logging
from collections import defaultdict
import sys
import glob
import os

# tree = ET.parse('/Users/liuman/Documents/n2c2/108.xml')
# tags= tree.find('TAGS')

# b = ET.SubElement(tags, 'AB')
# c = ET.SubElement(tags, 'AC')
# b.set('met', 'not met')
# c.set('met', 'met')

# tree.write('output.xml')

# sys.exit()
LABEL = {'N':'not met', 'M':'met'}

parser = ArgumentParser(description='test')
parser.add_argument('--pred', type=lambda x: glob.glob(os.path.join(x, '*.txt')), metavar='files', default=None, help='List of files to use as label data. ')
parser.add_argument('--original_files', type=lambda x: glob.glob(os.path.join(x, '*.xml')), metavar='files', default=None, help='List of files to use as label data. ')
parser.add_argument('--output_dir', help='output dir.')
args = parser.parse_args()

logging.basicConfig(format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

data = defaultdict(list)
for file in args.pred:
    with open(file, 'r') as inputf: data[os.path.basename(file).split('.')[0]] = [LABEL[e.strip()] for e in inputf.readlines()]

for i, file in enumerate(args.original_files):
    print(file)
    logger.info('process file {}...'.format(os.path.basename(file)))
    with open(file, 'r') as inputf:
        li = [e.strip() for e in inputf.readlines()]
        assert li[-3] == '<TAGS>'
        assert li[-2] == '</TAGS>'
        for k in sorted(data.keys()):
            v = data[k]
            print(len(v))
            li.insert(-2, '<{} met="{}" />'.format(k, v[i]))
    with open(os.path.join(args.output_dir, os.path.basename(file)), 'w') as outputf:
        for i in xrange(len(li)): 
            if i != len(li)-1: outputf.write(li[i] + '\n')
            else: outputf.write(li[i])








