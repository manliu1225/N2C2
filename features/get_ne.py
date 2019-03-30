import argparse
import os
import subprocess
import glob

parser = argparse.ArgumentParser()
parser.add_argument("inputf", type=open, help="clean data after preprocessing")
# parser.add_argument("outputf", type=argparse.FileType('w'), help="output file containing list of list")
args = parser.parse_args()



for i, line in enumerate(args.inputf.readlines(),start=202):
    with open('test_tm/{}.txt'.format(i), 'w') as f: f.write(line)

# for file in glob.glob(os.path.join('tmp', '*.txt')):
#     print file
#     bashCommand = './features/NER/CliNER_master/cliner predict --txt {}  --out tm --model ./features/NER/CliNER_master/models/silver.crf --format i2b2'.format(file)
#     process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE)
#     output, error = process.communicate()
#     print output
#     print error
#     break