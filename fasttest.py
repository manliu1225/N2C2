from .models.fasttextclf import FasttextClassifier
import numpy as np
from argparse import ArgumentParser
import codecs
import os

parser = ArgumentParser(description='fasttext')
parser.add_argument('--X_train', type=open,  metavar='file', default=None, help='List of files to use as training data.')
parser.add_argument('--y_train', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--X_test', type=open,  metavar='file', default=None,  help='List of files to use as training data.')
parser.add_argument('--y_test', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--model', help='load model.')
args = parser.parse_args()


def load_instance(f):
    return [e.strip() for e in f.readlines()]

# X_train = load_instance(args.X_train)
X_test = load_instance(args.X_test)
# y_train = load_instance(args.y_train)
# y_test = load_instance(args.y_test)

ftclf = FasttextClassifier(lr=10, epoch=20, dim=300, ws=50)
if args.model: 
    ftclf.load_model(args.model)
    outputf = os.path.join('/Users/liuman/Documents/n2c2/features/test_features/fasttext' , os.path.basename(args.model).split('.')[0] + 'txt')
    with open(outputf, 'w') as f:
        for n, m in ftclf.predict_prob(X_test): 
            f.write('{}\t{}\n'.format(n[1], m[1]))

else:
    ftclf.fit(X_train, y_train, '/Users/liuman/Documents/n2c2/models/saved_models/{}_fasttext.model'.format(os.path.basename(args.y_train.name).split('.')[0]))
# ftclf.test(X_test, y_test)
# print(ftclf.predict(X_test, y_test))
# print(ftclf.predict_prob(X_test, y_test))
# outputf = os.path.join('/Users/liuman/Documents/n2c2/features/fasttext' , os.path.basename(str(args.y_test.name)))
# with open(outputf, 'w') as f:
#     for n, m in ftclf.predict_prob(X_test, y_test): 
#         f.write('{}\t{}\n'.format(n[1], m[1]))

'''
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/ABDOMINAL_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/ADVANCED-CAD_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/ALCOHOL-ABUSE_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/ASP-FOR-MI_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/CREATININE_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/DIETSUPP-2MOS_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/DRUG-ABUSE_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/ENGLISH_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/HBA1C_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/KETO-1YR_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/MAJOR-DIABETES_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/MAKES-DECISIONS_li.txt
python -m n2c2.fasttest --X_train n2c2/data/TEXT_li.txt  --y_train n2c2/data/clean_data/MI-6MOS_li.txt

python -m n2c2.fasttest  --model n2c2/models/saved_models/ABDOMINAL_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/ADVANCED-CAD_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/ALCOHOL-ABUSE_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/ASP-FOR-MI_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/CREATININE_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/DIETSUPP-2MOS_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/DRUG-ABUSE_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/ENGLISH_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/HBA1C_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/KETO-1YR_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/MAJOR-DIABETES_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/MAKES-DECISIONS_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt
python -m n2c2.fasttest  --model n2c2/models/saved_models/MI-6MOS_li_fasttext.model.bin  --X_test /Users/liuman/Documents/n2c2/data/test_preprocessed/TEXT_li.txt


'''