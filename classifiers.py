from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import re
import glob
import os
import numpy as np
import logging
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import operator

logger = logging.getLogger(__name__)

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key, NER_DIR): 
        self.key = key
        self.NER_DIR = NER_DIR

    def _load_data(self, file):
        with open(file, 'r') as f: 
            for line in f.readlines(): data = [re.findall('"(.*?)"', line)]
        return zip(*data)

    def fit(self, X, y):    
        files = glob.glob(os.path.join(self.NER_DIR, '*.con'))
        self.data = np.array([self._load_data(file) for file in files])
        return self

    def transform(self, X): # X is list of index
        features = np.recarray(shape=(len(X),),
                               dtype=[('keyword', object), ('tag', object)])
        features['keyword'], features['tag'] = zip(*[self.data[index] for index  in X])
        if self.key == 'words': return features['keyword']
        if self.key == 'tags': return features['tag']
        else: raise Exception('Incorrect key.')

class FastTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, label, FASTTEXT_DIR):
        self.label = label
        self.FASTTEXT_DIR = FASTTEXT_DIR 

    def fit(self, X, y):        
        with open(os.path.join(self.FASTTEXT_DIR, '{}_li.txt'.format(self.label)), 'r') as ftfile:  self.li = ftfile.readlines()
        return self

    def transform(self, X): # X is a list of index
        return  np.array([self.li[i].strip().split('\t') for i in X]).astype(float)

class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, TEXT_FILE):
        self.TEXT_FILE = TEXT_FILE

    def fit(self, X, y):
        with open(self.TEXT_FILE, 'r') as textf: self.text_li = [e.strip() for e in textf.readlines()]
        return self

    def transform(self, X):
        li = np.array([self.text_li[index] for index in X])
        return np.array([self.text_li[index] for index in X])

class ExternalListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, li, number, TEXT_FILE):
        self.list =  li
        self.TEXT_FILE = TEXT_FILE
        self.number = number

    def fit(self, X, y):
        li = set([e.strip().lower() for e in self.list])
        with open(self.TEXT_FILE, 'r') as textf: text_li = [e.strip().lower() for e in textf.readlines()]
        self.data = np.array([filter(lambda x:x != None, [text.split(' ')[text.split(' ').index(e)-self.number:text.split(' ').index(e)+self.number+1]  if e in text.split(' ') else None for e in li]) for index, text in enumerate(text_li)])
        self.flat_data = np.array([[e for items in sublist for e in items] for sublist in self.data])
        return self

    def transform(self, X):
        li = np.array([self.flat_data[index] for index in X])
        print(li[14])
        return np.array([self.flat_data[index] for index in X])
 
class TagKeywordTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, NER_DIR, tag = 'treatment'):
        self.tag = tag
        self.NER_DIR = NER_DIR

    def _load_data(self, file):
        data_dict = defaultdict(list)
        with open(file, 'r') as f: 
            for line in f.readlines(): 
                word, tag = re.findall('"(.*?)"', line)
                data_dict[tag].append(word)
        return data_dict

    def fit(self, X, y):    
        files = glob.glob(os.path.join(self.NER_DIR, '*.con'))
        self.data = np.array([self._load_data(file) for file in files]) # list of dict
        return self

    def transform(self, X):
        return np.array([self.data[index][self.tag] for index in X])

class Cr_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, TEXT_FILE, label='CREATININE'):
        self.label = label
        self.TEXT_FILE = TEXT_FILE
        self.weights = None
    
    def fit(self, X, y):
        return self

    def __canwork(self, f, *args, **kw):
        try:
            f(*args, **kw)
            return True
        except Exception:
            return False

    def transform(self, X):
        cr_list = ['creatinine ', 'Cr']
        with open(self.TEXT_FILE, 'r') as textf: 
            text_li_all = textf.readlines()
            text_li = [text_li_all[index].lower() for index in X]
        result = np.empty(shape=len(text_li), dtype=object)
        for i, text in enumerate(text_li):
            for e in cr_list:
                t_li = text.lower().split(' ')
                if e.lower() in t_li:
                    li = [t_li[j+1] for j in [j for j,v in enumerate(t_li) if v==e.lower()]]
                    # result[i] = True in [((0.6 > float(e)) or (float(e) > 1.5)) for e in li if self.__canwork(float, e)] 
                    result[i] = True in [(float(e) > 1.5) for e in li if self.__canwork(float, e)] 
                else: result[i] = False
        return ['M' if e == True else 'N' for e in result]

class HBA1C_Transfomer(BaseEstimator, TransformerMixin):
    def __init__(self, TEXT_FILE, label='HBA1C'):
        self.label = label
        self.TEXT_FILE = TEXT_FILE
        self.weights = None
    
    def fit(self, X, y):
        return self

    def __canwork(self, f, *args, **kw):
        try:
            f(*args, **kw)
            return True
        except Exception:
            return False

    def transform(self, X):
        cr_list = ['A1c', 'HbA1c', 'Hb1c', 'HgA1C', 'A1C']
        with open(self.TEXT_FILE, 'r') as textf: 
            text_li_all = textf.readlines()
            text_li = [text_li_all[index].lower() for index in X]
        result = np.empty(shape=len(text_li), dtype=object)
        for i, text in enumerate(text_li):
            li = []
            for e in cr_list:
                t_li = text.lower().split(' ')
                if e.lower() in t_li: li.extend([[t_li[j+1].strip('Hh'), t_li[j+3].strip('Hh')] for j in [j for j,v in enumerate(t_li) if v==e.lower()]])
            result[i] = True in [(operator.le(6.5, float(e)) and operator.le(float(e), 9.5)) for sub_li in li for e in sub_li if self.__canwork(float, e)] 
        return ['M' if e == True else 'N' for e in result]

class AB_Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, TEXT_FILE, label='ABDOMINAL'):
        self.label = label
        self.TEXT_FILE = TEXT_FILE
        self.weights = None
    
    def fit(self, X, y):
        return self

    def predict(self, X):
        ab_list = ['bowel surgery', 'resection', 'bowel obstruction', 'intestine resection', 'abdominal surgery', 'appendectomy', 'caesarean section', 'c section', 'c-section'
        'inguinal hernia surgery', 'laparotomy','Laparoscopy', 'gallbladder removal']
        result = []
        with open(self.TEXT_FILE, 'r') as textf: 
            text_li_all = textf.readlines()
            text_li = [text_li_all[index].lower() for index in X]
        return np.array(['M' if r == True else 'N' for r in [True in [e in text for e in ab_list] for text in text_li]])

    def predict_proba(self, X):
        result = self.predict(X)
        return np.array([[1,0] if e == 'M' else [0,1] for e in result])

class printX(BaseEstimator, TransformerMixin):
    def fit(self, X, y): 
        print X.shape
        return self
    def transform(self, X): return X

class BaseClassifer(Pipeline):
    def __init__(self, label, NER_DIR, FASTTEXT_DIR, CAD_FILE, SUPP_FILE, AB_FILE, DIA_FILE, TEXT_FILE, ner_tag='treatment', n_estimators = 200, weights = [1, 2, 0.5, 0.2, 0, 0.1, 0, 0, 0, 0, 0, 0.5]):
        self.label = label
        self.NER_DIR = NER_DIR
        self.FASTTEXT_DIR = FASTTEXT_DIR
        self.CAD_FILE = CAD_FILE
        self.SUPP_FILE = SUPP_FILE
        self.AB_FILE = AB_FILE
        self.DIA_FILE = DIA_FILE
        self.TEXT_FILE = TEXT_FILE
        self.ner_tag = ner_tag
        self.weights = weights
        self.gbclf_n_estimators = n_estimators
        self.pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                ('text', Pipeline([
                    ('textTransformer', TextTransformer(self.TEXT_FILE)),
                    ('tfidf', TfidfVectorizer(min_df=3, analyzer=lambda x:x)),
                    ])),
                ('keywords', Pipeline([
                        ('selector', ItemSelector(key='words', NER_DIR = self.NER_DIR)),
                        ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])), # keyswords, from cliner
                ('tags', Pipeline([
                        ('selector', ItemSelector(key='tags', NER_DIR = self.NER_DIR)),
                        ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])), # tags, such as 'treatment'
                ('fasttext', FastTextTransformer(self.label, self.FASTTEXT_DIR)),
                ('cad', Pipeline([
                    ('CADTransformer', ExternalListTransformer(open(self.CAD_FILE).readlines(), 0, self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('supp', Pipeline([
                    ('suppTransformer', ExternalListTransformer(open(self.SUPP_FILE).readlines(), 0, self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('ab', Pipeline([
                    ('ABTransformer', ExternalListTransformer(open(self.AB_FILE).readlines(), 0, self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('asa', Pipeline([
                    ('ASATransformer', ExternalListTransformer(['ASA', 'acetylsalicylic', 'Aspirin', 'myocardial infarction', 'MI'], 0, self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('cr', Pipeline([
                    ('CRTransformer', Cr_Transformer(self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('HBA1C', Pipeline([
                    ('HBA1CTransformer', HBA1C_Transfomer(self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('dia', Pipeline([
                    ('DiaTransformer', ExternalListTransformer(open(self.DIA_FILE).readlines(), 0, self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('MI', Pipeline([
                    ('MITransformer', ExternalListTransformer(['NSTEMI', 'myocardial infarction', ' STEMI ', ' MI ', 'heart attack'], 2, self.TEXT_FILE)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ('tag_keywords', Pipeline([
                    ('tag_keywords', TagKeywordTransformer(tag = self.ner_tag, NER_DIR = self.NER_DIR)),
                    ('CountVectorizer', CountVectorizer(min_df=1, analyzer=lambda x:x)),
                    ])),
                ],
                # weight components in FeatureUnion
                transformer_weights=dict(zip(['text', 'keywords', 'tags', 'fasttext', 'cad' ,'supp', 'ab', 'asa', 'cr', 'HBA1Cr', 'dia', 'MI', 'tag_keywords'], self.weights))
            )),
            # Use a SVC classifier on the combined features
            ('pr', printX()),
            ('vote', VotingClassifier(estimators=[
                ('lr,', LogisticRegression(random_state=1, class_weight=None)),
                ('svc', SVC(kernel='linear', C = 10000000, gamma = 0.000001, probability=True, class_weight=None)),
                ('rfc', GradientBoostingClassifier(n_estimators=self.gbclf_n_estimators, min_samples_leaf=2, random_state=0)),
                ], 
                voting='soft', weights=[1.5,2,2]))
        ])

    def fit(self, X_train, y_train):
        logging.info('start to fit, the number of X_train is {}...'.format(len(X_train)))
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        logging.info('start to predict, the number of X_test is {}...'.format(len(X_test)))
        return self.pipeline.predict(X_test)

class KETO_DietY1Classifier():
    def __init__(self, label, TEXT_FILE):
        self.label = label
        self.TEXT_FILE = TEXT_FILE
        self.weights = None
    
    def fit(self, X, y):
        return self

    def predict(self, X):
        keto_diety1_lit = ['ketogenic diet','ketogenic', 'Keto Diet', 'Keto']
        with open(self.TEXT_FILE, 'r') as textf: text_li = [e.strip().lower() for e in textf.readlines()]
        self.data = np.array([[1  if e in text else None for e in keto_diety1_lit] for index, text in enumerate(text_li)])
        print(['N' if 1 not in self.data[index] else 'M' for index in X])
        return ['N' if 1 not in self.data[index] else 'M' for index in X]

class CliClassifier(BaseClassifer):
    def __init__(self, NER_DIR, FASTTEXT_DIR, CAD_FILE, SUPP_FILE, AB_FILE, DIA_FILE, TEXT_FILE):
        self.NER_DIR = NER_DIR
        self.FASTTEXT_DIR = FASTTEXT_DIR
        self.SUPP_FILE = SUPP_FILE
        self.CAD_FILE = CAD_FILE
        self.AB_FILE = AB_FILE
        self.DIA_FILE = DIA_FILE
        self.TEXT_FILE = TEXT_FILE
        self.ABDOMINLALclf = AB_Classifier(self.TEXT_FILE)
        self.ADVANCEDclf = BaseClassifer("ADVANCED-CAD", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, ner_tag='treatment', weights = [1, 2.5, 0.5, 0.5, 3, 0, 0, 0, 0, 0, 0, 0.5, 1])
        self.ALCOHOL_ABUSEclf = BaseClassifer("ALCOHOL-ABUSE", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1, 2, 0.5, 0.2, 0, 0.5, 0, 0, 0, 0, 0, 0, 5])
        self.ASP_FOR_MIclf = BaseClassifer("ASP-FOR-MI", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1, 2, 0.5, 0.2, 0, 0.1, 0, 3.5, 0.5, 0, 0, 1.5, 3])
        self.CREATININEclf = BaseClassifer("CREATININE", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1, 2, 0.5, 0.2, 0, 0.1, 0, 0, 4, 0, 0, 0, 5])
        self.DRUG_ABUSEclf = BaseClassifer("DRUG-ABUSE", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1.5, 3, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.DIETSUPP_2MOSclf = BaseClassifer("DIETSUPP-2MOS", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1, 2, 0.5, 0.2, 0, 5, 0, 0, 0, 0, 0, 0, 2])
        self.ENGLISHclf = BaseClassifer("ENGLISH", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [3, 1.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2])
        self.HBA1Cclf = BaseClassifer("HBA1C", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1.5, 2, 0.5, 0.2, 0, 0.1, 0, 0, 0, 4, 0, 0, 1])
        self.KETO_1YRclf = KETO_DietY1Classifier("KETO-1YR", self.TEXT_FILE)
        self.MAJOR_DIABETESclf = BaseClassifer("MAJOR-DIABETES", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1, 2, 0.5, 0.2, 0, 0.1, 0.5, 0, 0, 0, 4, 0.0, 5])
        self.MAKES_DECISIONSclf = BaseClassifer("MAKES-DECISIONS", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1, 2, 0.5, 0.2, 0, 0.1, 0, 0, 0, 0, 0, 0, 0])
        self.MI_6MOSclf = BaseClassifer("MI-6MOS", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, weights = [1.5, 2.5, 0.5, 0.2, 0, 0.1, 0, 0, 0, 0, 0, 3, 0.5])


        # self.ABDOMINLALclf = BaseClassifer("ABDOMINAL", self.NER_DIR, self.FASTTEXT_DIR, self.CAD_FILE, self.SUPP_FILE, self.AB_FILE, self.DIA_FILE, self.TEXT_FILE, ner_tag='problem', n_estimators = 100, weights = [1, 2, 0.5, 0.2, 0 ,0.1, 4, 0, 0, 0, 0, 0, 1])

    def fit(self, X, y):
        self.ABDOMINLALclf.fit(X, y[0])
        self.ADVANCEDclf.fit(X, y[1])
        self.ALCOHOL_ABUSEclf.fit(X, y[2])
        self.ASP_FOR_MIclf.fit(X, y[3])
        self.CREATININEclf.fit(X, y[4])
        self.DIETSUPP_2MOSclf.fit(X, y[5])
        self.DRUG_ABUSEclf.fit(X, y[6])
        self.ENGLISHclf.fit(X, y[7])
        self.HBA1Cclf.fit(X, y[8])
        self.KETO_1YRclf.fit(X, y[9])
        self.MAJOR_DIABETESclf.fit(X, y[10])
        self.MAKES_DECISIONSclf.fit(X, y[11])
        self.MI_6MOSclf.fit(X, y[12])

        # self.ABDOMINLALclf.fit(X, y)

        # logging.info("classifier is <<{}>>".format(self.clf.label))
        # logging.info('transformer weights are {}'.format(self.clf.weights))
        # logging.info('fit end...')

        return self

    def __writeY(self, y, clf):
        label = clf.label
        with open(os.path.join('/Users/liuman/Documents/n2c2/data/pred', label+'.pred.txt'), 'w') as outputf:
            for e in y: outputf.write(e+'\n')
        return 0


    def predict(self, X):
        ab_pred = self.ABDOMINLALclf.predict(X)
        ad_pred = self.ADVANCEDclf.predict(X)
        al_pred = self.ALCOHOL_ABUSEclf.predict(X)
        asp_pred = self.ASP_FOR_MIclf.predict(X)
        cre_pred = self.CREATININEclf.predict(X)
        diet_pred = self.DIETSUPP_2MOSclf.predict(X)
        drug_pred = self.DRUG_ABUSEclf.predict(X)
        eng_pred = self.ENGLISHclf.predict(X)
        hba_pred = self.HBA1Cclf.predict(X)
        keto_pred = self.KETO_1YRclf.predict(X)
        maj_pred = self.MAJOR_DIABETESclf.predict(X)
        make_pred = self.MAKES_DECISIONSclf.predict(X)
        mi_pred = self.MI_6MOSclf.predict(X)

        self.__writeY(ab_pred, self.ABDOMINLALclf)
        self.__writeY(ad_pred, self.ADVANCEDclf)
        self.__writeY(al_pred, self.ALCOHOL_ABUSEclf)
        self.__writeY(asp_pred, self.ASP_FOR_MIclf)
        self.__writeY(cre_pred, self.CREATININEclf)
        self.__writeY(diet_pred, self.DIETSUPP_2MOSclf)
        self.__writeY(drug_pred, self.DRUG_ABUSEclf)
        self.__writeY(eng_pred, self.ENGLISHclf)
        self.__writeY(hba_pred, self.HBA1Cclf)
        self.__writeY(keto_pred, self.KETO_1YRclf)
        self.__writeY(maj_pred, self.MAJOR_DIABETESclf)
        self.__writeY(make_pred, self.MAKES_DECISIONSclf)
        self.__writeY(mi_pred, self.MI_6MOSclf)

        return np.stack((ab_pred, ad_pred, al_pred, asp_pred, cre_pred, diet_pred, drug_pred, eng_pred, hba_pred, keto_pred, maj_pred, make_pred, mi_pred))
        
        # pred = self.ABDOMINLALclf.predict(X)
        # return pred


