#!/usr/local/bin/python

"""N2C2 Track 1 evaluation script."""

import argparse
import glob
import os
from collections import defaultdict
from xml.etree import cElementTree


class ClinicalCriteria(object):
    """Criteria in the Track 1 documents."""

    def __init__(self, tid, value):
        """Init."""
        self.tid = tid.strip().upper()
        self.ttype = self.tid
        self.value = value.upper().strip()

    def equals(self, other, mode='strict'):
        """Return whether the current criteria is equal to the one provided."""
        if other.tid == self.tid and other.value == self.value:
            return True
        return False


class RecordTrack1(object):
    """Record for Track 2 class."""

    def __init__(self, y_single):
        self.y_single = y_single
        self.annotations = self._get_annotations()
        self.text = None

    @property
    def tags(self):
        return self.annotations['tags']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        for (tag, value) in self.y_single:
            criterion = ClinicalCriteria(tag.upper(), value)
            annotations['tags'][tag.upper()] = criterion
            if value not in ('M', 'N'):
                assert '{}: Unexpected value ("{}") for the {} tag!'.format(
                    self.path, criterion.value, criterion.ttype)
        return annotations



class Measures(object):
    """Abstract Mhods and var to evaluate."""

    def __init__(self, tp=0.0, tn=0.0, fp=0.0, fn=0.0):
        """Initizialize."""
        assert type(tp) == int
        assert type(tn) == int
        assert type(fp) == int
        assert type(fn) == int
        self.tp = float(tp)
        self.tn = float(tn)
        self.fp = float(fp)
        self.fn = float(fn)

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision() * self.recall())
            den = beta**2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2


class SingleEvaluator(object):
    """Evaluate two single files."""

    def __init__(self, doc1, doc2, track, mode='strict', key=None, verbose=False):
        """Initialize."""
        assert isinstance(doc1, RecordTrack1)
        assert isinstance(doc2, RecordTrack1)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}}
        self.doc1 = doc1
        self.doc2 = doc2
        if key:
            gol = [t for t in doc1.tags.values() if t.ttype == key]
            sys = [t for t in doc2.tags.values() if t.ttype == key]
        else:
            gol = [t for t in doc1.tags.values()]
            sys = [t for t in doc2.tags.values()]
        self.scores['tags']['tp'] = len({s.tid for s in sys for g in gol if g.equals(s, mode)})
        self.scores['tags']['fp'] = len({s.tid for s in sys}) - self.scores['tags']['tp']
        self.scores['tags']['fn'] = len({g.tid for g in gol}) - self.scores['tags']['tp']
        self.scores['tags']['tn'] = 0
        if verbose and track == 2:
            tps = {s for s in sys for g in gol if g.equals(s, mode)}
            fps = set(sys) - tps
            fns = set()
            for g in gol:
                if not len([s for s in sys if s.equals(g, mode)]):
                    fns.add(g)
            for e in fps:
                print('FP: ' + str(e))
            for e in fns:
                print('FN:' + str(e))


class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, tag_type=None, mode='strict',
                 verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.scores = None
        if corpora.track == 1:
            self.track1(corpora)
        else:
            self.track2(corpora, tag_type, mode, verbose)

    def track1(self, corpora):
        """Compute measures for Track 1."""
        self.tags = ('ABDOMINAL',
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
        )
        self.scores = defaultdict(dict)
        Mrics = ('p', 'r', 'f1', 'specificity', 'auc')
        values = ('M', 'N')
        self.values = {'M': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                       'N': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}

        def evaluation(corpora, value, scores):
            predictions = defaultdict(list)
            for g, s in corpora.docs:
                for tag in self.tags:
                    predictions[tag].append(
                        (g.tags[tag].value == value, s.tags[tag].value == value))
            for tag in self.tags:
                # accumulate for micro overall measure
                self.values[value]['tp'] += predictions[tag].count((True, True))
                self.values[value]['fp'] += predictions[tag].count((False, True))
                self.values[value]['tn'] += predictions[tag].count((False, False))
                self.values[value]['fn'] += predictions[tag].count((True, False))
                # compute per-tag measures
                measures = Measures(tp=predictions[tag].count((True, True)),
                                    fp=predictions[tag].count((False, True)),
                                    tn=predictions[tag].count((False, False)),
                                    fn=predictions[tag].count((True, False)))
                scores[(tag, value, 'p')] = measures.precision()
                scores[(tag, value, 'r')] = measures.recall()
                scores[(tag, value, 'f1')] = measures.f1()
                scores[(tag, value, 'specificity')] = measures.specificity()
                scores[(tag, value, 'auc')] = measures.auc()
            return scores

        self.scores = evaluation(corpora, 'M', self.scores)
        self.scores = evaluation(corpora, 'N', self.scores)

        for measure in Mrics:
            for value in values:
                self.scores[('macro', value, measure)] = sum(
                    [self.scores[(t, value, measure)] for t in self.tags]) / len(self.tags)


def evaluate(corpora, mode='strict', verbose=False):
    """Run the evaluation by considering only files in the two folders."""
    assert mode in ('strict', 'lenient')
    evaluator_s = MultipleEvaluator(corpora, verbose)
    if corpora.track == 1:
        macro_f1, macro_auc = 0, 0
        print('{:*^96}'.format(' TRACK 1 '))
        print('{:20}  {:-^30}    {:-^22}    {:-^14}'.format('', ' M ',
                                                            ' N ',
                                                            ' overall '))
        print('{:20}  {:6}  {:6}  {:6}  {:6}    {:6}  {:6}  {:6}    {:6}  {:6}'.format(
            '', 'Prec.', 'Rec.', 'Speci.', 'F(b=1)', 'Prec.', 'Rec.', 'F(b=1)', 'F(b=1)', 'AUC'))
        for tag in evaluator_s.tags:
            print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
                tag.capitalize(),
                evaluator_s.scores[(tag, 'M', 'p')],
                evaluator_s.scores[(tag, 'M', 'r')],
                evaluator_s.scores[(tag, 'M', 'specificity')],
                evaluator_s.scores[(tag, 'M', 'f1')],
                evaluator_s.scores[(tag, 'N', 'p')],
                evaluator_s.scores[(tag, 'N', 'r')],
                evaluator_s.scores[(tag, 'N', 'f1')],
                (evaluator_s.scores[(tag, 'M', 'f1')] + evaluator_s.scores[(tag, 'N', 'f1')])/2,
                evaluator_s.scores[(tag, 'M', 'auc')]))
            macro_f1 += (evaluator_s.scores[(tag, 'M', 'f1')] + evaluator_s.scores[(tag, 'N', 'f1')])/2
            macro_auc += evaluator_s.scores[(tag, 'M', 'auc')]
        print('{:20}  {:-^30}    {:-^22}    {:-^14}'.format('', '', '', ''))
        m = Measures(tp=evaluator_s.values['M']['tp'],
                     fp=evaluator_s.values['M']['fp'],
                     fn=evaluator_s.values['M']['fn'],
                     tn=evaluator_s.values['M']['tn'])
        nm = Measures(tp=evaluator_s.values['N']['tp'],
                      fp=evaluator_s.values['N']['fp'],
                      fn=evaluator_s.values['N']['fn'],
                      tn=evaluator_s.values['N']['tn'])
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
            'Overall (micro)', m.precision(), m.recall(), m.specificity(),
            m.f1(), nm.precision(), nm.recall(), nm.f1(),
            (m.f1() + nm.f1()) / 2, m.auc()))
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
            'Overall (macro)',
            evaluator_s.scores[('macro', 'M', 'p')],
            evaluator_s.scores[('macro', 'M', 'r')],
            evaluator_s.scores[('macro', 'M', 'specificity')],
            evaluator_s.scores[('macro', 'M', 'f1')],
            evaluator_s.scores[('macro', 'N', 'p')],
            evaluator_s.scores[('macro', 'N', 'r')],
            evaluator_s.scores[('macro', 'N', 'f1')],
            macro_f1 / len(evaluator_s.tags),
            evaluator_s.scores[('macro', 'M', 'auc')]))
        print('{:>20}  {:^74}'.format('', '  {} files found  '.format(len(corpora.docs))))


class Corpora(object):

    def __init__(self, y_test, y_pred, track_num):
        """Initialize."""
        self.track = track_num
        self.y_test = y_test
        self.y_pred = y_pred
        # files1 = set(y_test)
        # files2 = set(y_pred)
        # common_files = files1 & files2     # intersection
        # if not common_files:
        #     print('ERROR: None of the files match.')
        # else:
        #     if files1 - common_files:
        #         print('Files skipped in {}:'.format(self.folder1))
        #         print(', '.join(sorted(list(files1 - common_files))))
        #     if files2 - common_files:
        #         print('Files skipped in {}:'.format(self.folder2))
        #         print(', '.join(sorted(list(files2 - common_files))))
        # for file in common_files:
        #     if track_num == 1:
        self.docs = []
        for i in xrange(len(self.y_test)):
            g = RecordTrack1(self.y_test[i]) # y_test is a list of list [[(tag, value), (tag, value)...],[...], ...]
            s = RecordTrack1(self.y_pred[i])
            self.docs.append((g, s))


def f1_score_measure(y_test, y_pred, track, verbose=False):
    """Main."""
    corpora = Corpora(y_test, y_pred, track)
    if corpora.docs:
        evaluate(corpora, verbose=verbose)
# y_test = [[('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')]]
# y_pred = [[('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'N')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'N'), ('ADVANCED-CAD', 'M')], [('ABDOMINAL', 'M'), ('ADVANCED-CAD', 'M')]]
# tag_list = ['ABDOMINAL', 'ADVANCED-CAD']

# f1_score_measure(y_test, y_pred, 1)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='n2c2: Track 1 evaluation script')
#     parser.add_argument('folder1', help='First data folder path (gold)')
#     parser.add_argument('folder2', help='Second data folder path (system)')
#     args = parser.parse_args()
#     main(os.path.abspath(args.folder1), os.path.abspath(args.folder2), 1)
