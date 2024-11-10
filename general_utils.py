import numpy
import os
import pickle
import random
import re
import scipy

from scipy import stats
from tqdm import tqdm

def read_lancaster_ratings():
    norms = dict()
    ### sensory ratings
    file_path = os.path.join(
                             '..',
                             'ratings',
                             'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                             )
    assert os.path.exists(file_path)
    with open(file_path) as i:
        counter = 0
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w.lower() for w in line]
                relevant_keys = [w.lower() for w in line if '.mean' in w or w=='Minkowski3.perceptual']
                continue
            assert len(line) == len(header)
            word = line[0].strip().lower()
            marker = False
            for k in relevant_keys:
                if 'inkow' in k:
                    continue
                try:
                    assert float(line[header.index(k)]) <= 5 
                except AssertionError:
                    ### correcting for a silly mistake in the ratings...
                    line[header.index(k)] = '.{}'.format(line[header.index(k)])
            if word not in norms.keys():
                norms[word] = dict()
            for k in relevant_keys:
                var = k.split('.')[0]
                val = float(line[header.index(k)])
                ### minimum is 0, max is 5
                if 'inkow' not in k:
                    assert val >= 0. and val <= 5.
                    curr_val = float(val) / 5
                else:
                    curr_val = float(val)
                norms[word][var] = curr_val
    '''
    ### concreteness
    if concreteness:
        norms['concreteness'] = dict()
        with open(os.path.join('data', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                ### minimum is 1, max is 5
                val = float(line[2])
                assert val >= 1. and val <= 5.
                curr_val = (val - 1) / (5 - 1)
                w = line[0].lower().strip()
                if w in norms['visual'].keys():
                    norms['concreteness'][w] = curr_val
    '''

    return norms

def build_ppmi_vecs(coocs, vocab, row_words, col_words, smoothing=False, power=1.):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in col_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(col_words)
    if power != 1.:
        pmi_mtrx = numpy.power(pmi_mtrx, power)
    #matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0])
    axis_one_sum = pmi_mtrx.sum(axis=1)
    #axis_one_mtrx = numpy.divide(1, axis_one_sum, where=axis_one_sum!=0).reshape(-1, 1)
    axis_one_mtrx = numpy.array([1/val if val!=0 else val for val in axis_one_sum]).reshape(-1, 1)
    assert True not in numpy.isnan(axis_one_mtrx)
    axis_zero_sum = pmi_mtrx.sum(axis=0)
    #axis_zero_mtrx = numpy.divide(1, axis_zero_sum, where=axis_zero_sum!=0).reshape(1, -1)
    axis_zero_mtrx = numpy.array([1/val if val!=0 else val for val in axis_zero_sum]).reshape(1, -1)
    assert True not in numpy.isnan(axis_one_mtrx)
    ### raising to 0.75 as suggested in Levy & Goldberg 2015
    if smoothing:
        total_sum = numpy.power(pmi_mtrx, 0.75).sum()
    else:
        total_sum = pmi_mtrx.sum()
    #trans_pmi_mtrx = numpy.multiply(numpy.multiply(numpy.multiply(pmi_mtrx,1/pmi_mtrx.sum(axis=1).reshape(-1, 1)), 1/pmi_mtrx.sum(axis=0).reshape(1, -1)), pmi_mtrx.sum())
    trans_pmi_mtrx = numpy.multiply(
                                    numpy.multiply(
                                                   numpy.multiply(
                                                                  pmi_mtrx,axis_one_mtrx), 
                                                   axis_zero_mtrx), 
                                    total_sum)
    trans_pmi_mtrx[trans_pmi_mtrx<1.] = 1
    assert True not in numpy.isnan(trans_pmi_mtrx.flatten())
    ### checking for nans
    trans_pmi_vecs = {w : numpy.log2(trans_pmi_mtrx[w_i]) for w_i, w in enumerate(row_words)}
    for v in trans_pmi_vecs.values():
        assert True not in numpy.isnan(v)

    return trans_pmi_vecs
