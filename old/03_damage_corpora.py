import gensim
import logging
import math
import multiprocessing
import numpy
import os
import pickle
import random
import re
import sklearn
import stop_words
import string

from gensim.models import Word2Vec
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from utils import prepare_input_output_folders, read_args

def noisify_sentence(sentence):
    #print(sentence)
    noisified_sentence = list()
    idxs = [ready_vocab[w] for w in sentence]
    for w in sentence:
        idx = ready_vocab[w]
        if idx == 0:
            noisified_sentence.append(w)
            continue
        try:
            subst_prob = sensorimotor_scores[idx]
        except KeyError:
            noisified_sentence.append(w)
            continue
        subst = random.choices([True, False], cum_weights=[subst_prob, subst_prob+(1-subst_prob)])[0]
        if subst:
            new_w = random.choices(word_idxs, cum_weights=word_cum_inverse_weights, k=1)[0]
            noisified_sentence.append(idx_to_word[new_w])
            #print([w, idx_to_word[new_w], subst_prob])
        else:
            noisified_sentence.append(w)
    #print(noisified_sentence)
    return noisified_sentence


def process_file(file_names):
    in_f = file_names[0]
    out_f = file_names[1]
    with open(out_f, 'w') as o:
        if 'opensubs_ready' in in_f:
            ### grouping by chunks of 512 tokens
            sentence = list()
            with open(in_f) as i:
                for l in i:
                    line = re.sub(r'-', r'', l)
                    line = re.sub('\W', ' ', line)
                    line = re.sub('\s+', r' ', line)
                    line = line.lower().split()
                    sentence.extend(line)
                    if len(sentence) >= 512:
                        ### add noise in sentence
                        sentence = noisify_sentence(sentence)
                        o.write('{}\n'.format(' '.join(sentence)))
                        sentence = list()
                if len(sentence) > 1:
                    sentence = noisify_sentence(sentence)
                    o.write('{}\n'.format(' '.join(sentence)))
                    #yield(sentence)
        else:
            with open(in_f) as i:
                marker = True
                sentence = list()
                for l in i:
                    line = l.strip().split('\t')
                    if line[0][:4] == '</s>':
                        sentence = noisify_sentence(sentence)
                        #yield sentence
                        o.write('{}\n'.format(' '.join(sentence)))
                        sentence = list()
                    elif line[0][0] == '<':
                        continue

                    if len(line) < 5:
                        continue
                    if line[5]=='P':
                        continue
                    if line[0] in string.punctuation:
                        continue
                    else:
                        if '$' in line[1]:
                            continue
                        else:
                            sentence.append(line[0].lower())

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

args = read_args(mode='damage')

relu_base = int(args.relu_base)/100

### removing stopwords

stopwords = stop_words.get_stop_words('english')
stopwords = list(set(stopwords + ["'{}".format(w.split("'")[1]) for w in stopwords if "'" in w]))
#print(stopwords)

### using fasttext
predicted_file = os.path.join('predictions', 'sensory_predicted_fasttext.tsv')
assert os.path.exists(predicted_file)
training_file = os.path.join('predictions', 'sensory_training_fasttext.tsv')
assert os.path.exists(predicted_file)

### reading frequencies
ratings = dict()
for corpus in ['opensubs', 'wac']:
    freqs = pickle.load(open(os.path.join(
                                 '..', 
                                 'psychorpus', 
                                 'pickles', 
                                 args.language, 
                                 '{}_{}_word_freqs.pkl'.format(
                                             args.language, 
                                             corpus
                                             ),
                                 ), 
                                 'rb')
                                 )
    for original_k, v in freqs.items():
        k = original_k.lower()
        ### frequencies are not lowercased
        if k not in ratings.keys():
            ratings[k] = v
        else:
            ratings[k] += v

### thresholding at 50 occurrences
logging.info('now creating the vocabulary dictionary')
ready_vocab = {k[0] : k_i if k[1]>50 else 0 for k_i, k in enumerate(ratings.items())}
idx_to_word = {v : k for k, v in ready_vocab.items()}

if args.semantic_modality == 'auditory':
    relevant_key = 'Auditory.mean'
elif args.semantic_modality == 'action':
    raise RuntimeError('To be implemented')

sensorimotor_scores = dict()

with tqdm() as line_counter:
    for f in [training_file, predicted_file]:
        with open(f) as i:
            counter = 0
            for l in i:
                line = l.strip().split('\t')
                if counter == 0:
                    header = line.copy()
                    counter += 1
                    continue
                assert len(line) == len(header)
                word_idx = header.index('word')
                if line[word_idx] in stopwords:
                    continue
                if len(line[word_idx]) < 2:
                    continue
                if line[word_idx].isnumeric():
                    continue
                key_idx = header.index(relevant_key)
                val = max(0, min(5., float(line[key_idx]))) / 5.
                if args.function == 'sigmoid':
                    val = 1 / (1+ math.e**-val)
                elif args.function == 'exponential':
                    val = math.e**val
                elif args.function == 'logarithmic':
                    if val > 0.1:
                        val = math.log(val)
                try:
                    sensorimotor_scores[ready_vocab[line[word_idx]]] = val
                    line_counter.update(1)
                except KeyError:
                    continue

if args.function == 'relu-raw':
    threshold = numpy.quantile(list(sensorimotor_scores.values()), relu_base)
    print(threshold)
    sensorimotor_scores = {k : v if v > threshold else 0. for k, v in sensorimotor_scores.items()}
if 'relu-raw-thresholded' in args.function:
    threshold = numpy.quantile(list(sensorimotor_scores.values()), relu_base)
    print(threshold)
    sensorimotor_scores = {k : min(v, int(args.function[-2:])/100) if v > threshold else 0. for k, v in sensorimotor_scores.items()}
if args.function == 'relu-step':
    threshold = numpy.quantile(list(sensorimotor_scores.values()), relu_base)
    print(threshold)
    sensorimotor_scores = {k : random.randint(85, 95)/100 if v > threshold else 0. for k, v in sensorimotor_scores.items()}
elif args.function == 'relu-exponential':
    threshold = numpy.quantile(list(sensorimotor_scores.values()), relu_base)
    print(threshold)
    sensorimotor_scores = {k : math.e**v if v > threshold else 0. for k, v in sensorimotor_scores.items()}
elif args.function == 'relu-logarithmic':
    threshold = numpy.quantile(list(sensorimotor_scores.values()), relu_base)
    print(threshold)
    sensorimotor_scores = {k : math.log(v) if v > threshold else 0. for k, v in sensorimotor_scores.items()}
elif args.function == 'relu-sigmoid':
    threshold = numpy.quantile(list(sensorimotor_scores.values()), relu_base)
    print(threshold)
    sensorimotor_scores = {k : 1 / (1+ math.e**-val) if v > threshold else 0. for k, v in sensorimotor_scores.items()}
if 'exponential' in args.function or 'sigmoid' in args.function or 'logarithmic' in args.function:
    ### need scaling in 0 -1
    v_max = max(sensorimotor_scores.values())
    v_min = min(sensorimotor_scores.values())
    sensorimotor_scores = {k : (v - v_min) / (v_max - v_min) for k, v in sensorimotor_scores.items()}
if 'step' not in args.function and 'threshold' not in args.function:
    v_max = max(sensorimotor_scores.values())
    assert v_max == 1.
v_min = min(sensorimotor_scores.values())
assert v_min == 0.

### preparing two separate lists: idxs and weights
logging.info('now preparing the weights for sampling')
word_idxs = list()

word_raw_weights = list()
word_inverse_weights = list()

word_cum_weights = list()
word_cum_inverse_weights = list()

initial = 0
initial_inverse = 0
for k, v in tqdm(sensorimotor_scores.items()):
    #print(v)
    word_idxs.append(k)
    word_raw_weights.append(v)
    word_inverse_weights.append(1-v)
    initial += v
    initial_inverse += 1-v
    word_cum_weights.append(initial)
    word_cum_inverse_weights.append(initial_inverse)

logging.info('now finally damaging the data!')
file_names, model_file = prepare_input_output_folders(args)

if args.debugging:
    for f in tqdm(file_names):
        process_file(f)
else:
    with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
        pool.map(process_file, file_names)
        pool.terminate()
        pool.join()
