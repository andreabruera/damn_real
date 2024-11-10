import fasttext
import logging
import multiprocessing
import os
import pickle
import random
import re
import string
import stop_words

from tqdm import tqdm

from utf_utils import transform_german_word

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def noisify_sentence(sentence):
    #print(sentence)
    noisified_sentence = list()
    for w in sentence:
        try:
            subst_prob = scores[w]
            assert w not in stopwords
            subst = random.choices([True, False], cum_weights=[subst_prob, subst_prob+(1-subst_prob)])[0]
            if subst:
                new_w = random.choice(model_vocabulary)
                #print([w, subst_prob, new_w])
                noisified_sentence.append(new_w)
                #print([w, idx_to_word[new_w], subst_prob])
            else:
                noisified_sentence.append(w)
        except KeyError:
            noisified_sentence.append(w)
    #print(noisified_sentence)
    return noisified_sentence


def process_file(in_f):
    sents = list()
    if 'opensubs_ready' in in_f:
        ### grouping by chunks of 512 tokens
        sentence = list()
        with open(in_f) as i:
            for l in i:
                line = re.sub(r'-', r'', l)
                line = re.sub('\W', ' ', line)
                line = re.sub('\s+', r' ', line)
                line = line.split()
                sentence.extend(line)
                if len(sentence) >= 256:
                    ### add noise in sentence
                    sentence = noisify_sentence(sentence)
                    sentence = re.sub('\s+', ' ', ' '.join(sentence))
                    #o.write('{}\n'.format(' '.join(sentence)))
                    #yield sentence
                    sents.append(sentence)
                    sentence = list()
            if len(sentence) > 1:
                sentence = noisify_sentence(sentence)
                #o.write('{}\n'.format(' '.join(sentence)))
                sentence = re.sub('\s+', ' ', ' '.join(sentence))
                #yield sentence
                sents.append(sentence)
    else:
        with open(in_f) as i:
            marker = True
            sentence = list()
            for l in i:
                line = l.strip().split('\t')
                if line[0][:4] == '</s>':
                    #print(sentence)
                    sentence = noisify_sentence(sentence)
                    sentence = re.sub('\s+', ' ', ' '.join(sentence))
                    #yield sentence
                    sents.append(sentence)
                    #o.write('{}\n'.format(' '.join(sentence)))
                    sentence = list()
                elif line[0][0] == '<':
                    continue
                if len(line) < 3:
                    continue
                #if line[5]=='P':
                #    continue
                if line[0] in string.punctuation:
                    continue
                else:
                    if '$' in line[1]:
                        continue
                    else:
                        sentence.append(line[0])
    print(len(sents))
    return sents

stopwords = stop_words.get_stop_words('de')

logging.info('now loading fasttext')
model = fasttext.load_model(os.path.join(
                                '/',
                                'import',
                                'cogsci',
                                'andrea',
                                'dataset',
                                'word_vectors',
                                'de',
                                'cc.de.300.bin',
                                )
                                )
model_vocabulary = model.words

in_fs = list()
corpora = [
           'opensubs_ready', 
           'sdewac-v3-tagged_smaller_files'
           ]
for c in corpora:
    corpus_folder = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'corpora', 'de', c)
    for f in os.listdir(corpus_folder):
        in_fs.append(os.path.join(corpus_folder, f))

### reading words to be left out
with open(os.path.join('pickles', 'cn_sims.pkl'), 'rb') as i:
    all_sims = pickle.load(i)

pruning = 0.01
### pruning
all_sims = {k : {w.replace('_', '') : val for w, val in v.items() if w.replace('_', '') not in stopwords} for k, v in all_sims.items()}
all_sims = {k : {w : prob for w, prob in sorted(v.items(), key=lambda item : item[1], reverse=True)[:int(len(v.values())*pruning)]} for k, v in all_sims.items()}
mins = {k : min(v.values()) for k, v in all_sims.items()}
maxs = {k : max(v.values()) for k, v in all_sims.items()}
### rescaling between 0.5-0.75
all_sims = {k : {w : 0.25*((val - mins[k]) / (maxs[k]-mins[k]))+0.5 for w, val in v.items()} for k, v in all_sims.items()}  

full_sims = {k : {w_spelling : prob for w, prob in v.items() for w_spelling in transform_german_word(w)} for k, v in all_sims.items()}
full_sims['undamaged'] = dict()

for case, scores in full_sims.items():
    print(case)
    debugging = False
    #debugging = True
    if debugging:
        for f in tqdm(in_fs):
            results = process_file(f)
    else:
        with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
            results = pool.map(process_file, in_fs)
            pool.terminate()
            pool.join()

    out_folder = os.path.join(
                              '/', 
                              'import', 
                              'cogsci', 
                              'andrea', 
                              'dataset', 
                              'corpora', 
                              'de', 
                              'ready_for_fasttext_{}'.format(pruning),
                              )
    os.makedirs(out_folder, exist_ok=True)

    out_file = os.path.join(out_folder, '{}_wac_subs_for_fasttext.txt'.format(case))
    with open(out_file, 'w') as o:
        for lst in tqdm(results):
            for l in lst:
                o.write('{}\n'.format(l))
