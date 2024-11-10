import numpy
import os
import random
import re
import torch

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class ContextualizedModelCard:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.model_name, self.to_cuda = self.read_names(args)
        self.cuda_device = 'cuda:{}'.format(0)
        self.model = self.load_model()
        self.required_shape, self.max_len, self.n_layers = self.read_details()
        self.tokenizer = self.load_tokenizer()

    def read_names(self, args):
        to_cuda = False
        ### llama
        if args.model == 'llama-1b':
            model_name = "meta-llama/Llama-3.2-1B"
            to_cuda = True
        ### llama
        if args.model == 'llama-3b':
            model_name = "meta-llama/Llama-3.2-3B"
            to_cuda = True
        ### XLM-roberta
        if 'large' in args.model:
            model_name = args.model
            to_cuda =True
        if 'xxl' in args.model:
            model_name = 'facebook/{}'.format(args.model)
            to_cuda = False
        if 'rta-xl' in args.model:
            model_name = 'facebook/{}'.format(args.model)
            to_cuda = True
        ### XGLM
        if args.model == 'xglm-564m':
            model_name = "facebook/xglm-564M"
            to_cuda = True
        if args.model == 'xglm-1.7b':
            model_name = "facebook/xglm-1.7B"
            if torch.cuda.get_device_properties(int(0)).total_memory/1e6 > 20000:
                to_cuda = True
        if args.model == 'xglm-2.9b':
            model_name = "facebook/xglm-2.9B"
            if torch.cuda.get_device_properties(int(0)).total_memory/1e6 > 20000:
                to_cuda = True
        if args.model == 'xglm-4.5b':
            model_name = "facebook/xglm-4.5B"
            if torch.cuda.get_device_properties(int(0)).total_memory/1e6 > 20000:
                to_cuda = True
        if args.model == 'xglm-7.5b':
            model_name = "facebook/xglm-7.5B"

        return model_name, to_cuda

    def load_model(self):
        cache = os.path.join('/', 'data', 'tu_bruera', 'hf_models')
        os.makedirs(cache, exist_ok=True)
        model = AutoModel.from_pretrained(
                                          self.model_name, 
                                          cache_dir=cache,
                                          )
        if self.to_cuda:
            model.to(self.cuda_device)
        return model

    def read_details(self):
        if 'xglm' in self.model_name:
            required_shape = self.model.config.d_model
            max_len = self.model.config.max_position_embeddings
            n_layers = self.model.config.num_layers
        elif 'xlm' in self.model_name or 'llama' in self.model_name:
            required_shape = self.model.config.hidden_size
            max_len = self.model.config.max_position_embeddings
            n_layers = self.model.config.num_hidden_layers
        print('Dimensionality: {}'.format(required_shape))
        print('Number of layers: {}'.format(n_layers))

        return required_shape, max_len, n_layers

    def load_tokenizer(self):
        cache = os.path.join('/', 'data', 'tu_bruera', 'hf_tokenizers')
        os.makedirs(cache, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
                                                  self.model_name, 
                                                  cache_dir=cache,
                                                  sep_token='[SEP]',
                                                  max_length=self.max_len,
                                                  truncation=True,
                                                  clean_up_tokenization_spaces=True,
                                                  )
        return tokenizer

def read_all_sentences(args):
    ### reducing words...
    ### reading all words for a certain language
    w_path = os.path.join('words', args.lang)
    words = list()
    for f in os.listdir(w_path):
        if 'tsv' not in f:
            continue
        print(f)
        #    continue
        #if 'mitchell' not in f:
        #    continue
        #if 'social' not in f and 'distr' not in f:
        #if 'blind' not in f and 'anew' not in f and 'behav' not in f:
        #if 'prod' not in f and 'sound' not in f and 'phon' not in f:
        if 'dirani' not in f and 'kaneshiro' not in f:
            continue
        with open(os.path.join(w_path, f)) as i:
            for l in i:
                w = l.strip()
                if w != '':
                    words.append(w)
    all_sentences = dict()
    sentences_folder = os.path.join('sentences', args.lang, args.corpus)

    for f in os.listdir(sentences_folder):
        if 'tsv' not in f:
            continue
        word = f.split('.')[0]
        if word not in words:
            continue
        with open(os.path.join(sentences_folder, f)) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
            ### removing cases where less than 5 words come before the actual word
            for l in lines:
                if l.index(word) < 5:
                    continue
                ### adding a marker
                sep_l = [w.strip() if w!=word else '[SEP] {}[SEP]'.format(w.strip()) for w in l]
                ### keeping only reasonable sentences...
                if len(sep_l) > 128:
                    continue
                assert len(sep_l) >= 1
                joint_l = ' '.join(sep_l)
                ### websites are crap...
                if 'http' in joint_l:
                    continue
                ### html too...
                if 'href' in joint_l:
                    continue
                try:
                    all_sentences[word].append(joint_l)
                except KeyError:
                    all_sentences[word] = [joint_l]
    for k, v in all_sentences.items():
        assert len(v) >= 1
    random.seed(11)
    ### sampling 20, so as to avoid bad surprises when extracting...
    all_sentences = {k : random.sample(v, k=min(20, len(v))) for k, v in all_sentences.items()}
    return all_sentences

def extract_vectors(args, model_card, sentences):

    entity_vectors = dict()
    entity_sentences = dict()

    with tqdm() as pbar:
        for stimulus, stim_sentences in sentences.items():
            print(stimulus)
            #entity_vectors[stimulus] = list()
            #entity_sentences[stimulus] = list()
            assert len(stim_sentences) >= 1
            for l_i, l in enumerate(stim_sentences):
                #print(l)

                inputs = model_card.tokenizer(
                                   l, 
                                   return_tensors="pt",
                                   truncation_strategy='longest_first', 
                                   max_length=int(model_card.max_len*0.75), 
                                   truncation=True,
                                   )
                spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                        i==model_card.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]]
                if len(spans) % 2 != 0:
                    continue
                try:
                    check_tokens = inputs['input_ids'].numpy().reshape(-1)[spans[0]+1:spans[1]]
                except IndexError:
                    continue
                check_str = ''.join(model_card.tokenizer.convert_ids_to_tokens(check_tokens))
                if stimulus not in check_str:
                    continue
                del inputs
                old_l = '{}'.format(l)
                l = re.sub(r'\[SEP\]', ' ', l)
                l = re.sub('\s+', r' ', l)
                inputs = model_card.tokenizer(
                                   l, 
                                   return_tensors="pt",
                                   truncation_strategy='longest_first', 
                                   #max_length=model_card.max_len, 
                                   max_length=int(model_card.max_len*0.75), 
                                   truncation=True,
                                   )
                if inputs['input_ids'][0].shape[0] > model_card.max_len:
                    print('tokenization error with sentence (length {}):'.format(len(model_card.tokenizer.tokenize(l))))
                    print(l)
                    continue
                ### Correcting spans
                correction = list(range(1, len(spans)+1))
                spans = [max(0, s-c) for s,c in zip(spans, correction)]
                split_spans = list()
                for i in list(range(len(spans)))[::2]:
                    try:
                        if 'xlm' in model_card.model_name:
                            if 'erta-xl' in model_card.model_name:
                                current_span = (spans[i], spans[i+1])
                            else:
                                current_span = (spans[i]+1, spans[i+1]+1)
                        elif 'llama' in model_card.model_name:
                            #current_span = (spans[i], spans[i+1]+1)
                            current_span = (spans[i], spans[i+1])
                            #if spans[i] == 0:
                            #    current_span = (spans[i], spans[i+1]+1)
                            #else:
                            #    current_span = (spans[i], spans[i+1])
                        elif 'xglm' in model_card.model_name:
                            if '564m' in args.model:
                                if spans[i] == 0:
                                    #current_span = (spans[i]+2, spans[i+1]+3)
                                    current_span = (spans[i]+2, spans[i+1]+2)
                                else:
                                    #current_span = (spans[i]+1, spans[i+1]+2)
                                    current_span = (spans[i]+1, spans[i+1]+1)
                            else:
                                if spans[i] == 0:
                                    #current_span = (spans[i]+1, spans[i+1]+2)
                                    current_span = (spans[i]+1, spans[i+1]+1)
                                else:
                                    #current_span = (spans[i], spans[i+1]+1)
                                    current_span = (spans[i], spans[i+1])
                        #error = inputs['input_ids'][0][current_span[0]:current_span[1]]
                        #error_str = ''.join(model_card.tokenizer.convert_ids_to_tokens(error))
                        #print(error_str)
                    except IndexError:
                        print('there was a mistake with: {}'.format(l))
                        continue
                    ### final checks...
                    marker = True
                    for c_i in range(len(check_tokens)):
                        try:
                            if inputs['input_ids'][0][current_span[0]+c_i] != check_tokens[c_i]:
                                marker = False
                        except IndexError:
                            marker = False
                    if marker == False:
                        continue
                    if len(range(current_span[0], current_span[1])) > len(check_tokens):
                        current_span[1] = current_span[0]+len(check_tokens)+1
                    split_spans.append(current_span)
                del inputs
                ### taking only right-most mention
                if len(split_spans) > 1:
                    split_spans = [split_spans[-1]]
                try:
                    inputs = model_card.tokenizer(
                                       l, 
                                       return_tensors="pt",
                                       truncation_strategy='longest_first', 
                                       #max_length=model_card.max_len, 
                                       max_length=int(model_card.max_len*0.75), 
                                       truncation=True,
                                       )
                    if model_card.to_cuda:
                        inputs.to(model_card.cuda_device)
                except RuntimeError:
                    del inputs
                    print('input error')
                    print(l)
                    continue
                try:
                    outputs = model_card.model(
                            **inputs, 
                                    output_attentions=False,
                                    output_hidden_states=True, 
                                    return_dict=True,
                                    )
                except RuntimeError:
                    del inputs
                    print('output error')
                    print(l)
                    continue

                hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                del outputs
                #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                for beg, end in split_spans:
                    ### If there are less than two tokens that must be a mistake
                    if len(inputs['input_ids'][0][beg:end]) < 1:
                        continue
                    tkns = model_card.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][beg:end])
                    #if stimulus not in ''.join(tkns):
                    #    continue
                    print(tkns)
                    mention = hidden_states[:-1, beg:end, :]
                    mention = numpy.average(mention, axis=1)
                    #mention = mention[layer_start:layer_end, :]

                    #mention = numpy.average(mention, axis=0)
                    assert mention.shape == (model_card.n_layers, model_card.required_shape, )
                    try:
                        entity_vectors[stimulus].append(mention)
                        entity_sentences[stimulus].append(old_l)
                    except KeyError:
                        entity_vectors[stimulus] = [mention]
                        entity_sentences[stimulus] = [old_l]
                    pbar.update(1)
                del inputs
    for k, v in entity_vectors.items():
        for mention in v:
            assert mention.shape == (model_card.n_layers, model_card.required_shape, )

    return entity_vectors, entity_sentences
