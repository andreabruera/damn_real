import argparse
import gensim
import logging
import math
import multiprocessing
import numpy
import os
import pickle
import random
import sklearn
import stop_words
import string

from gensim.models import Word2Vec
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from utils import prepare_input_output_folders, read_args

class DamagedCorpora:
    def __init__(self, file_names):
        self.file_names = [f[1] for f  in file_names]
    def __iter__(self):
        for file_name in self.file_names:
            assert setup_info in file_name
            with open(file_name) as i:
                for l in i:
                    line = l.strip().split()
                    if len(line) > 3:
                        yield line

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

args = read_args(mode='training')

logging.info('now loading the folders!')
file_names, setup_info = prepare_input_output_folders(args)

os.makedirs('models', exist_ok=True)
model_file = os.path.join(
                        'models', 
                        "word2vec_{}_damaged_{}_param-mandera2017_min-count-50.model".format(
                                   args.language, 
                                   setup_info)
                        )

model = Word2Vec(
                 sentences=DamagedCorpora(file_names), 
                 size=300, 
                 window=6, 
                 workers=int(os.cpu_count()),
                 min_count=50,
                 negative=10,
                 sg=0,
                 sample=1e-5,
                 )
model.save(model_file)
