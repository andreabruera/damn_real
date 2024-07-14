import numpy
import os


f_path = 'word_similarity_relatedness/sim-rel_results/de/fasttext/fasttext/'
for f in os.listdir(f_path):
    with open(os.path.join(f_path, f)) as i:
        lines = i.readlines()
        assert len(lines) ==1
        line = lines[0].strip().split('\t')
        if 'sound-act' not in line[2]:
            continue
        if 'actiontask' not in line[2]:
            continue
        print(line[2])
        print(numpy.average([float(v) for v in line[3:] if v!='']))
        print('\n')

