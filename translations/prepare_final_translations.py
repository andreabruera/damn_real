import os

de_to_en = dict()
with open('lancaster_english_to_german.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        de_to_en[line[1]] = line[0]
with open('fernandino_english_to_german.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        de_to_en[line[1]] = line[0]

with open('lanc_fern_de_to_en.tsv', 'w') as o:
    for d, e in de_to_en.items():
        o.write('{}\t{}\n'.format(d, e))

it_to_en = dict()
with open('en_to_it.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        it_to_en[line[1].lower()] = line[0]

with open('lanc_fern_it_to_en.tsv', 'w') as o:
    for d, e in it_to_en.items():
        o.write('{}\t{}\n'.format(d, e))
