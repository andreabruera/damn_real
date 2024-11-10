import numpy
import os

with open(os.path.join(
                   'ALL_Trial_Data.txt',
                   )) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        ### just checking all's fine
        if len(line) != 9:
            continue
        if l_i == 0:
            header = [w.strip() for w in line]
            data = {h : list() for h in header}
            data['log_rt']  = list()
            continue
        if line[header.index('Resp')] == '':
            assert line[header.index('RT')] == ''
            print(line)
            continue
        for h_i, h in enumerate(header):
            if h == 'Task':
                mapper = {
                        'L' : 'lexical_decision',
                        'A' : 'Handlung',
                        'S' : 'Geraeusch',
                        }
                data[h].append(mapper[line[h_i]])
            elif h in ['AF', 'MF']:
                mapper = {
                        '' : 'NA',
                        '0' : '-1',
                        '1' : '1',
                        }
                data[h].append(mapper[line[h_i]])
            elif h == 'RT':
                data[h].append(line[h_i])
                data['log_rt'].append(numpy.log10(float(line[h_i])))
            elif h == 'TMS':
                mapper = {
                        'active' : 'pIPL',
                        'sham' : 'sham',
                        }
                data[h].append(mapper[line[h_i]])
            else:
                data[h].append(line[h_i])
### checking all went fine
items = set([len(v) for v in data.values()])
assert len(items) == 1
items = list(items)[0]

mapper = {
          'AF' : 'sound_word',
          'MF' : 'motor_word',
          'Exp_Resp' : 'expected_response',
          'Resp' : 'response',
          'TMS' : 'condition',
          'vpid' : 'subject',
          }
keys = [w for w in data.keys()]
with open('de_tms_sound-act_pipl.tsv', 'w') as o:
    ### title
    for k in keys:
        if k not in mapper.keys():
            o.write('{}\t'.format(k.lower()))
        else:
            o.write('{}\t'.format(mapper[k]))
    o.write('\n')
    ### data
    for _ in range(items):
        for k in keys:
            o.write('{}\t'.format(data[k][_]))
        o.write('\n')
