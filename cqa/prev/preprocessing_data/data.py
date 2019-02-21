#########################################################################
# File Name: data.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年10月12日 星期五 14时42分12秒
#########################################################################

import numpy as np, sys, math, os
import pickle
import time
import numpy as np
import json
Home = '/home/wwang/Projects/QAR_data'


run_aids = 1
run_pair_mode = ['all', 'pos-neg', 'sample']

def main():
    print('hello world, data.py')
    ds = 'test'
    if len(sys.argv) > 1:
        ds = sys.argv[-1]
    home = '{}/{}_data'.format(Home, ds)
    to_dir = 'data'
    os.system('mkdir {}/{}'.format(home, to_dir))

    train_qids = pickle.load(open('{}/raw/train_qids_list.pkl'.format(home), 'rb'))
    vali_qids = pickle.load(open('{}/raw/vali_qids_list.pkl'.format(home), 'rb'))
    test_qids = pickle.load(open('{}/raw/test_qids_list.pkl'.format(home), 'rb'))
    qid2vau = pickle.load(open('{}/raw/qid2vote_aid_uid_dict.pkl'.format(home), 'rb'))
    aid2vote = pickle.load(open('{}/raw/aid2vote_dict.pkl'.format(home), 'rb'))
    print('load ok')

    if run_aids:
        np.random.seed(123456)
        aid2qu = {}
        for qids, name in ((train_qids, 'train'), (vali_qids, 'vali'), (test_qids, 'test')):
            aids = []
            for qid in qids:
                for v, a, u in qid2vau[qid]:
                    aid2qu[a] = (qid, u)
                    aids.append(a)
            np.random.shuffle(aids)
            pickle.dump(aids, open('{}/{}/{}_aids_list.pkl'.format(home, to_dir, name), 'wb'), protocol = 4)
        pickle.dump(aid2qu, open('{}/{}/aid2qu_dict.pkl'.format(home, to_dir), 'wb'), protocol = 4)

    for mode in run_pair_mode:
        np.random.seed(123456)
        def gen_pairs(vas):
            vas = sorted(vas, key = lambda x: -x[0])
            #print(vas)
            n = len(vas)
            if mode == 'all':
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if vas[i][0] > vas[j][0]:
                            yield (vas[i][1], vas[j][1])
            elif mode == 'pos-neg':
                m = 3
                for i in range(n - 1):
                    j = i + 1
                    while j < n and vas[i][0] <= vas[j][0]:
                        j += 1
                    r = min(m, n -  j)
                    st = set()
                    while len(st) < r:
                        k = np.random.randint(j, n)
                        if k in st:
                            continue
                        yield (vas[i][1], vas[k][1])
                        st.add(k)
            elif mode == 'sample':
                ij = []
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if vas[i][0] > vas[j][0]:
                            ij.append((i, j))
                np.random.shuffle(ij)
                ij = ij[: n * 3]
                for i, j in ij:
                    yield (vas[i][1], vas[j][1])

        pairs = []
        for qid in train_qids:
            pairs.extend(list(gen_pairs(qid2vau[qid])))
        np.random.shuffle(pairs)
        #print(pairs)
        pickle.dump(pairs, open('{}/{}/train_aid_pairs_{}_list.pkl'.format(home, to_dir, mode), 'wb'), protocol = 4)



if __name__ == '__main__':
    main()

