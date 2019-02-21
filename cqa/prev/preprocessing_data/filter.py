#########################################################################
# File Name: filter.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年10月12日 星期五 09时12分35秒
#########################################################################

import numpy as np, sys, math, os
import pickle
import time
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer

Home = '/home/wwang/Projects/QAR_data'

def ft(qau, qid2text, aid2text):
    print('input len:', len(qau))
    
    aid_t = {}
    qid_t = {}

    u_cnt = {}
    q_cnt = {}

    for q, a, u in qau:
        if not u: continue
        qid_t[q] = qid2text[q]
        aid_t[a] = aid2text[a]

        u_cnt.setdefault(u, 0)
        q_cnt.setdefault(q, 0)
        u_cnt[u] += 1
        q_cnt[q] += 1
    
    ids = list(qid_t.keys()) + list(aid_t.keys())
    texts = list(qid_t.values()) + list(aid_t.values())
    nb_q = len(qid_t)
    nb_a = len(aid_t)
    print('nb_q, nb_a:', nb_q, nb_a)
    #texts = texts[:10]
    zero_time = time.time()
    cv = CountVectorizer(token_pattern = token_pattern, min_df = min_df)
    tlen = np.array(cv.fit_transform(texts).sum(1).T)[0]

    qid_tlen = dict(zip(ids[:nb_q], tlen[:nb_q]))
    aid_tlen = dict(zip(ids[nb_q:], tlen[nb_q:]))
    word_dict = cv.vocabulary_
    print('nb_words', len(word_dict))
    print('fit_transform over, time: {:.2f}s'.format(time.time() - zero_time))
    
    
    ok = True
    new_qau = []
    for q, a, u in qau:
        if not u: continue
        if u_cnt[u] >= min_u_cnt and q_cnt[q] >= min_q_cnt and aid_tlen[a] >= min_tlen and qid_tlen[q] >= min_tlen:
            new_qau.append((q, a, u))
        else:
            ok = False
    print('output len', len(new_qau)); print()
    return new_qau, word_dict, ok


def filter_dict(keys, d):
    ret = {}
    for k in keys:
        if k in d:
            ret[k] = d[k]
    return ret

run_test = True
run_test = False

min_u_cnt = 5
min_q_cnt = 5
min_df = 10
min_tlen = 5
p = 100
N = 100
token_pattern = '[\u4E00-\u9FA50-9a-zA-Z]\S*'

if run_test:
    min_u_cnt = 2
    min_q_cnt = 2
    min_df = 1
    min_tlen = 2

def main():
    print('hello world, filter.py')
    ds = 'so'
    if len(sys.argv) > 1:
        ds = sys.argv[-1]
    home = '{}/raw{}_data/raw'.format(Home, ds)

    qau = pickle.load(open('{}/qau_list.pkl'.format(home), 'rb'))
    qid2text = pickle.load(open('{}/qid2title_dict.pkl'.format(home), 'rb'))
    aid2text = pickle.load(open('{}/aid2text_dict.pkl'.format(home), 'rb'))
    aid2vote = pickle.load(open('{}/aid2vote_dict.pkl'.format(home), 'rb'))
    print('load ok')

    to_dir = '{}/{}_data/raw'.format(Home, ds)
    if run_test:
        qau = qau[:100]
        to_dir = '{}/test_data/raw'.format(Home)

    os.system('mkdir {}'.format('/'.join(to_dir.split('/')[:-1])))
    os.system('mkdir {}'.format(to_dir))
    np.random.seed(123456)

    print('nb_raw', len(qau))
    qau = [e for e in qau if np.random.randint(N) < p and e[0] in qid2text]
    print('after random filter', len(qau))

    while True:
        qau, word_dict, ok = ft(qau, qid2text, aid2text)
        if ok:
            break

    qid_set = set()
    aid_set = set()
    uid_set = set()
    qid2vote_aid_uid = {}
    for q, a, u in qau:
        qid_set.add(q)
        aid_set.add(a)
        uid_set.add(u)
        qid2vote_aid_uid.setdefault(q, [])
        qid2vote_aid_uid[q].append((aid2vote[a], a, u))
    print('nb_q', len(qid_set))
    print('nb_a', len(aid_set))
    print('nb_u', len(uid_set))
    uid2int = dict(zip(sorted(uid_set), range(len(uid_set))))

    qid_list = sorted(qid_set)
    np.random.shuffle(qid_list)

    nb_questions = len(qid_set)
    nb_train = nb_questions // 10 * 8
    nb_vali = (nb_questions - nb_train) // 2
    nb_test = nb_questions - nb_train - nb_vali
    train_qids = qid_list[:nb_train]
    vali_qids = qid_list[nb_train: nb_train + nb_vali]
    test_qids = qid_list[nb_train + nb_vali:]


    qid2text = filter_dict(qid_set, qid2text)
    aid2text = filter_dict(aid_set, aid2text)
    aid2vote = filter_dict(aid_set, aid2vote)
    for k in word_dict.keys():
        word_dict[k] = int(word_dict[k])

    pickle.dump(qid2text, open('{}/qid2text_dict.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(aid2text, open('{}/aid2text_dict.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(aid2vote, open('{}/aid2vote_dict.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(qid2vote_aid_uid, open('{}/qid2vote_aid_uid_dict.pkl'.format(to_dir), 'wb'), protocol = 4)

    #pickle.dump(qau, open('{}/qau_list.pkl'.format(to_dir), 'wb'), protocol = 4)

    pickle.dump(qid_set, open('{}/qid_set.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(aid_set, open('{}/aid_set.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(uid_set, open('{}/uid_set.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(word_dict, open('{}/word_dict.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(uid2int, open('{}/uid2int_dict.pkl'.format(to_dir), 'wb'), protocol = 4)
    json.dump(word_dict, open('{}/word_dict.json'.format(to_dir), 'w'), indent = 2, sort_keys = True)

    pickle.dump(train_qids, open('{}/train_qids_list.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(vali_qids, open('{}/vali_qids_list.pkl'.format(to_dir), 'wb'), protocol = 4)
    pickle.dump(test_qids, open('{}/test_qids_list.pkl'.format(to_dir), 'wb'), protocol = 4)

    numbers = {}
    numbers['nb_words'] = len(word_dict)
    numbers['nb_questions'] = len(qid2text)
    numbers['nb_answers'] = len(aid2text)
    numbers['min_u_cnt'] = min_u_cnt
    numbers['min_q_cnt'] = min_q_cnt
    numbers['min_tlen'] = min_tlen
    numbers['nb_train'] = nb_train
    numbers['nb_vali'] = nb_vali
    numbers['nb_test'] = nb_test
    pickle.dump(numbers, open('{}/numbers_dict.pkl'.format(to_dir), 'wb'), protocol = 4)
    json.dump(numbers, open('{}/numbers.json'.format(to_dir), 'w'), indent = 2, sort_keys = True)
    print(json.dumps(numbers, indent = 2, sort_keys = True))
    

if __name__ == '__main__':
    main()

