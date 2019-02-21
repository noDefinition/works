import os
import pickle
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

Home = '/home/wwang/Projects/QAR_data'
token_pattern = '[\u4E00-\u9FA50-9a-zA-Z]\S*'
q_maxlen = 20
a_maxlen = 200

run_tfidf = 1
run_wids = 0
run_wordVec = 1
run_qf = 1


def main():
    print('hello world, features.py')
    ds = 'test'
    if len(sys.argv) > 1:
        ds = sys.argv[-1]
    home = '{}/{}_data'.format(Home, ds)
    to_dir = 'features'
    os.system('mkdir {}/{}'.format(home, to_dir))
    
    qid2text = pickle.load(open('{}/raw/qid2text_dict.pkl'.format(home), 'rb'))
    aid2text = pickle.load(open('{}/raw/aid2text_dict.pkl'.format(home), 'rb'))
    # uid2int = pickle.load(open('{}/raw/uid2int_dict.pkl'.format(home), 'rb'))
    numbers = pickle.load(open('{}/raw/numbers_dict.pkl'.format(home), 'rb'))
    word_dict = pickle.load(open('{}/raw/word_dict.pkl'.format(home), 'rb'))
    
    print('load ok')
    
    if run_tfidf:
        ids = list(qid2text.keys()) + list(aid2text.keys())
        texts = list(qid2text.values()) + list(aid2text.values())
        nb_q = len(qid2text)
        nb_a = len(aid2text)
        
        zero_time = time.time()
        tfidf = TfidfVectorizer(token_pattern=token_pattern, vocabulary=word_dict)
        tfidf_text = tfidf.fit_transform(texts)
        print('nb_words', len(tfidf.vocabulary_))
        print('fit_transform over, time: {:.2f}s'.format(time.time() - zero_time))
        idf = list(map(float, tfidf.idf_))
        pickle.dump(idf, open('{}/{}/idf_list.pkl'.format(home, to_dir), 'wb'), protocol=4)
        
        q_ids = ids[:nb_q]
        qid2int = dict(zip(q_ids, range(nb_q)))
        q_tfidf = tfidf_text[:nb_q]
        pickle.dump((qid2int, q_tfidf),
                    open('{}/{}/qid2int_tfidf_tuple.pkl'.format(home, to_dir), 'wb'), protocol=4)
        
        a_ids = ids[nb_q:]
        aid2int = dict(zip(a_ids, range(nb_a)))
        a_tfidf = tfidf_text[nb_q:]
        pickle.dump((aid2int, a_tfidf),
                    open('{}/{}/aid2int_tfidf_tuple.pkl'.format(home, to_dir), 'wb'), protocol=4)
        print('tfidf ok')
    
    if run_wids:
        def text_to_wid(text, maxlen):
            ret = [word_dict[w] + 1 for w in text.strip().split() if w in word_dict]
            if len(ret) >= maxlen:
                return ret[:maxlen]
            return ret + [0] * (maxlen - len(ret))
        
        qid2wids = {}
        for qid, text in qid2text.items():
            qid2wids[qid] = text_to_wid(text, q_maxlen)
        pickle.dump(qid2wids, open('{}/{}/qid2wids_dict.pkl'.format(home, to_dir), 'wb'),
                    protocol=4)
        
        aid2wids = {}
        for aid, text in aid2text.items():
            aid2wids[aid] = text_to_wid(text, a_maxlen)
        pickle.dump(aid2wids, open('{}/{}/aid2wids_dict.pkl'.format(home, to_dir), 'wb'),
                    protocol=4)
        
        print('wids ok')
    
    if run_wordVec:
        from gensim.models import Word2Vec
        size = 64
        nb_iter = 100
        texts = list(qid2text.values()) + list(aid2text.values())
        texts = [[w for w in text.strip().split() if w in word_dict] for text in texts]
        """
        wset = {}
        for text in texts:
            for w in text:
                if w not in wset:
                    wset[w] = 0
                wset[w] += 1
        """
        # print(texts)
        model = Word2Vec(texts, size=size, iter=nb_iter, min_count=0, workers=10)
        model.save('{}/{}/word2vec_{}d.model'.format(home, to_dir, size))
        nb_words = len(word_dict)
        wordVec = [0] * nb_words
        cnt = 0
        for w, wid in word_dict.items():
            if w in model.wv:
                wordVec[wid] = model.wv[w]
            else:
                # print(w)
                # print(wset[w])
                cnt += 1
        print('error cnt', cnt)
        wordVec = np.array(wordVec)
        print(wordVec.shape)
        pickle.dump(wordVec, open('{}/{}/wordVec_{}d.pkl'.format(home, to_dir, size), 'wb'),
                    protocol=4)


if __name__ == '__main__':
    main()
