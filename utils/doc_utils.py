from collections import Counter, OrderedDict as Od

import numpy as np
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer

import utils.io_utils as iu
import utils.pattern_utils as pu
from utils.id_freq_dict import IdFreqDict
from typing import List


# noinspection PyAttributeOutsideInit
class Document:
    def __init__(self):
        self.clear()

    def clear(self):
        self.docid = self.topic = None
        self.text: str = ''
        self.tokens: List[str] = list()
        self.tokenids: List[int] = list()
        self.pos_fill = self.neg_fill = self.tf = self.tfidf = None
        'tfidf will be sparse from 190711, and getting tfidf matrix may need transform / vstack'

    def set(self, docid, topic, text):
        self.docid, self.topic, self.text = docid, topic, text
        return self

    def to_dict(self):
        attrs = ['docid', 'topic', 'text', 'tokens', 'tokenids']
        return Od([(k, getattr(self, k)) for k in attrs])

    def from_dict(self, d):
        self.clear()
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


def dump_docarr(file, docarr):
    iu.dump_array(file, [d.to_dict() for d in docarr])


def load_docarr(file) -> List[Document]:
    return [Document().from_dict(d) for d in iu.load_array(file)]


def make_docarr(args_list) -> List[Document]:
    return [Document().set(*arg) for arg in args_list]


def docarr_fit_tf(word2vec: dict, docarr):
    v_size = len(word2vec) + 1
    for doc in docarr:
        if doc.tf is None:
            tf = np.zeros(v_size, dtype=np.int32)
            for wid in doc.tokenids:
                tf[wid] += 1
            doc.tf = csr_matrix(tf)
    print('tf transform over')
    return docarr


def docarr_fit_tfidf(word2vec: dict, docarr: List[Document]):
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    docarr = docarr_fit_tf(word2vec, docarr)
    tf_list = [d.tf for d in docarr]
    tf = vstack(tf_list)
    tfidf = transformer.fit_transform(tf)
    tfidf = tfidf * np.sqrt(tfidf.shape[1])
    tfidf = preprocessing.normalize(tfidf, norm='l2') * 200
    tfidf = tfidf.astype(np.float32)
    for d, v in zip(docarr, tfidf):
        d.tfidf = v
    print('tfidf transform over')
    return docarr


def tokenize_docarr(docarr: List[Document], stemming: bool):
    stem_func = pu.stemming if stemming else None
    for doc in docarr:
        doc.tokens = pu.tokenize(doc.text.lower().strip(), pu.tokenize_pattern)
        if stem_func:
            doc.tokens = list(map(stem_func, doc.tokens))
    return docarr


def validate_tokens(tokens: List[str], w_verify_func):
    return [t for t in tokens if w_verify_func(t)]


def get_ifd_from_docarr(docarr: List[Document]):
    """ assume that docarr has been tokenized """
    ifd = IdFreqDict()
    for doc in docarr:
        ifd.count_words(doc.tokens)
    ifd.reset_id()
    return ifd


def validate_docarr_by_ifd(docarr: List[Document], ifd):
    for d in docarr:
        d.tokens = [w for w in d.tokens if w in ifd]
        d.tokenids = [ifd.word2id(w) for w in d.tokens]


def docarr_bootstrap_ifd(docarr, wf_flt_func):
    ifd = get_ifd_from_docarr(docarr)
    print('vocab:{}, freq sum:{}'.format(ifd.vocabulary_size(), ifd.get_freq_sum()))
    ifd.reserve_words(wf_flt_func)
    print('vocab:{}, freq sum:{}'.format(ifd.vocabulary_size(), ifd.get_freq_sum()))
    validate_docarr_by_ifd(docarr, ifd)
    return docarr, ifd


def filter_docarr(docarr: List[Document], doc_filter_func):
    if doc_filter_func is None:
        return docarr
    return [d for d in docarr if doc_filter_func(d)]
    # import utils.array_utils as au
    # groups = dict()
    # for d in docarr:
    #     groups.setdefault(d.topic, list()).append(d)
    # for topic, array in list(groups.items()):
    #     max_len, tkn_len = 2000, 5
    #     if len(array) >= max_len:
    #         groups[topic] = [a for a in array if len(a.tokens) >= tkn_len][:max_len]
    #         print(topic, len(array), '->', len(groups[topic]))
    # docarr = au.merge(groups.values())
    # return docarr


def filter_docarr_by_topic(docarr: List[Document], topic_filter_func):
    topics = [d.topic for d in docarr]
    accept_topics, reject_topics = Counter(), Counter()
    for rank, (topic, freq) in enumerate(Counter(topics).most_common()):
        if topic_filter_func(rank, freq):
            accept_topics[topic] = freq
        else:
            reject_topics[topic] = freq
    flt_docarr = [d for d in docarr if d.topic in accept_topics]
    return accept_topics, reject_topics, flt_docarr


def filter_duplicate_docid(docarr):
    idset = set()
    flt_arr = list()
    for d in docarr:
        docid = d.docid
        if docid not in idset:
            idset.add(docid)
            flt_arr.append(d)
    print('pre-filter id len:{}, post-filter id len:{}'.format(len(docarr), len(flt_arr)))
    return flt_arr


def word_verify(len_min, len_max, alpha_thres, stop_corpus):
    def is_valid_word(word):
        word = word.lower().strip()
        not_too_short = (len(word) >= len_min if len_min else True)
        not_too_long = (len(word) <= len_max if len_max else True)
        len_valid = not_too_short and not_too_long
        is_stop = word in stop_corpus if stop_corpus else False
        enough_alpha = pu.has_enough_alpha(word, alpha_thres) if alpha_thres else True
        return len_valid and enough_alpha and not is_stop

    return is_valid_word
