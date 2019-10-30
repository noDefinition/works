from typing import List
import numpy as np
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer

from utils import au, iu, mu


class Document:
    def __init__(self):
        self.attrs = ['tag', 'uint', 'title', 'body']
        self.tag: str = ''
        self.uint: int = 0
        self.title: List[int] = list()
        self.body: List[List[int]] = list()

        self.tseq: List[int] = list()
        self.bseq: List[int] = list()
        self.all_texts: List[List[int]] = list()
        self.tf = self.tfidf = None

    def from_dict(self, obj: dict):
        for k in self.attrs:
            setattr(self, k, obj[k])
        return self

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.attrs}

    def flatten_body(self) -> List[int]:
        from itertools import chain
        return chain.from_iterable(self.body)
        # return (wint for wints in self.body for wint in wints)

    def fit_tf(self, word2wint: dict, add_body: bool):
        v_size = len(word2wint) + min(word2wint.values())
        tf = np.zeros(v_size, dtype=np.int32)
        for wid in self.title:
            tf[wid] += 1
        if add_body:
            body_gen = (wint for wints in self.body for wint in wints)
            for wid in body_gen:
                tf[wid] += 1
        self.tf = csr_matrix(tf)

    def split_texts(self, max_text_len: int):
        all_texts = [self.title]
        bwids = list(self.flatten_body())
        n = len(bwids)
        if n <= 20:
            bs = n
        elif n <= 40:
            bs = int(np.ceil(n / 2))
        elif n <= 120:
            bs = int(np.ceil(n / 3))
        else:
            bs = max_text_len
        wint_split = au.split_slices(bwids, batch_size=bs)
        all_texts.extend(wint_split)
        self.all_texts = all_texts


def docarr_fit_tf(docarr: List[Document], word2wint: dict, add_body: bool):
    for doc in docarr:
        doc.fit_tf(word2wint, add_body)
    return [doc.tf for doc in docarr]


def docarr_fit_tf_multi(docarr: List[Document], word2wint: dict, add_body: bool, p_num: int):
    docarr_list = mu.split_multi(docarr, p_num)
    args = [(docarr, word2wint, add_body) for docarr in docarr_list]
    tfarr_list = mu.multi_process(docarr_fit_tf, args)
    tf_gen = (tf for tfarr in tfarr_list for tf in tfarr)
    for doc, tf in zip(docarr, tf_gen):
        doc.tf = tf
    print('fit tf done')


def docarr_fit_tfidf(docarr: List[Document]):
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    tf = vstack([d.tf for d in docarr])
    tfidf = transformer.fit_transform(tf)
    tfidf = tfidf * np.sqrt(tfidf.shape[1])
    tfidf = preprocessing.normalize(tfidf, norm='l2') * 10
    tfidf = tfidf.astype(np.float32)
    for d, v in zip(docarr, tfidf):
        d.tfidf = v
    print('fit tfidf done')


def load_docarr(file: str):
    return [Document().from_dict(d) for d in iu.load_array(file)]


def dump_docarr(file: str, docarr: List[Document]):
    iu.dump_array(file, [doc.to_dict() for doc in docarr])
