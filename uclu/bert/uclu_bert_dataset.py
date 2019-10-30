from itertools import chain
from typing import List

import numpy as np
import torch
from numpy.random import randint, random
from torch.utils.data import Dataset

from uclu.data.datasets import Sampler
from uclu.data.document import Document


class UcluBertDataset(Dataset):
    PAD = 0
    UNK = 1
    CLS = 2
    EOS = 3
    MSK = 4

    def __init__(self, sampler: Sampler, max_text_len: int):
        # self.max_text_len = 50
        self.max_text_len = max_text_len
        self.max_sample_len = self.max_text_len * 2 + 3
        self.pos_sample_num = 16
        self.neg_sample_num = 16
        self.sampler = sampler
        self.bounds = [
            (self.sampler.vocab_min, self.sampler.vocab_size),
            (self.sampler.user_min, self.sampler.user_size),
        ]

    def __len__(self):
        return len(self.sampler.docarr)

    def __getitem__(self, index: int):
        max_index_val = self.sampler.vocab_size + self.sampler.user_size
        items = [list() for _ in range(4)]
        for values in chain(self.gen_pos_samples(index), self.gen_neg_samples(index)):
            for t, v in zip(items, values):
                if isinstance(v, List):
                    for x in v:
                        if not x < max_index_val:
                            print(v)
                            raise ValueError('SHIT WHY VALUE TOO LARGE')
                    len_v = len(v)
                    if not len_v <= self. max_sample_len:
                        raise ValueError('too long', len_v)
                    # padding manually
                    if len_v < self.max_sample_len:
                        v += [self.PAD] * (self.max_sample_len - len_v)
                t.append(v)
        ret = list(map(torch.tensor, items))
        return ret

    @staticmethod
    def collate_fn(args):
        return [torch.cat(tensors, dim=0) for tensors in zip(*args)]

    def gen_pos_samples(self, index: int):
        doc_pos = self.sampler.docarr[index]
        n_text = len(doc_pos.all_texts)
        pairs = randint(0, n_text, (min(self.pos_sample_num * 3, n_text * 6), 2))
        pairs.sort(axis=1)
        pairs = list({(a, b) for a, b in pairs if a != b})[:self.pos_sample_num]
        for i1, i2 in pairs:
            w1 = doc_pos.all_texts[i1]
            w2 = doc_pos.all_texts[i2]
            if len(w1) > self.max_text_len or len(w2) > self.max_text_len:
                raise ValueError('tooooo long text', len(w1), len(w2), w1, w2)
            wints = [self.CLS] + w1 + [self.EOS] + w2 + [self.EOS]
            wints, labels = self.mask_wints(wints)
            segments = self.get_segmnets(w1, w2)
            if not len(wints) == len(labels) == len(segments):
                raise ValueError('length inconsistent', len(wints), len(labels), len(segments))
            yield wints, labels, segments, 1

    def gen_neg_samples(self, index: int):
        doc_pos = self.sampler.docarr[index]
        for _ in range(self.neg_sample_num):
            doc_neg = self.sampler.docarr[other_index(index, len(self))]
            w1 = doc_pos.all_texts[randint(len(doc_pos.all_texts))]
            w2 = doc_neg.all_texts[randint(len(doc_neg.all_texts))]
            wints = [self.CLS] + w1 + [self.EOS] + w2 + [self.EOS]
            wints, labels = self.mask_wints(wints)
            segments = self.get_segmnets(w1, w2)
            if not len(wints) == len(labels) == len(segments):
                raise ValueError('length inconsistent', len(wints), len(labels), len(segments))
            yield wints, labels, segments, 0

    def get_segmnets(self, w1: List, w2: List):
        return [1] * (len(w1) + 2) + [2] * (len(w2) + 1)

    def mask_wints(self, wints: List[int]):
        labels = [self.PAD] * len(wints)
        skip_wint = {self.PAD, self.EOS, self.CLS}
        for i, wint in enumerate(wints):
            if wint in skip_wint:
                continue
            p = np.random.random()
            if p < 0.15:  # 15% do mask, 85% do nothing
                labels[i] = int(wint)
                p /= 0.15
                if p < 0.8:  # 80% randomly change wint to mask wint
                    wints[i] = self.MSK
                elif p < 0.9:  # 10% randomly change wint to random wint
                    lower, upper = self.bounds[int(random() > 0.95)]
                    wints[i] = randint(lower, upper)
                # 10% do nothing
        return wints, labels


def other_index(exclude_index: int, max_index: int, min_index: int = 0):
    while True:
        i = np.random.randint(min_index, max_index)
        if i != exclude_index:
            return i
