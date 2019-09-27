from typing import List

import numpy as np
import torch
from numpy.random import randint
from torch.utils.data import Dataset

from uclu.data.datasets import Sampler
from uclu.data.document import Document


class UcluBertDataset(Dataset):
    def __init__(self, sampler: Sampler, max_seq_len: int):
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.eos_index = 3
        self.msk_index = 4

        self.max_text_len = 50
        self.max_sample_len = self.max_text_len * 2 + 3
        self.pos_sample_size = 16
        self.neg_sample_size = 16

        self.sampler = sampler
        self.max_seq_len = max_seq_len
        for doc in self.sampler.docarr:
            doc.split_texts(self.max_text_len)
            doc.all_texts.append([doc.uint + self.sampler.vocab_size])

    def __len__(self):
        return len(self.sampler.docarr)

    def __getitem__(self, index: int):
        from itertools import chain
        # from torch.nn.utils.rnn import pad_sequence
        items = [list() for _ in range(4)]
        for values in chain(self.gen_pos_samples(index), self.gen_neg_samples(index)):
            for t, v in zip(items, values):
                if isinstance(v, List):
                    len_v = len(v)
                    if not len_v <= self. max_sample_len:
                        raise ValueError('too long', len_v)
                    if len_v < self.max_sample_len:
                        v += [self.pad_index] * (self.max_sample_len - len_v)
                # else:
                #     v = [v]
                t.append(v)
                # t.append(torch.tensor(v))
        # ret = []
        # for array in inputs:
        # array = pad_sequence(array, batch_first=True, padding_value=self.pad_index)
        # print(np.array(array).shape)
        # ret.append(array)
        ret = list(map(torch.tensor, items))
        # for r in ret: print(r.size())
        # return items
        return ret

        # t1, t2, is_next = self.get_pair(index)
        # l1, l2 = self.clip_length(len(t1), len(t2), self.max_seq_len - 3)
        # t1, t2 = t1[:l1], t2[:l2]
        # f1_wints, f1_label = self.random_word(t1)
        # f2_wints, f2_label = self.random_word(t2)
        # f1_wints = [self.cls_index] + f1_wints + [self.eos_index]
        # f1_label = [self.pad_index] + f1_label + [self.pad_index]
        # f2_wints = f2_wints + [self.eos_index]
        # f2_label = f2_label + [self.pad_index]
        # padding = [self.pad_index] * max(0, self.max_seq_len - len(f1_wints) - len(f2_wints))
        # wints_inputs = f1_wints + f2_wints + padding
        # labels_inputs = f1_label + f2_label + padding
        # segment_inputs = [1] * len(f1_wints) + [2] * len(f2_wints) + padding
        # if not len(wints_inputs) == len(labels_inputs) == len(segment_inputs) == self.max_seq_len:
        #     print()
        #     print(len(t1), len(t2), l1, l2)
        #     print(len(wints_inputs), len(labels_inputs), len(segment_inputs))
        #     raise ValueError('length inconsistent')
        # return list(map(torch.tensor, [wints_inputs, labels_inputs, segment_inputs, is_next]))

    @staticmethod
    def collate_fn(args):
        return [torch.cat(t_list, dim=0) for t_list in zip(*args)]

    def gen_pos_samples(self, index: int):
        doc_pos = self.sampler.docarr[index]
        n_text = len(doc_pos.all_texts)
        pairs = randint(0, n_text, (n_text * 6, 2))
        pairs = sorted({(min(a, b), max(a, b)) for a, b in pairs if a != b})
        print(n_text, pairs)
        for i1, i2 in pairs:
            w1 = doc_pos.all_texts[i1]
            w2 = doc_pos.all_texts[i2]
            if len(w1) > self.max_text_len or len(w2) > self.max_text_len:
                raise ValueError('tooooo long text', len(w1), len(w2), w1, w2)
            wints = [self.cls_index] + w1 + [self.eos_index] + w2 + [self.eos_index]
            wints, labels = self.mask_wints(wints)
            segments = self.get_segmnets(w1, w2)
            if not len(wints) == len(labels) == len(segments):
                raise ValueError('length inconsistent', len(wints), len(labels), len(segments))
            yield wints, labels, segments, 1

    def gen_neg_samples(self, index: int):
        doc_pos = self.sampler.docarr[index]
        for _ in range(self.neg_sample_size):
            doc_neg = self.sampler.docarr[other_index(index, len(self))]
            w1 = doc_pos.all_texts[randint(len(doc_pos.all_texts))]
            w2 = doc_neg.all_texts[randint(len(doc_neg.all_texts))]
            wints = [self.cls_index] + w1 + [self.eos_index] + w2 + [self.eos_index]
            wints, labels = self.mask_wints(wints)
            segments = self.get_segmnets(w1, w2)
            if not len(wints) == len(labels) == len(segments):
                raise ValueError('length inconsistent', len(wints), len(labels), len(segments))
            yield wints, labels, segments, 0

    def get_segmnets(self, w1: List, w2: List):
        return [1] * (len(w1) + 2) + [2] * (len(w2) + 1)

    def mask_wints(self, wints: List[int]):
        labels = [self.pad_index] * len(wints)
        skip_wint = {self.pad_index, self.eos_index, self.cls_index}
        for i, wint in enumerate(wints):
            if wint in skip_wint:
                continue
            p = np.random.random()
            if p < 0.15:  # 15% do mask, 85% do nothing
                labels[i] = wint
                p /= 0.15
                if p < 0.8:  # 80% randomly change wint to mask wint
                    wints[i] = self.msk_index
                elif p < 0.9:  # 10% randomly change wint to random wint
                    wints[i] = randint(self.sampler.vocab_min, self.sampler.vocab_size)
                # 10% do nothing
        return wints, labels

    # def get_inputs(self, input_wints, is_next: int):
    #     wints, labels, segment = list(), list(), list()
    #     for wints in input_wints:
    #         mask_wints, mask_labels = self.mask_wints(wints)
    #         segment = self.get_segmnet(wints)
    #         wints.append(mask_wints)
    #         labels.append(mask_labels)
    #         segment.append(segment)
    #     inputs = (wints, labels, segment, is_next)
    #     return list(map(torch.tensor, inputs))
    #
    # def get_segmnet(self, wints: List):
    #     len_part1 = wints.index(self.eos_index) + 1
    #     assert len_part1 < len(wints)
    #     segment = [1] * len_part1 + [2] * (len(wints) - len_part1)
    #     return segment


def other_index(exclude_index: int, max_index: int, min_index: int = 0):
    while True:
        i = np.random.randint(min_index, max_index)
        if i != exclude_index:
            return i


# def get_ap(length: int):
#     a = np.arange(length)
#     p = np.ones(length)
#     p[0] += 2
#     p = p / np.sum(p)
#     return a, p
#
#
# def with_prob(prob: float):
#     return np.random.random() < prob
