from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from uclu.me.models import *
from uclu.data.datasets import Sampler
from uclu.data.document import Document
from utils import au, mu, tmu
from .main import UcluRunner, UcluArgs


def get_dint_einit(docarr: List[Document], w_embed):
    w_num, w_dim = w_embed.shape
    dint_einit = list()
    for doc in docarr:
        trep = sum((w_embed[wint] for wint in doc.all_wints), np.zeros(w_dim))
        einit = trep / len(doc.all_wints)
        dint_einit.append([doc.dint, einit])
    return dint_einit


@tmu.stat_time_elapse
def get_d_embed(docarr: List[Document], w_embed):
    docarr_parts = mu.split_multi(docarr, 6)
    args = [(p, w_embed) for p in docarr_parts]
    res_list = mu.multi_process(get_dint_einit, args)
    dint_einit = sum(res_list, list())
    d_embed = [None] * len(docarr)
    for dint, einit in dint_einit:
        d_embed[dint] = einit
    for e in d_embed:
        assert e is not None
    d_embed = np.array(d_embed)
    return d_embed


class D2vRunner(UcluRunner):
    def __init__(self, args: UcluArgs):
        super(D2vRunner, self).__init__(args)
        self.window_size = args.ws

    def load(self):
        self.sampler.load(self.d_embed, self.title_pad, self.body_pad, topic_ratio=1)
        for dint, doc in enumerate(self.sampler.docarr):
            doc.dint = dint
            doc.all_wints = doc.title + list(doc.flatten_body())
        d_embed = get_d_embed(self.sampler.docarr, self.sampler.w_embed)
        self.model.build(self.sampler.w_embed, self.sampler.c_embed, self.sampler.u_embed, d_embed)
        self.num_c, _ = self.sampler.c_embed.shape

        self.dataset = D2vDataset(self.sampler, window_size=self.window_size)
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=D2vDataset.collate_fn,
            num_workers=4,
        )

    def run(self):
        for e in range(self.num_epoch):
            self.one_epoch()
            if e % 5 == 0 and self.should_early_stop():
                exit()
                return

    def one_epoch(self):
        assert isinstance(self.model, Doc2vec)
        for bid, data in enumerate(self.data_loader):
            args = [d.to(self.device) for d in data]
            self.model.train_step(*args)

    def evaluate(self):
        from clu.baselines.kmeans import fit_kmeans
        d_embed = self.model.d_embed.weight.cpu().detach().numpy()
        y_pred = fit_kmeans(d_embed, self.num_c, max_iter=200, n_jobs=6).predict(d_embed)
        y_true = [None] * len(self.sampler.docarr)
        for doc in self.sampler.docarr:
            y_true[doc.dint] = doc.tag
        return au.scores(y_true, y_pred)


class D2vDataset(Dataset):
    def __init__(self, sampler: Sampler, window_size: int):
        self.sampler = sampler
        self.window_size = window_size // 2

    def __len__(self):
        return len(self.sampler.docarr)

    def __getitem__(self, index: int):
        doc = self.sampler.docarr[index]
        n = len(doc.all_wints)
        all_idx = list(range(n))
        size = np.clip(int(n * 0.3), a_min=4, a_max=40)
        smp_idx = np.random.choice(all_idx, size, replace=False)
        contexts = list()
        for i in smp_idx:
            j_range = range(i - self.window_size, i + self.window_size + 1)
            context = [doc.all_wints[j] for j in j_range if 0 <= j < n and j != i]
            contexts.append(torch.tensor(context))
        targets = [doc.all_wints[i] for i in smp_idx]
        smp_size = len(targets)
        return contexts, targets, [doc.uint] * smp_size, [doc.dint] * smp_size

    @staticmethod
    def collate_fn(args):
        contexts, targets, uints, dints = [sum(ls, list()) for ls in zip(*args)]
        contexts = pad_sequence(contexts, batch_first=True)
        targets = torch.tensor(targets)
        uints = torch.tensor(uints)
        dints = torch.tensor(dints)
        return contexts, targets, uints, dints
