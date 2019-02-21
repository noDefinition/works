import numpy as np, sys, math, os
import tensorflow as tf
from . import utils
from . import dataset
import time
from . import log
from . import q_maxlen, a_maxlen
from . import BasicPair

class Que(BasicPair):
    tc = dict(BasicPair.tc)
    def get_question_rep(self, q):
        return self.text_embs(q, name = 'QueEmb')

    def rep(self, ans, user):
        q, q_mask = self.question_rep
        return self.mask_mean(q, q_mask)


def main():
    print('hello world, Concat.py')

if __name__ == '__main__':
    main()

