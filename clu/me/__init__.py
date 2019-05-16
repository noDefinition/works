from utils.tune.arg_keys import Args


class CluArgs(Args):
    def __init__(self):
        super(CluArgs, self).__init__()

        self.cn = self.add_arg('cn', type=int, help='cluster num')
        self.sc = self.add_arg('sc', type=float, help='scale for normal init')
        self.md = self.add_arg('md', type=int, default=None, help='model inner dim')
        self.bs = self.add_arg('bs', type=int, help='pos batch size')
        self.ns = self.add_arg('ns', type=int, help='neg batch num')

        self.l1 = self.add_arg('l1', type=float, help='lambda for negative samples')
        self.l2 = self.add_arg('l2', type=float, help='lambda for cluster similarity')
        self.l3 = self.add_arg('l3', type=float, help='lambda for point-wise loss')
        self.l4 = self.add_arg('l4', type=float, help='lambda for regularization loss')

        self.mgn = self.add_arg('mgn', type=float, default=1., help='margin in hinge loss')
        self.bn = self.add_arg('bn', type=int, help='if use batch normalization')
        self.smt = self.add_arg('smt', type=float, help='smoothness of cross entropy')

        self.wini = self.add_arg('wini', type=int, help='word embed 0:random, 1:pre-trained')
        self.cini = self.add_arg('cini', type=int, help='cluster embed 0:random, 1:pre-trained')
        self.wtrn = self.add_arg('wtrn', type=int, help='if train word embedding')
        self.ctrn = self.add_arg('ctrn', type=int, help='if train cluster embedding')

        self.ptn = self.add_arg('ptn', type=int, help='epoch num for pre-training')
        self.worc = self.add_arg('worc', type=int, help='0:word noise, 1: embed noise')
        self.eps = self.add_arg('eps', type=float, help='epsilon for adv grad')
        self.tpk = self.add_arg('tpk', type=int, help='top k for word feature')

    # l1 = 'l1'
    # l2 = 'l2'
    # l3 = 'l3'
    # l4 = 'l4'
    #
    # cn = 'cn'
    # ns = 'ns'
    # bn = 'bn'
    # md = 'md'
    # sc = 'sc'
    #
    # trn = 'trn'
    # mgn = 'mgn'
    # nis = 'nis'
    # smt = 'smt'
    #
    # ptn = 'ptn'
    # stn = 'stn'
    # dtn = 'dtn'
    # gtn = 'gtn'
    #
    # eps = 'eps'
    # worc = 'worc'
    # tpk = 'topk'
    #
    # wini = 'wini'
    # cini = 'cini'
    # wtrn = 'wtrn'
    # ctrn = 'ctrn'


C = CluArgs()
