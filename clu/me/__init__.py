from utils.tune.arg_keys import Args


class CluArgs(Args):
    def __init__(self):
        super(CluArgs, self).__init__()

        self.cn = self.add_arg('cn', type=int, help='cluster num')
        self.ns = self.add_arg('ns', type=int, help='neg batch num')
        self.sc = self.add_arg('sc', type=float, help='scale for normal init')
        # self.lr = self.add_arg('lr', type=float, help='learning rate')
        self.pad = self.add_arg('pad', type=int, help='if pad sequences')

        self.ed = self.add_arg('ed', type=int, help='embed dim')
        self.hd = self.add_arg('hd', type=int, help='hidden dim')
        self.md = self.add_arg('md', type=int, help='model inner dim')

        self.l1 = self.add_arg('l1', type=float, help='lambda for negative samples')
        self.l2 = self.add_arg('l2', type=float, help='lambda for cluster similarity')
        self.l3 = self.add_arg('l3', type=float, help='lambda for point-wise gen2')
        self.l4 = self.add_arg('l4', type=float, help='lambda for regularization gen2')

        self.ptn = self.add_arg('ptn', type=int, help='epoch num for pre-training')
        self.worc = self.add_arg('worc', type=int, help='0:word noise, 1: embed noise')
        self.eps = self.add_arg('eps', type=float, help='epsilon for adv grad')
        self.tpk = self.add_arg('tpk', type=int, help='top k for word feature')

        self.drp = self.add_arg('drp', type=float, help='dropout keep prob')
        self.mgn = self.add_arg('mgn', type=float, help='margin in hinge gen2')
        self.ckl = self.add_arg('ckl', type=float, help='coefficient of kl divergence in VAE')
        self.psmp = self.add_arg('psmp', type=int, help='position of sampling')
        self.regmu = self.add_arg('regmu ', type=int, help='if regularize mu in kl divergence')
        self.lbsmt = self.add_arg('smt', type=float, help='smoothness of labels')

        self.bn = self.add_arg('bn', type=int, help='if use batch normalization')
        self.wtrn = self.add_arg('wtrn', type=int, help='if train word embedding')
        self.ctrn = self.add_arg('ctrn', type=int, help='if train cluster embedding')
        self.wini = self.add_arg(
            'wini', type=int, default=1, help='word embed 0:random, 1:pre-trained')
        self.cini = self.add_arg(
            'cini', type=int, default=1, help='cluster embed 0:random, 1:pre-trained')

        """ sbx """
        self.useb = self.add_arg('useb', type=int, help='if use bias in denses')
        self.creg = self.add_arg('creg', type=float, help='coefficient of regularization')

        """ notations """
        self.alpha = self.add_arg('alpha', type=float, help='hyper-param: alpha')
        self.gamma = self.add_arg('gamma', type=float, help='hyper-param: gamma')
        self.beta = self.add_arg('beta', type=float, help='hyper-param: beta')
        # self.span = self.add_arg('span', type=float, help='span of changing alpha')

        """ spectral """
        self.sigma = self.add_arg('sigma', type=float, help='sigma of kernel')
        self.cspec = self.add_arg('cspec', type=float, help='coeff of spectral loss')
        self.cdecd = self.add_arg('cdecd', type=float, help='coeff of decode loss')
        self.cencd = self.add_arg('cencd', type=float, help='coeff of encode loss')
        self.cpurt = self.add_arg('cpurt', type=float, help='coeff of purity loss')
        self.ptncmb = self.add_arg('ptncmb', type=int, help='pretrain combination')


C = CluArgs()
