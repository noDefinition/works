from utils.tune.arg_keys import Args


class UcluBertArgs(Args):
    def __init__(self):
        super(UcluBertArgs, self).__init__()
        self.cn = self.add_arg('cn', type=int, help='cluster num')
        self.mxl = self.add_arg('mxl', type=int, help='max sequence len of inputs')
        self.ptn = self.add_arg('ptn', type=int, help='epochs of pre-training')

        self.tpad = self.add_arg('tpad', type=int, help='padding of title')
        self.bpad = self.add_arg('bpad', type=int, help='padding of body')

        self.ly = self.add_arg('ly', type=int, help='number of layers')
        self.dh = self.add_arg('dh', type=int, help='dim hidden')
        self.nh = self.add_arg('nh', type=int, help='number of heads')
        self.do = self.add_arg('do', type=float, help='dropout rate')


UB: UcluBertArgs = UcluBertArgs()
