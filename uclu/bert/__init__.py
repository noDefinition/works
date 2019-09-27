from utils.tune.arg_keys import Args


class UcluBertArgs(Args):
    def __init__(self):
        super(UcluBertArgs, self).__init__()
        # self.lr = self.add_arg('lr', type=float, help='learning rate')
        # self.bs = self.add_arg('bs', type=int, help='pos batch size')
        self.cn = self.add_arg('cn', type=int, help='cluster num')
        self.mxl = self.add_arg('mxl', type=int, help='max sequence len of inputs')

        self.ly = self.add_arg('ly', type=int, help='number of layers')
        self.nh = self.add_arg('nh', type=int, help='number of heads')
        self.dh = self.add_arg('dh', type=int, help='dim hidden')


UB: UcluBertArgs = UcluBertArgs()
