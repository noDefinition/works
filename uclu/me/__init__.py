from utils.tune.arg_keys import Args


class UcluArgs(Args):
    def __init__(self):
        super(UcluArgs, self).__init__()

        self.cn = self.add_arg('cn', type=int, help='cluster num')
        self.tpad = self.add_arg('tpad', type=int, help='padding of title')
        self.bpad = self.add_arg('bpad', type=int, help='padding of body')

        self.ed = self.add_arg('ed', type=int, help='embed dim')
        self.hd = self.add_arg('hd', type=int, help='hidden dim')
        # self.md = self.add_arg('md', type=int, help='model inner dim')
        # self.ns = self.add_arg('ns', type=int, help='neg batch num')

        self.pair = self.add_arg('pair', type=int, help='how to pairwise')
        self.addu = self.add_arg('addu', type=float, help='if add user info')
        self.addb = self.add_arg('addb', type=float, help='if add body')
        self.addpnt = self.add_arg('addpnt', type=float, help='add extra pointwise')

        self.ws = self.add_arg('ws', type=int, help='window size')


U = UcluArgs()
