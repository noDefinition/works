from utils import au, iu, lu
from utils.tune.arg_keys import Args, X


class Runner:
    def __init__(self, args: Args):
        self.args = args
        self.batch_size: int = args.bs
        self.num_epoch: int = args.ep
        self.gpu_id: float = args.gi
        self.gpu_frac: float = args.gp

        args_dict = args.get_dict()
        self.logger = None
        self.log_path: str = args.lg
        self.log_name: str = au.entries2name(args_dict, exclude={X.gi, X.gp, X.lg})
        if self.log_path and self.log_name:
            log_file = iu.join(self.log_path, self.log_name + '.txt')
            self.logger = lu.get_logger(log_file)

        self.ppp(args_dict, json=True)
        self.epoch: int = 0
        self.history: list = list()

    def ppp(self, info, json=False):
        if json:
            info = iu.dumps(info)
        print(info)
        if self.logger:
            self.logger.info(info)

    def load(self):
        pass

    def build(self):
        pass

    def run(self):
        pass

    def main(self):
        self.load()
        self.build()
        self.run()
