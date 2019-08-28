import inspect
from typing import List, Dict
from utils import iu, tmu, tu


class BaseGrid(object):
    def __init__(self):
        self.comment: str = ''
        self.vers: List = list()
        self.datas: List = list()
        self.copy_modules: List = list()
        self.gpu_ids = self.gpu_max = None
        self.is_debug = self.is_debugging()

    def grid_od_list(self) -> List[Dict]:
        return list()

    @staticmethod
    def is_debugging() -> bool:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-d', '--debug', action='store_true', default=False)
        args = parser.parse_args()
        return args.debug

    @staticmethod
    def get_comment(default: str = None) -> str:
        import re
        if default:
            return default
        while True:
            comment = input('comment:')
            if re.findall(r'\W', comment):
                print('invalid comment {}, reinput'.format(comment))
                continue
            return comment

    def make_log_path(self) -> str:
        tstr = tmu.format_date()[2:]
        dstr = ''.join([d.name[0] for d in self.datas])
        vstr = '+'.join([v.__name__ for v in self.vers])
        log_path = './' + '-'.join([tstr, vstr, dstr, self.comment])
        if self.is_debug:
            ret_path = log_path + '_debug'
            iu.mkdir(ret_path, rm_prev=True)
            return ret_path
        else:
            for i in range(10):
                ret_path = log_path + f'_{i + 1}'
                if not iu.exists(ret_path):
                    iu.mkdir(ret_path, rm_prev=False)
                    return ret_path
            raise ValueError('You\'ve FUCKING assigned too many paths to one name, check it out.')

    def copy_files_to(self, log_path):
        iu.copy(__file__, log_path)
        for m in self.copy_modules:
            iu.copy(inspect.getfile(m), log_path)

    def main(self):
        log_path = self.make_log_path()
        self.copy_files_to(log_path)
        od_list = self.grid_od_list()
        od_list = tu.update_od_list(od_list, log_path, shuffle=False)
        tu.run_od_list('python3.6 ./main.py ', od_list, self.gpu_ids, self.gpu_max)
        if self.is_debug:
            return
        new_path = log_path + '=='
        iu.rename(log_path, new_path)
        out_path = './logs'
        iu.mkdir(out_path, rm_prev=False)
        iu.move(new_path, out_path)
