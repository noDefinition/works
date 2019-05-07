from utils import iu, mu, au, tmu
from typing import List
from . import tweet_keys as tk


# from utils.node_utils import Nodes


class MyUser:
    lut = dict()
    user_pkl_base = './twitter/pkls'
    user_json_base = './twitter/users'
    user_label_base = './twitter/labels'
    user_pkl_files = iu.list_children(user_pkl_base, pattern='.pkl', full_path=True)
    user_json_files = iu.list_children(user_json_base, pattern='.txt', full_path=True)
    label_files = iu.list_children(user_label_base, pattern='.txt', full_path=True)

    def __init__(self, profile, twarr):
        self.twarr: List[dict] = twarr
        self.profile: dict = profile
        self.uid: str = profile[tk.id_str]

    def sort_twarr_by_time(self):
        self.twarr = sorted(
            self.twarr, key=lambda tw: tmu.timestamp_of_created_at(tw[tk.created_at]))

    def get_created_at_list(self):
        return [tw[tk.created_at] for tw in self.twarr]

    def reindex_twarr(self, index_arr: List):
        assert len(index_arr) == len(self.twarr)
        assert set(index_arr) == set(range(len(index_arr)))
        self.twarr = [self.twarr[i] for i in index_arr]


class Helper:
    @staticmethod
    def sort_users_twarr_by_time_multi(usrarr: List[MyUser], p_num):
        created_at_lists: List[List[str]] = [user.get_created_at_list() for user in usrarr]
        created_at_lists_parts: List[List[List[str]]] = \
            au.split_multi_process(created_at_lists, p_num)
        args_list = [(p,) for p in created_at_lists_parts]
        arg_sort_lists = mu.multi_process(Helper.argsort_created_at_lists, args_list)
        arg_sort_list = au.merge(arg_sort_lists)
        for user, arg_sort in zip(usrarr, arg_sort_list):
            print(user.uid, end=' ', flush=True)
            user.reindex_twarr(arg_sort)

    @staticmethod
    def argsort_created_at_lists(created_at_lists: List[List[str]]) -> List[List[int]]:
        return [Helper.argsort_created_at_list(c) for c in created_at_lists]

    @staticmethod
    def argsort_created_at_list(created_at_list: List[str]) -> List[int]:
        import numpy as np
        timestamp_list: List[float] = [tmu.timestamp_of_created_at(s) for s in created_at_list]
        arg_sort = list(np.argsort(timestamp_list))  # order=ascent
        return arg_sort

    @staticmethod
    def load_user_from_file(file: str) -> MyUser:
        if file.endswith('txt') or file.endswith('json'):
            load_func = iu.load_array
        elif file.endswith('pkl'):
            load_func = iu.load_pickle
        else:
            raise ValueError('unidentified postfix of ', file)
        objarr = load_func(file)
        profile, twarr = objarr[0], objarr[1:]
        user = MyUser(profile, twarr)
        return user

    @staticmethod
    def load_users_from_files(files: List[str]) -> List[MyUser]:
        return [Helper.load_user_from_file(file) for file in files]
