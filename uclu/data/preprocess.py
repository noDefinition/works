import re
from utils import iu, au, mu, tmu
from uclu.data import tweet_keys as tk


def rename_ground_truth():
    in_path = '/home/cdong/works/uclu/data/userClustering_origin/groud-truth-clusters'
    out_path = '/home/cdong/works/uclu/data/twitter/labels'
    files = iu.list_children(in_path, ctype=iu.FILE, full_path=True, pattern=r'^\d')
    for file in files:
        fname = iu.get_name(file)
        fname_new = re.sub(r'\b(\d)\b', '0\\1', fname)
        s = fname_new.split('-', maxsplit=3)
        s.insert(0, s.pop(2))
        fname_new = '-'.join(s)
        print(fname, '=>', fname_new, '\n')

        formated = reformat_ground_truth(file)
        iu.dump_array(iu.join(out_path, fname_new), formated)


def reformat_ground_truth(file):
    with open(file, mode='r') as fp:
        formated = list()
        for line in fp:
            splits = re.findall(r'\d+', line)
            cluid, usrids = splits[0], splits[1:]
            formated.append((cluid, usrids))
        formated = sorted(formated, key=lambda x: int(x[0]))
    return formated


def filter_tw_from_file(file):
    desire_tw_keys = ['created_at', 'id_str', 'retweet_count', 'text', ]
    desire_user_keys = [
        'followers_count', 'friends_count', 'statuses_count',
        'time_zone', 'verified', 'id_str', 'description', 'name',
    ]
    desire_ent_keys = ['symbols', 'hashtags', ]
    twarr = iu.load_array(file)
    new_twarr = list()
    for tidx, tw in enumerate(twarr):
        new_tw = {k: tw[k] for k in desire_tw_keys}
        new_tw['entities'] = {k: tw['entities'][k] for k in desire_ent_keys}
        new_twarr.append(new_tw)
    profile = {k: twarr[-1]['user'][k] for k in desire_user_keys}
    return profile, new_twarr


def filter_tw_from_files(files, out_path_json, out_path_pkl):
    for fidx, file in enumerate(files):
        if fidx > 0 and fidx % 10 == 0:
            print(fidx, end=' ', flush=True)
        profile, twarr = filter_tw_from_file(file)
        twarr = sorted(twarr, key=lambda tw: tmu.timestamp_of_created_at(tw[tk.created_at]))
        twarr.insert(0, profile)

        fname = iu.get_name(file)
        fjson = fname[fname.rfind('_') + 1:]
        fpkl = fjson.replace('txt', 'pkl')
        iu.dump_pickle(iu.join(out_path_pkl, fpkl), twarr)
        iu.dump_array(iu.join(out_path_json, fjson), twarr)


def extract_tweets():
    in_path = '/home/cdong/works/uclu/data/userClustering_origin/data'
    out_path_json = '/home/cdong/works/uclu/data/twitter/users/'
    out_path_pkl = '/home/cdong/works/uclu/data/twitter/pkls/'
    files = iu.list_children(in_path, ctype=iu.FILE, pattern='^E', full_path=True)
    print('total files', len(files))
    files_parts = au.split_multi_process(files, 20)
    args_list = [(part, out_path_json, out_path_pkl) for part in files_parts]
    res_list = mu.multi_process(filter_tw_from_files, args_list)
    # key_set = set(au.merge(res_list))
    # print(sorted(key_set))


if __name__ == '__main__':
    mmm()
