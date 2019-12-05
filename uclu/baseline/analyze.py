import pandas as pd

from utils import iu

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


def get_log_path():
    log_files = iu.list_children('./', iu.FILE, r'\.json$', True)
    # old_paths = iu.list_children('./logs', iu.DIR, r'^(log)?\d+', True)
    # log_paths = new_paths + old_paths
    log_path = iu.choose_from(log_files)
    print('logging path:', log_path)
    return log_path


# def get_df(file: str):
#     arr = iu.load_array(file)
#     sp = iu.get_name(file).split('_')
#     vs, dn, addb = sp[0], sp[1], {'no': 0, 'add': 1}[sp[2]]
#     s_list = []
#     for scores in arr:
#         s = pd.Series(scores)
#         s_list.append(s)
#     df = pd.concat(s_list, axis=1).T
#     df['vs'] = vs
#     df['dn'] = dn
#     df['addb'] = addb
#     return df


def main():
    # files = iu.list_children('./', pattern=r'^kmeans.+json$', full_path=True)
    # for file in files:
    file = get_log_path()
    print(file)
    arr = iu.load_array(file)
    s_list = []
    for scores in arr:
        s = pd.Series(scores)
        s_list.append(s)
    df = pd.concat(s_list, axis=1).T
    df.to_csv(f'{iu.get_name(file)}.csv')
    print(df)


if __name__ == '__main__':
    main()
