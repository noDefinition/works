import pandas as pd

from uclu.data.datasets import *
from utils import au, iu, mu

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


def get_df(file: str):
    arr = iu.load_array(file)
    sp = iu.get_name(file).split('_')
    vs, dn, addb = sp[0], sp[1], {'no': 0, 'add': 1}[sp[2]]
    s_list = []
    for scores in arr:
        s = pd.Series(scores)
        s_list.append(s)
    df = pd.concat(s_list, axis=1).T
    df['vs'] = vs
    df['dn'] = dn
    df['addb'] = addb
    return df

def main2():
    df_list = []
    sf_files = iu.list_children(f'./{DataSf.name}', full_path=True)
    au_files = iu.list_children(f'./{DataAu.name}', full_path=True)
    for file in sf_files + au_files:
        df_list.append(get_df(file))
    df = pd.concat(df_list, axis=0)
    df.to_csv('au_sf.csv')
    print(df)

def main():
    files = iu.list_children('./', pattern=r'^kmeans.+json$', full_path=True)
    for file in files:
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
