import pandas as pd

from utils import iu, au, mu


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


def main():
    # file = f'./gsdmm_results_title.json'
    # file = f'./lda_results_no_body.json'
    # file = f'./kmeans_results_no_body.json'
    files = iu.list_children('./', pattern='atm.+json$', full_path=True)
    for file in files:
        print(file)
        arr = iu.load_array(file)
        s_list = []
        for scores in arr:
            s = pd.Series(scores)
            s_list.append(s)
        df = pd.concat(s_list, axis=1).T
        print(df)
        df.to_csv(f'{iu.get_name(file)}.csv')


if __name__ == '__main__':
    main()
