import pandas as pd

from utils import iu, au, mu


def main():
    baseline = 'lda'
    arr = iu.load_array(f'./{baseline}_results.json')
    s_list = []
    for scores in arr:
        s = pd.Series(scores)
        s_list.append(s)
    df = pd.concat(s_list, axis=1).T
    print(df)
    df.to_csv(f'{baseline}.csv')


if __name__ == '__main__':
    main()
