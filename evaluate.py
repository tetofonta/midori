import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_diffs(df, group, weight):
    for k in range(df[group].max()):
        pesi = list(map(lambda x: float(x.replace(",", ".")), res.loc[res[group] == k][weight]))
        yield max(pesi) - min(pesi)


def get_size(df, group):
    for k in range(df[group].max()):
        yield res.loc[res[group] == k].shape[0]


if __name__ == "__main__":
    res = pd.read_csv('result.csv', sep=',')
    diffs = np.array(list(get_diffs(res, "Gruppo", "Peso")))
    unique, count = np.unique(list(get_size(res, "Gruppo")), return_counts=True)

    fig, ax = plt.subplots(2)
    ax[0].hist(diffs)
    ax[1].bar(unique, count)
    fig.show()
