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

    perch_90 = np.percentile(diffs, 90)
    differences_90 = np.array(list(filter(lambda v: v <= perch_90, diffs)))
    std_dev = np.std(diffs)
    mean = np.mean(diffs)
    std_dev_90 = np.std(differences_90)
    mean_90 = np.mean(differences_90)

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(diffs)
    ax[0].axvline(mean, color="red")
    ax[0].axvline(mean+std_dev, color="green")
    ax[0].axvline(perch_90, color="orange")
    ax[0].set_title("Differenze di peso")

    ax[1].bar(unique, count)
    ax[1].set_title("Numero di atleti per gruppo")
    fig.show()
