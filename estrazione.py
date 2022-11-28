import itertools
import sys
import numpy as np
import pandas as pd
import pulp


def _normalize_name(name):
    return str(name).lower().capitalize().strip()


def _to_int(value):
    if type(value) == str:
        return float(value.replace(",", ".").strip())
    return value


normalize_name = np.vectorize(_normalize_name)
to_int = np.vectorize(_to_int)


def diff_group(group):
    tmp = list(map(lambda g: g[1], group))
    return max(tmp) - min(tmp)


def get_sets(df, size):
    for i in range(0, df.shape[0] - size + 1):
        group = []
        for a in df[i:i + size].itertuples():
            group.append((a.Idx, a.Peso))
        yield group, diff_group(group)


def ids(data):
    used = []
    for s in data:
        for m in s[0]:
            if m[0] not in used:
                yield m[0]
                used += [m[0]]


def solve(pools):
    obj = pulp.LpVariable('max_difference', lowBound=0, cat=pulp.LpContinuous)
    model = pulp.LpProblem("Pool_Model", pulp.LpMinimize)
    x = pulp.LpVariable.matrix('group', pools, lowBound=0, upBound=1, cat=pulp.LpInteger)
    model += obj  # Objective function

    for i, group in enumerate(pools):
        model += group[1] * x[i] <= obj, f"minimax_group_{i}"

    for index in ids(pools):
        model += sum([x[i] for i, g in enumerate(pools) if
                      index in map(lambda gg: gg[0], g[0])]) == 1, f"Must_seat_{index}"

    model.solve()

    #Second solution
    model += sum([abs(group[1]) * x[i] for i, group in enumerate(pools)])  # Objective function
    for i, group in enumerate(pools):
        model += group[1] * x[i] <= obj.value(), f"minimax_2_group_{i}"
    model.solve()

    return map(lambda v: v[1], filter(lambda v: x[v[0]].value() > 0.5, enumerate(pools)))


def prepare(df, lb):
    df.sort_values(["Peso"], axis=0, ascending=[True], inplace=True)
    available_pools = itertools.chain.from_iterable(map(lambda s: get_sets(df, s), range(lb, 6)))
    return available_pools


if __name__ == "__main__":
    athletes = pd.read_csv('athletes.csv', sep=',')
    athletes["Peso"] = to_int(athletes["Peso"])
    athletes["Idx"] = range(athletes.shape[0])

    redo_athletes = athletes
    working_list = prepare(athletes, 4)
    consolidated = []

    MAX = 4
    for i in range(0, MAX - 1):

        working_list = prepare(redo_athletes, MAX - i)

        partial_sol = list(solve(list(working_list)))
        sol = consolidated + partial_sol

        differences = np.array(list(map(lambda s: s[1], sol)))
        perch_90 = np.percentile(differences, 90)
        differences_90 = np.array(list(filter(lambda v: v <= perch_90, map(lambda s: s[1], sol))))

        std_dev = np.std(differences)
        mean = np.mean(differences)
        std_dev_90 = np.std(differences_90)
        mean_90 = np.mean(differences_90)

        redo = list(filter(lambda g: g[1] >= mean + std_dev_90, sol))
        redo_athletes = pd.DataFrame(itertools.chain.from_iterable(map(lambda x: x[0], redo)), columns=['Idx', 'Peso'])

        print(f"Iteration {i+1}")
        print("Weight Difference Average:...........{:.4f}".format(mean))
        print("Weight Difference Deviation:.........{:.4f}".format(std_dev))
        print("Weight Difference Maximum:...........{:.4f}".format(np.amax(differences)))
        print("Weight Difference Average (90th%):...{:.4f}".format(mean_90))
        print("Weight Difference Deviation (90th%):.{:.4f}".format(std_dev_90))
        print("Weight Difference Maximum (90th%):...{:.4f}".format(np.amax(differences_90)))
        print(f"Exceeding: {len(redo)} groups / {redo_athletes.shape[0]} athletes")

        consolidated = list(filter(lambda g: g[1] < mean + std_dev_90, sol))

        if len(redo) == 0 or i == MAX-2:
            consolidated += redo
            break

    total = sum(1 for _ in ids(consolidated))
    print(f"Assigned {total} athletes out of {athletes.shape[0]}")

    result = pd.DataFrame([], columns=list(athletes.columns)[:-1] + ["Gruppo"])
    for idx, group in enumerate(consolidated):
        for member in group[0]:
            result.loc[member[0]] = list(athletes.loc[member[0]].values)[:-1] + [str(idx)]

    result.sort_values(["Peso"], axis=0, ascending=[True], inplace=True)
    result.to_csv("./result.csv", float_format=lambda x: "{:.2f}".format(x).replace(".", ","), index=False)
