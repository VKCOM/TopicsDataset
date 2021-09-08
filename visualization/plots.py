import math
import matplotlib.pyplot as plt


CC = 0.95


def plot_conf_int(stat, init_size, n_instances, n_queries, stat_name, ax=None, **plt_kwargs):
    if len(stat) == 0:
        return
    q = n_queries + 1
    n = len(stat)
    mean = [0 for i in range(q)]
    sum_1 = [0 for i in range(q)]
    sum_2 = [0 for i in range(q)]

    for i in range(len(stat)):
        accurs = stat[i]
        for j in range(q):
            mean[j] += accurs[j] / n
            sum_1[j] += accurs[j] ** 2 / n
            sum_2[j] += accurs[j] / n

    D = [sum_1[i] - sum_2[i] ** 2 for i in range(q)]
    sigma = [CC * math.sqrt(d) / math.sqrt(n) for d in D]

    (plt if ax is None else ax).fill_between(
        range(init_size, init_size + n_queries * n_instances + 1, n_instances),
        [m + s for (m, s) in zip(mean, sigma)],
        [m - s for (m, s) in zip(mean, sigma)],
        label=stat_name,
        alpha=0.7,
        **plt_kwargs
    )
