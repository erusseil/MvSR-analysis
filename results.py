import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import sympify, Symbol, exp, log
import matplotlib.colors

plt.rcParams["font.size"] = 20


def get_values(df):
    vals = df.losses.values
    return np.array(
        list(
            map(
                lambda x: eval(x.replace("inf", "np.inf").replace("nan", "np.inf")),
                vals,
            )
        )
    )


ds = []
ns = []
ss = []
mvsr = []
ls = []
nps = []
deltas = []
lens = []
q25s = []
q75s = []

datasets = ["polynomial0", "polynomial_partial", "friedman1", "friedman2"]
noises = ["perfect", "noisy_0.033", "noisy_0.066", "noisy_0.1"]
sizes = [f"max{i}" for i in [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]]
n2i = {
    "perfect": 0,
    "noisy_0.05": 0.05,
    "noisy_0.033": 0.033,
    "noisy_0.066": 0.066,
    "noisy_0.1": 0.1,
}

alphabet = "ABCDEFGHIJKLMNOPQRSTUV"
par2num = {k: (v + 1) for v, k in enumerate(alphabet)}
optpar = {
    "gaussian": 3,
    "polynomial0": 4,
    "polynomial_partial": 4,
    "friedman1": 5,
    "friedman2": 5,
    "f8": 4,
    "f7": 4,
    "f1": 4,
}


def countpars(expr):
    for a in alphabet[::-1]:
        if a in expr:
            return par2num[a]
    return 0


def treesize(expr):
    tot = 1
    for a in expr.args:
        tot += treesize(a)
    return tot


for d in datasets:
    print(d)
    for n in noises:
        print(n)
        for s in sizes:
            print(s)

            print("===========================================")
            diffparams = []
            for i in range(4):
                df = pd.read_csv(f"toy_results/{d}/{n}/{s}/example{i}_results.csv")
                losses = get_values(df)
                v = np.median(np.max(losses, axis=1))
                print(f"example {i}: ", v)
                ds.append(d)

                ns.append(n2i[n])
                ss.append(int(s.replace("max", "")))
                mvsr.append(f"ex{i+1}")
                ls.append(v)
                q25, q75 = np.percentile(np.max(losses, axis=1), [25, 75])
                q25s.append(q25)
                q75s.append(q75)
                ps = np.median(list(map(countpars, df.expression.values)))
                nps.append(ps)
                deltas.append(ps - optpar[d])
                trees = []
                for e in df.expression.values:
                    try:
                        trees.append(
                            sympify(
                                e,
                                locals={k: Symbol(k) for k in "ABCDEFGHIJKLMNOPQRSTUV"},
                            )
                        )
                    except:
                        print(e)
                ts = np.median(list(map(treesize, trees)))
                lens.append(ts)
                diffparams.append(
                    np.array(list(map(countpars, df.expression.values))) - optpar[d]
                )

            df = pd.read_csv(f"toy_results/{d}/{n}/{s}/MvSR_results.csv")

            losses = get_values(df)
            v = np.median(np.max(losses, axis=1))

            ps = np.median(list(map(countpars, df.expression.values)))
            print("multiview: ", v)
            print("===========================================")
            ds.append(d)
            ns.append(n2i[n])
            ss.append(int(s.replace("max", "")))
            mvsr.append("MvSR")
            ls.append(v)
            q25, q75 = np.percentile(np.max(losses, axis=1), [25, 75])
            q25s.append(q25)
            q75s.append(q75)
            nps.append(ps)
            deltas.append(ps - optpar[d])
            trees = []
            for e in df.expression.values:
                try:
                    trees.append(
                        sympify(
                            e, locals={k: Symbol(k) for k in "ABCDEFGHIJKLMNOPQRSTUV"}
                        )
                    )
                except:
                    print(e)
            ts = np.median(list(map(treesize, trees)))
            lens.append(ts)
            plt.figure()
            diffparams.append(
                np.array(list(map(countpars, df.expression.values))) - optpar[d]
            )
            plt.boxplot(diffparams, labels=["ex1", "ex2", "ex3", "ex4", "MvSR"])
            plt.savefig(f"boxplots/boxplot_{d}_{n}_{s}.pdf")
            plt.close()

df_n = pd.DataFrame(
    {
        "dataset": ds,
        "noise": ns,
        "maxsize": ss,
        "algorithm": mvsr,
        "loss": ls,
        "q25": q25s,
        "q75": q75s,
        "pars": nps,
        "delta": deltas,
        "size": lens,
    }
).fillna(np.inf)
df_n.to_csv("summary.csv", index=False)

for d in datasets:
    print(d)
    pv = df_n[(df_n.noise == 0) & (df_n.dataset == d)].pivot_table(
        values="loss", index="maxsize", columns="algorithm"
    )
    pv25 = df_n[(df_n.noise == 0) & (df_n.dataset == d)].pivot_table(
        values="q25", index="maxsize", columns="algorithm"
    )
    pv75 = df_n[(df_n.noise == 0) & (df_n.dataset == d)].pivot_table(
        values="q75", index="maxsize", columns="algorithm"
    )

    err = []
    for col in pv25:  # Iterate over bar groups (represented as columns)
        err.append([pv25[col].values, pv75[col].values])
    err = np.abs(err)

    pv.plot(
        xticks=[5, 10, 15, 20, 25], marker="o", xlim=(4, 26), markersize=10, alpha=0.4
    )
    plt.savefig(f"plot/plot_sizes_perfect_{d}.pdf")
    print(pv)
    print(pv + pv75)

for d in datasets:
    print(d)
    pv = df_n[(df_n["maxsize"] == 25) & (df_n.dataset == d)].pivot_table(
        values="loss", index="noise", columns="algorithm"
    )
    pv25 = df_n[(df_n["maxsize"] == 25) & (df_n.dataset == d)].pivot_table(
        values="q25", index="noise", columns="algorithm"
    )
    pv75 = df_n[(df_n["maxsize"] == 25) & (df_n.dataset == d)].pivot_table(
        values="q75", index="noise", columns="algorithm"
    )

    print(pv25)
    print(pv75)
    err = []
    for col in pv25:  # Iterate over bar groups (represented as columns)
        err.append([pv25[col].values, pv75[col].values])
    err = np.abs(err)

    pv.plot(
        xticks=[0, 0.05, 0.1],
        marker="o",
        xlim=(-0.01, 0.11),
        ylim=(-0.01, 8),
        markersize=10,
        alpha=0.4,
    )
    plt.savefig(f"plot/plot_noises_10_{d}.pdf")
    print(pv)
    print(pv + pv75)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["white", "#22114c"]
)  # D3D3D3

for d in datasets:
    nrows = 12
    ncols = 12

    for a in df_n.algorithm.unique():
        pv = (
            df_n[(df_n.algorithm == a) & (df_n.dataset == d)]
            .pivot_table(values="loss", index="noise", columns="maxsize")
            .replace([np.inf, -np.inf], 5)
        )
        plt.figure()
        sns.heatmap(pv, cmap=cmap, square=False, cbar=a == "MvSR", vmin=0, vmax=5)
        plt.xticks(rotation=45)
        plt.savefig(f"heat/heat_{d}_{a}.pdf", bbox_inches="tight")
        plt.close()
