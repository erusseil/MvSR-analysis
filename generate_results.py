import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_values(df):
    vals = df.losses.values
    return np.array(list(map(lambda x: eval(x.replace("inf", "np.inf").replace("nan", "np.inf")), vals)))

ds = []
ns = []
ss = []
mvsr = []
ls = []
nps = []
deltas = []
q25s = []
q75s = []

datasets = ['polynomial0', 'polynomial_partial', "friedman2",  "friedman1"]
noises = ["perfect", "noisy_0.033", "noisy_0.066", "noisy_0.1"]
sizes = [f"max{i}" for i in np.arange(5, 27, 2)]
n2i = {'perfect':0, 'noisy_0.033':0.033, 'noisy_0.066':0.066,'noisy_0.1':0.1}

alphabet = "ABCDEFGHIJKLMNOPQRSTUV"
par2num = { k:(v+1) for v,k in enumerate(alphabet) }
optpar = {"friedman1" : 4, "friedman2" : 4, "polynomial0" : 4, "polynomial_partial" : 4}

def countpars(expr):
    for a in alphabet[::-1]:
        if a in expr:
            return par2num[a]
    return 0

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
                ss.append(int(s.replace("max","")))
                mvsr.append(f"ex{i+1}")
                ls.append(v)
                q25, q75 = np.percentile(np.max(losses, axis=1), [25,75])
                q25s.append(q25)
                q75s.append(q75)
                ps = np.median(list(map(countpars, df.expression.values)))
                nps.append(ps)
                deltas.append(ps - optpar[d])
                diffparams.append(np.array(list(map(countpars, df.expression.values)))-optpar[d])

            df = pd.read_csv(f"toy_results/{d}/{n}/{s}/MvSR_results.csv")

            losses = get_values(df)
            v = np.median(np.max(losses, axis=1))

            ps = np.median(list(map(countpars, df.expression.values)))
            print("multiview: ", v)
            print("===========================================")
            ds.append(d)
            ns.append(n2i[n])
            ss.append(int(s.replace("max","")))
            mvsr.append("MvSR")
            ls.append(v)
            q25, q75 = np.percentile(np.max(losses, axis=1), [25,75])
            q25s.append(q25)
            q75s.append(q75)
            nps.append(ps)
            deltas.append(ps - optpar[d])

            plt.figure()
            diffparams.append(np.array(list(map(countpars, df.expression.values)))-optpar[d])
            plt.boxplot(diffparams, labels=["ex1", "ex2", "ex3", "ex4", "MvSR"])
            plt.savefig(f"result_plots/boxplots/boxplot_{d}_{n}_{s}.png")
            plt.close()

df_n = pd.DataFrame({"dataset":ds, "noise":ns, "size":ss, "algorithm":mvsr, "loss":ls, "q25":q25s, "q75":q75s, "pars":nps, "delta":deltas})
df_n.to_csv("summary.csv", index=False)

for d in datasets:
    pv = df_n[(df_n.noise == 0) & (df_n.dataset == d)].pivot_table(values="loss", index="size", columns="algorithm")
    pv25 = df_n[(df_n.noise == 0) & (df_n.dataset == d)].pivot_table(values="q25", index="size", columns="algorithm")
    pv75 = df_n[(df_n.noise == 0) & (df_n.dataset == d)].pivot_table(values="q75", index="size", columns="algorithm")
    
    err = []

    for col in pv25:  # Iterate over bar groups (represented as columns)
        both = []
        if col in pv25:
            both.append(pv25[col].values)
        else:
            both.append([np.nan]*len(sizes))
            
        if col in pv75:
            both.append(pv75[col].values)
        else:
            both.append([np.nan]*len(sizes))

        err.append(both)

    err = np.abs(err)

    plot_pv = pv.copy()
    plot_pv[plot_pv < 0.01] = 0.01
    plot_pv.plot(xticks=[5,10,15,20,25], marker="o", xlim=(4,26), markersize=10, alpha=0.4, logy=True) #
    plt.savefig(f"result_plots/plot_sizes_perfect_{d}.png")
    print(pv)
    print(pv+pv75)

for d in datasets:
    print(d)
    pv = df_n[(df_n["size"] == 15) & (df_n.dataset == d)].pivot_table(values="loss", index="noise", columns="algorithm")
    pv25 = df_n[(df_n["size"] == 15) & (df_n.dataset == d)].pivot_table(values="q25", index="noise", columns="algorithm")
    pv75 = df_n[(df_n["size"] == 15) & (df_n.dataset == d)].pivot_table(values="q75", index="noise", columns="algorithm")
    
    err = []
    for col in pv25:  # Iterate over bar groups (represented as columns)
        err.append([pv25[col].values, pv75[col].values])
    err = np.abs(err)

    plot_pv = pv.copy()
    plot_pv[plot_pv < 0.01] = 0.01
    plot_pv.plot(xticks=[0, 0.05, 0.1], marker='o', xlim=(-0.01, 0.11), markersize=10, alpha=0.4, logy=True)
    plt.savefig(f"result_plots/plot_noises_10_{d}.png")
    print(pv)
    print(pv+pv75)

