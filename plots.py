import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from critdd import Diagram

df = pd.read_csv("summary.csv")

for dataset in ["polynomial0", "polynomial_partial", "friedman1", "friedman2"]:
    plt.figure()
    sns.lineplot(df[df.dataset == dataset], x="maxsize", y="size", hue="algorithm")
    plt.savefig(f"maxsize_size_{dataset}.png")

plt.figure()
df["absdelta"] = np.abs(df.delta)
sns.boxplot(df, x="algorithm", y="absdelta")
plt.savefig(f"params_hist.png")

# create a CD diagram from the Pandas DataFrame
df_piv = df.pivot(
    index=["dataset", "noise", "maxsize"], columns="algorithm", values="absdelta"
)
diagram = Diagram(
    df_piv.to_numpy(), treatment_names=df_piv.columns, maximize_outcome=False
)

# inspect average ranks and groups of statistically indistinguishable treatments
diagram.average_ranks  # the average rank of each treatment
diagram.get_groups(alpha=0.05, adjustment="holm")

# export the diagram to a file
diagram.to_file(
    "example.pdf",
    alpha=0.05,
    adjustment="holm",
    reverse_x=False,
)
