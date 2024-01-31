import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("simulation_study.csv")[["dim","FI"]]
data = data.groupby("dim")

v = 0.01

for dim, group in data:
    true_lnZ = (dim / 2) * (np.log(v) - np.log(1 + v))
    plt.figure()
    plt.hist(group["FI"], label=dim, bins=30)
    plt.axvline(true_lnZ, color="red")
    plt.xlabel("LnZ")
    plt.ylabel("Density")
    plt.title(f"p={dim}, n-simulations={len(group)}")
    plt.tight_layout()
    plt.savefig(f"FIcase{dim}.png")


