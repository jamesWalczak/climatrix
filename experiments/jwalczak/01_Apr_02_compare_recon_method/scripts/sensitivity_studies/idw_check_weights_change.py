from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRICS_PATH = Path(__file__).parent / "results" / "weight_change"
METRICS_PATH.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
power_median: float = 2.87

p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 1.0])

euclid = []
w_euclid = []
w_manhattan = []
w_czebyszev = []
for d in np.linspace(0.1, 0.2, num=10000):
    p2 += np.array([d, d])
    l2 = np.linalg.norm(p2 - p1)
    euclid.append(l2)
    w_euclid.append(l2 ** (-power_median))
    w_manhattan.append(np.sum(np.abs(p2 - p1)) ** (-power_median))
    w_czebyszev.append(np.max(np.abs(p2 - p1)) ** (-power_median))

plt.figure(figsize=(5, 3))
plt.plot(euclid, w_euclid, label=r"Weights ($\beta=2$)")
plt.plot(euclid, w_manhattan, label=r"Weights ($\beta=1$)")
plt.plot(euclid, w_czebyszev, label=r"Weights ($\beta=\infty$)")
plt.xscale("log")
plt.xlabel("Euclidean distance")
plt.ylabel("Weight")
plt.legend()
plt.savefig(METRICS_PATH / "idw_weights_by_distance.pdf", bbox_inches="tight")
plt.show()
