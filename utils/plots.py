from scipy.interpolate import make_smoothing_spline
import numpy as np
import matplotlib.pyplot as plt
import json
import re

def plot_sentiment_bench(results, filename, title="Rewards VS $D_{KL}$"):
    plt.figure(figsize=(10, 5))
    for model_name, res in results.items():
        match = re.search(r'imdb-(\w+)-sentiment', model_name)
        if match:
            alg = match.group(1)
        else:
            continue

        label = alg.upper()

        match = re.search(r'-a([\d.]+)', model_name)
        if match:
            alpha = (match.group(1))
            label =label + r" $\alpha" + f"={alpha}$"

        X = np.array(res['kl'])
        Y = np.array(res['acc'])
        T = np.linspace(0, 1, len(X))
        x_spline = make_smoothing_spline(T, X)
        y_spline = make_smoothing_spline(T, Y)
        T = np.linspace(0, 1, 10*len(X))
        plt.plot(x_spline(T), y_spline(T))
        plt.scatter(X, Y, label=label)
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./results/plots/{filename}.png")