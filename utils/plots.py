from scipy.interpolate import make_smoothing_spline
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import pickle

def bootstrap_conf(
    kl_acc_array,
    n_bootstrap=5000,
    alpha=0.05,
    random_seed=42
):
    """
    Cluster bootstrap over prompts.
    kl_acc_array : np.ndarray
        Shape (n_prompts, n_completions, 2)
        Columns: [KL, ACC]

    Returns dict with mean + percentile CI
    """

    rng = np.random.default_rng(random_seed)

    n_prompts = kl_acc_array.shape[0]

    # ---- Estimand (population estimate) ----
    kl_mean = kl_acc_array[:, :, 0].mean()
    acc_mean = kl_acc_array[:, :, 1].mean()

    boot_kl = np.empty(n_bootstrap)
    boot_acc = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_prompts, n_prompts)
        sample = kl_acc_array[idx]  # (n_prompts, n_completions, 2)

        # 2. compute statistic on resampled dataset
        boot_kl[i] = sample[:, :, 0].mean()
        boot_acc[i] = sample[:, :, 1].mean()

    # ---- percentile CI ----
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    kl_ci = np.percentile(boot_kl, [lower, upper])
    acc_ci = np.percentile(boot_acc, [lower, upper])

    return {
        "kl": {
            "ci_mean": float(kl_mean),
            "ci_lower": float(kl_ci[0]),
            "ci_upper": float(kl_ci[1]),
        },
        "accuracy": {
            "ci_mean": float(acc_mean),
            "ci_lower": float(acc_ci[0]),
            "ci_upper": float(acc_ci[1]),
        },
    }


def plot_sentiment_bench_acc_v_steps(data, experiment_tag, beta=None):
    if beta:
        title = "Accuracy" + rf"($\beta={beta}$)"
    fig = plt.figure(figsize=(8, 6))
    for model_name in data.keys():
        label = data[model_name]['label']
        conf_results = data[model_name]['conf_results']

        t = np.array(data[model_name]['checkpoint_list'])
        t_min, t_max = t.min(), t.max()
        t = (t - t_min)/(t_max-t_min)

        y_mean_spline = make_smoothing_spline(t, [c['accuracy']['ci_mean'] for c in conf_results])
        y_lower_spline = make_smoothing_spline(t, [c['accuracy']['ci_lower'] for c in conf_results])
        y_upper_spline = make_smoothing_spline(t, [c['accuracy']['ci_upper'] for c in conf_results])
        

        t_smooth = np.linspace(t.min(), t.max(), 100 * len(t))
        y_mean = y_mean_spline(t_smooth)
        y_lower = y_lower_spline(t_smooth)
        y_upper = y_upper_spline(t_smooth)
        

        mean_line = plt.plot((t_smooth)*(t_max-t_min)+t_min, y_mean, '-', linewidth=1.5, label=label)
        color = mean_line[0].get_color()
        plt.scatter(np.array(data[model_name]['checkpoint_list']), [c['accuracy']['ci_mean'] for c in conf_results], color=color, s=10)
        plt.fill_between((t_smooth)*(t_max-t_min)+t_min, y_upper, y_lower, color=color, alpha=0.2)



    plt.legend(loc='upper left', framealpha=0.35)
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./results/plots/{experiment_tag}_acc_v_steps.png")
    with open(f"./results/plots/{experiment_tag}_acc_v_steps.png.pickle", 'wb') as f:
        pickle.dump(fig, f)
    plt.close()

def plot_sentiment_bench_kl_v_steps(data, experiment_tag, beta=None):
    if beta:
        title = "$D_{KL}$" + rf"($\beta={beta}$)"
    fig = plt.figure(figsize=(8, 6))
    for model_name in data.keys():
        label = data[model_name]['label']
        conf_results = data[model_name]['conf_results']
        t = np.array(data[model_name]['checkpoint_list'])
        t_min, t_max = t.min(), t.max()
        t = (t - t_min)/(t_max-t_min)

        x_mean_spline = make_smoothing_spline(t, [c['kl']['ci_mean'] for c in conf_results])
        x_lower_spline = make_smoothing_spline(t, [c['kl']['ci_lower'] for c in conf_results])
        x_upper_spline = make_smoothing_spline(t, [c['kl']['ci_upper'] for c in conf_results])


        t_smooth = np.linspace(t.min(), t.max(), 100 * len(t))
        x_mean = x_mean_spline(t_smooth)
        x_lower = x_lower_spline(t_smooth)
        x_upper = x_upper_spline(t_smooth)

        mean_line = plt.plot((t_smooth)*(t_max-t_min)+t_min, x_mean, '-', linewidth=1.5, label=label)
        color = mean_line[0].get_color()
        plt.scatter(np.array(data[model_name]['checkpoint_list']), [c['kl']['ci_mean'] for c in conf_results], color=color, s=10)
        plt.fill_between((t_smooth)*(t_max-t_min)+t_min, x_upper, x_lower, color=color, alpha=0.15)
    
    plt.legend(loc='upper left', framealpha=0.35)
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./results/plots/{experiment_tag}_kl_v_steps.png")
    with open(f"./results/plots/{experiment_tag}_kl_v_steps.png.pickle", 'wb') as f:
        pickle.dump(fig, f)
    plt.close()

def plot_sentiment_bench_acc_v_kl(data, experiment_tag, beta=None):
    if beta:
        title = "Accuracy VS $D_{KL}$" + rf"($\beta={beta}$)"
    fig = plt.figure(figsize=(8, 6))
    for model_name in data.keys():
        label = data[model_name]['label']
        conf_results = data[model_name]['conf_results']

        t = np.array(data[model_name]['checkpoint_list'])
        t = (t - t.min())/(t.max()-t.min())

        x_mean_spline = make_smoothing_spline(t, [c['kl']['ci_mean'] for c in conf_results])
        y_mean_spline = make_smoothing_spline(t, [c['accuracy']['ci_mean'] for c in conf_results])

        t_smooth = np.linspace(t.min(), t.max(), 100 * len(t))
        x_mean = x_mean_spline(t_smooth)
        y_mean = y_mean_spline(t_smooth)

        mean_line = plt.plot(x_mean, y_mean, '-', linewidth=1.5)
        color = mean_line[0].get_color()
        plt.plot(x_mean, y_mean, '-', linewidth=1.5, color=color)
        plt.scatter([c['kl']['ci_mean'] for c in conf_results], [c['accuracy']['ci_mean'] for c in conf_results], label=label, color=color)

    plt.legend(loc='lower right', framealpha=0.35)
    plt.grid(which="both")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./results/plots/{experiment_tag}_acc_v_kl.png")
    with open(f"./results/plots/{experiment_tag}_acc_v_kl.png.pickle", 'wb') as f:
        pickle.dump(fig, f)
    plt.close()

def plot_sentiment_bench(data, experiment_tag, beta=None, conf_n_bootstrap=500, conf_alpha=0.05):
    for model_name in data.keys():
        match = re.search(r'imdb-(\w+)-sentiment', model_name)
        if match:
            alg = match.group(1)
            label = alg.upper()
        else:
            continue

        match = re.search(r'-a([\d.]+)', model_name)
        if match:
            alpha = (match.group(1))
            label = label + r" $\alpha" + f"={alpha}$"

        data[model_name]["conf_results"] = [bootstrap_conf(np.array(checkpoint), n_bootstrap=conf_n_bootstrap, alpha=conf_alpha) for checkpoint in data[model_name]["checkpoint_results"]]
        data[model_name]["label"] = label

    plot_sentiment_bench_acc_v_steps(data, experiment_tag, beta)
    plot_sentiment_bench_kl_v_steps(data, experiment_tag, beta)
    plot_sentiment_bench_acc_v_kl(data, experiment_tag, beta)
