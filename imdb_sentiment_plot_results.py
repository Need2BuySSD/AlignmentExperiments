import argparse
import json
from utils.plots import plot_sentiment_bench
import os
import re
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":\
    # JSON files
    json_results = [f for f in os.listdir('./results') if f.endswith('.json')]
    for res in json_results:
        with open(f'./results/{res}', 'r', encoding='utf-8') as file:
            data = json.load(file)
        match = re.search(r'b([\d.]+)', res)
        if match:
            beta = match.group(1)
        else:
            beta = None

        experiment_tag = res.replace('.json', '')
        plot_sentiment_bench(data, experiment_tag, beta=beta, conf_n_bootstrap=2000, conf_alpha=0.05)
        print("Done", experiment_tag)