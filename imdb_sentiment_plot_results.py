import argparse
import json
from utils.plots import plot_sentiment_bench 
import os
import re

if __name__ == "__main__":
    json_results = [f for f in os.listdir('./results') if f.endswith('.json')]
    
    for res in json_results:
        with open(f'./results/{res}', 'r', encoding='utf-8') as file:
            data = json.load(file)
        match = re.search(r'b([\d.]+)', res)
        title = r"Rewards VS $D_{KL}$"
        if match:
            beta = match.group(1)
            title = title + fr" ($\beta={beta}$)"

        filename = res.replace('.json', '')
        print("Done", filename)
        plot_sentiment_bench(data, filename, title=title)