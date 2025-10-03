#!/usr/bin/env python3
"""Evaluate data quality for Challenge 1 (n1).

Produces:
 - n1/output/metrics_data_quality.json
 - n1/output/missingness_bar.png
 - n1/output/anomalies_top10.png
 - n1/output/anomalies_sample.csv

Usage:
  python n1/evaluate_n1.py --clean n1/output/WPP2024_DEMOGRAPHIC_CLEAN.csv --anomalies n1/output/WPP2024_DEMOGRAPHIC_ANOMALIES.csv

If files are missing the script exits cleanly with diagnostics.
"""
from pathlib import Path
import argparse
import json
import sys

try:
    import pandas as pd
except Exception:
    pd = None

import numpy as np
import matplotlib.pyplot as plt


def ensure_pandas():
    if pd is None:
        print('pandas is required for this script. Please install it: pip install pandas')
        sys.exit(1)


def load_csv(path):
    ensure_pandas()
    p = Path(path)
    if not p.exists():
        print(f'File not found: {p}')
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f'Error reading {p}: {e}')
        return None


def compute_missingness(df):
    miss = df.isna().mean() * 100
    miss = miss.sort_values(ascending=False)
    return miss


def summary_numeric(df, n=10):
    num = df.select_dtypes(include=[np.number])
    stats = {}
    for c in num.columns:
        col = num[c].dropna()
        if len(col) == 0:
            continue
        stats[c] = {
            'count': int(col.count()),
            'min': float(col.min()),
            'p10': float(np.percentile(col, 10)),
            'median': float(np.median(col)),
            'mean': float(col.mean()),
            'p90': float(np.percentile(col, 90)),
            'max': float(col.max())
        }
    return stats


def detect_duplicate_keys(df):
    # try common key combinations
    keys = []
    candidates = ['ISO', 'ISO3', 'Country', 'CountryName', 'Location', 'LocID', 'Year']
    present = [c for c in candidates if c in df.columns]
    if 'Year' in present and ('ISO' in present or 'Location' in present or 'Country' in present):
        if 'ISO' in present:
            keys = ['ISO', 'Year']
        elif 'Location' in present:
            keys = ['Location', 'Year']
        else:
            keys = ['Country', 'Year']
    else:
        # fallback: try first two columns
        if len(df.columns) >= 2:
            keys = [df.columns[0], df.columns[1]]
    if not keys:
        return {'keys_used': [], 'duplicate_count': 0, 'total_rows': int(len(df))}
    dup = df.duplicated(subset=keys).sum()
    return {'keys_used': keys, 'duplicate_count': int(dup), 'total_rows': int(len(df))}


def save_metrics(out_dir, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / 'metrics_data_quality.json'
    with p.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print('Saved metrics to', p)


def plot_missingness(miss, out_dir):
    if miss is None or len(miss)==0:
        return None
    fig, ax = plt.subplots(figsize=(8, max(2, len(miss)*0.2)))
    miss.head(50).plot.barh(ax=ax)
    ax.set_xlabel('% missing')
    ax.set_title('Top 50 missingness by column')
    plt.tight_layout()
    out = out_dir / 'missingness_bar.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved missingness plot to', out)


def plot_anomalies_top10(df_anom, out_dir):
    if df_anom is None or len(df_anom)==0:
        return None
    # try to detect a column describing anomaly type
    possible = [c for c in ['issue','type','reason','anomaly_type'] if c in df_anom.columns]
    if possible:
        col = possible[0]
        top = df_anom[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6,4))
        top.plot.bar(ax=ax)
        ax.set_title(f'Top 10 anomaly types ({col})')
        plt.tight_layout()
        out = out_dir / 'anomalies_top10.png'
        fig.savefig(out)
        plt.close(fig)
        print('Saved anomalies top10 to', out)
        return out
    else:
        # fallback: just count rows by a generic reason column if exists
        if 'description' in df_anom.columns:
            top = df_anom['description'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,4))
            top.plot.bar(ax=ax)
            plt.tight_layout()
            out = out_dir / 'anomalies_top10.png'
            fig.savefig(out)
            plt.close(fig)
            print('Saved anomalies top10 to', out)
            return out
    return None


def create_anomalies_sample(df_anom, out_dir, n=200):
    out = out_dir / 'anomalies_sample.csv'
    if df_anom is None or len(df_anom)==0:
        print('No anomalies file to sample')
        return None
    df_anom.head(n).to_csv(out, index=False)
    print('Saved anomalies sample to', out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=Path, default=Path('n1/output/WPP2024_DEMOGRAPHIC_CLEAN.csv'))
    parser.add_argument('--anomalies', type=Path, default=Path('n1/output/WPP2024_DEMOGRAPHIC_ANOMALIES.csv'))
    parser.add_argument('--out_dir', type=Path, default=Path('n1/output'))
    args = parser.parse_args()

    df_clean = load_csv(args.clean)
    df_anom = load_csv(args.anomalies)

    metrics = {'files': {}, 'data_quality': {}, 'anomalies': {}}

    if df_clean is None:
        print('Clean file not found or unreadable. Exiting.')
        sys.exit(1)

    metrics['files']['clean_file'] = str(args.clean)
    metrics['files']['anomalies_file'] = str(args.anomalies) if args.anomalies.exists() else None

    # basic counts
    metrics['data_quality']['rows'] = int(len(df_clean))
    metrics['data_quality']['cols'] = int(len(df_clean.columns))

    # missingness
    miss = compute_missingness(df_clean)
    metrics['data_quality']['missingness_percent_by_column'] = miss.to_dict()

    # numeric summary
    metrics['data_quality']['numeric_summary'] = summary_numeric(df_clean)

    # duplicates
    metrics['data_quality']['duplicates'] = detect_duplicate_keys(df_clean)

    # anomalies summary
    if df_anom is not None:
        metrics['anomalies']['count'] = int(len(df_anom))
        # top by any label column
        possible = [c for c in ['issue','type','reason','anomaly_type','description'] if c in df_anom.columns]
        if possible:
            metrics['anomalies']['top_counts'] = df_anom[possible[0]].value_counts().head(20).to_dict()
        else:
            metrics['anomalies']['sample_rows'] = min(20, len(df_anom))
    else:
        metrics['anomalies']['count'] = 0

    # save metrics
    save_metrics(args.out_dir, metrics)

    # plots
    plot_missingness(miss, args.out_dir)
    plot_anomalies_top10(df_anom, args.out_dir)

    # anomalies sample
    create_anomalies_sample(df_anom, args.out_dir)


if __name__ == '__main__':
    main()
