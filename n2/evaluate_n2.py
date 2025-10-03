#!/usr/bin/env python3
"""Combined evaluator for n2 tasks (1: matching, 2: age, 3: OCR/fraud).

Usage examples:
  python evaluate_n2.py --task 1 --submission n2/outputs/tache1/submissions/task1_matching_submission.csv
  python evaluate_n2.py --task 2 --submission n2/outputs/tache2/submissions/task2_age_submission.csv
  python evaluate_n2.py --task 3 --submission n2/outputs/tache3/submissions/task3_ocr_fraud_submission.csv

If --ground_truth is provided, the script will attempt to compute the matching metric (accuracy) or MAE depending on the task.
"""
import argparse
from pathlib import Path
import sys
import csv

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None


def load_csv(path):
    if pd:
        return pd.read_csv(path)
    else:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)


# Task 1: matching
def eval_task1(submission, ground_truth=None, limit=20):
    sub = load_csv(submission)
    print('Submission loaded:', submission)
    if ground_truth and Path(ground_truth).exists():
        gt = load_csv(ground_truth)
        # attempt pandas merge
        if pd and isinstance(sub, pd.DataFrame) and isinstance(gt, pd.DataFrame):
            merged = gt.merge(sub, on='image', how='left')
            if 'predicted_id' in merged.columns and 'true_id' in merged.columns:
                correct = (merged['predicted_id'] == merged['true_id']).sum()
                total = len(merged)
                print(f'Accuracy: {correct/total:.4f} ({int(correct)}/{int(total)})')
                return
        # fallback
        gt_map = {r['image']: r.get('true_id') or r.get('target_id') for r in gt}
        total = 0
        correct = 0
        for r in sub:
            img = r.get('image')
            pred = r.get('predicted_id')
            if img in gt_map and gt_map[img] is not None:
                total += 1
                if str(pred) == str(gt_map[img]):
                    correct += 1
        if total>0:
            print(f'Accuracy: {correct/total:.4f} ({correct}/{total})')
        else:
            print('No overlapping labelled rows found to compute accuracy.')
    else:
        print('\nNo ground truth provided. Showing sample predictions and stats:')
        if pd and isinstance(sub, pd.DataFrame):
            print(sub.head(limit).to_string(index=False))
            if 'predicted_id' in sub.columns:
                print('\nUnique predicted IDs:', sub['predicted_id'].nunique())
            print('Total predictions:', len(sub))
        else:
            for r in sub[:limit]:
                print(r)
            preds = set(r.get('predicted_id') for r in sub)
            print('\nUnique predicted IDs (sampled):', len(preds))
            print('Total predictions:', len(sub))


# Task 2: age
def eval_task2(submission, ground_truth=None, limit=20):
    sub = load_csv(submission)
    print('Submission loaded:', submission)
    if ground_truth and Path(ground_truth).exists():
        gt = load_csv(ground_truth)
        # pandas path
        if pd and isinstance(sub, pd.DataFrame) and isinstance(gt, pd.DataFrame):
            # assume both have column 'image' and 'age' (gt has 'age')
            merged = gt.merge(sub, on='image', how='left')
            if 'age_x' in merged.columns and 'age_y' in merged.columns:
                try:
                    mae = (merged['age_y'].astype(float) - merged['age_x'].astype(float)).abs().mean()
                    print(f'MAE: {mae:.3f} over {len(merged)} samples')
                    return
                except Exception:
                    pass
        # fallback
        gt_map = {r['image']: float(r.get('age')) for r in gt}
        total = 0
        s = 0.0
        for r in sub:
            img = r.get('image')
            pred = r.get('age')
            if img in gt_map and pred not in (None, ''):
                total += 1
                s += abs(float(pred) - gt_map[img])
        if total>0:
            print(f'MAE: {s/total:.3f} over {total} samples')
        else:
            print('No overlapping labelled rows found to compute MAE.')
    else:
        print('\nNo ground truth provided. Showing sample predictions and stats:')
        if pd and isinstance(sub, pd.DataFrame):
            print(sub.head(limit).to_string(index=False))
            if 'age' in sub.columns:
                try:
                    print('\nStatistics: age min/max/mean ->', sub['age'].astype(float).min(), sub['age'].astype(float).max(), sub['age'].astype(float).mean())
                except Exception:
                    pass
        else:
            for r in (sub[:limit]):
                print(r)
            ages = []
            for r in sub:
                a = r.get('age')
                if a not in (None, ''):
                    try:
                        ages.append(float(a))
                    except:
                        pass
            if ages:
                print('Stats: min/max/mean =>', min(ages), max(ages), sum(ages)/len(ages))


# Task 3: OCR & fraud
def eval_task3(submission, ground_truth=None, limit=20):
    sub = load_csv(submission)
    print('Submission loaded:', submission)
    if not sub:
        print('Submission empty or unreadable')
        return
    if ground_truth and Path(ground_truth).exists():
        gt = load_csv(ground_truth)
        if pd and isinstance(sub, pd.DataFrame) and isinstance(gt, pd.DataFrame):
            merged = gt.merge(sub, on='image', how='left')
            if 'class' in merged.columns and 'true_class' in merged.columns:
                correct = (merged['class'] == merged['true_class']).sum()
                total = len(merged)
                print(f'Accuracy: {correct/total:.4f} ({int(correct)}/{int(total)})')
                return
        gt_map = {r['image']: r.get('class') or r.get('true_class') for r in gt}
        total = 0
        ok = 0
        for r in sub:
            img = r.get('image')
            pred = r.get('class')
            if img in gt_map and gt_map[img] is not None:
                total += 1
                if str(pred) == str(gt_map[img]):
                    ok += 1
        if total>0:
            print(f'Accuracy: {ok/total:.4f} ({ok}/{total})')
        else:
            print('No overlapping labelled rows found to compute accuracy.')
    else:
        print('\nNo ground truth provided. Showing sample predictions and OCR text:')
        if pd and isinstance(sub, pd.DataFrame):
            print(sub.head(limit).to_string(index=False))
        else:
            for r in sub[:limit]:
                print(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[1,2,3], required=True, help='Task number to evaluate')
    parser.add_argument('--submission', type=Path, required=False)
    parser.add_argument('--ground_truth', type=Path, required=False)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    # set defaults if not provided
    if not args.submission:
        defaults = {1: Path('n2/outputs/tache1/submissions/task1_matching_submission.csv'),
                    2: Path('n2/outputs/tache2/submissions/task2_age_submission.csv'),
                    3: Path('n2/outputs/tache3/submissions/task3_ocr_fraud_submission.csv')}
        args.submission = defaults.get(args.task)

    if not args.submission.exists():
        print('Submission not found:', args.submission)
        sys.exit(1)

    if args.task == 1:
        eval_task1(args.submission, args.ground_truth, args.limit)
    elif args.task == 2:
        eval_task2(args.submission, args.ground_truth, args.limit)
    else:
        eval_task3(args.submission, args.ground_truth, args.limit)


if __name__ == '__main__':
    main()
