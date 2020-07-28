from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette([[0.2, 0.2, 0.2, 1.], [0.884375, 0.5265625, 0, 1.]])
output_dir = Path('E:/Coding/fincausal-paper')
COMBINED = False

if __name__ == '__main__':
    data = output_dir / 'backbones_performance.csv'

    df = pd.read_csv(data)
    aggregated = df.groupby('Model').agg(
        {
            'F1': ['min', 'max', 'mean'],
            'Exact match': ['min', 'max', 'mean'],
            'order': 'median'
        }
    )

    aggregated_f1 = aggregated[['F1', 'order']]
    aggregated_f1.columns = aggregated_f1.columns.droplevel()
    aggregated_f1 = aggregated_f1.sort_values('median')
    aggregated_f1 = aggregated_f1.reset_index()

    yerr = [aggregated_f1['mean'] - aggregated_f1['min'], aggregated_f1['max'] - aggregated_f1['mean']]
    sns.set_palette([[0.2, 0.2, 0.2, 1.], [0.884375, 0.5265625, 0, 1.]])
    sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 20}, font_scale=1.3)

    if COMBINED:
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
    else:
        plt.figure(figsize=(8, 5))

    ax = sns.pointplot(x="mean", y="Model", data=aggregated_f1, orient='h', ci=None, join=False)
    ax.set(xlabel='F1 score', ylabel='')
    plt.errorbar(y=list(range(len(aggregated_f1))), x=aggregated_f1['mean'],
                 xerr=yerr, fmt='none', capsize=3)
    plt.axhline(y=len(aggregated_f1) / 2 - 0.5, color='lightgrey', linewidth=1, linestyle='--')
    plt.tight_layout()
    if not COMBINED:
        plt.savefig(output_dir / 'f1_scores.pdf', dpi=300, format='pdf')

    aggregated_exact_match = aggregated[['Exact match', 'order']]
    aggregated_exact_match.columns = aggregated_exact_match.columns.droplevel()
    aggregated_exact_match = aggregated_exact_match.sort_values('median')
    aggregated_exact_match = aggregated_exact_match.reset_index()

    yerr = [aggregated_exact_match['mean'] - aggregated_exact_match['min'],
            aggregated_exact_match['max'] - aggregated_exact_match['mean']]

    if COMBINED:
        plt.subplot(1, 2, 2)
    else:
        plt.figure(figsize=(8, 5))
    ax = sns.pointplot(x="mean", y="Model", data=aggregated_exact_match, orient='h', ci=None, join=False)
    ax.set(xlabel='Exact match', ylabel='')
    plt.errorbar(y=list(range(len(aggregated_exact_match['mean']))), x=aggregated_exact_match['mean'],
                 xerr=yerr, fmt='none', capsize=3)
    plt.axhline(y=len(aggregated_f1) / 2 - 0.5, color='lightgrey', linewidth=1, linestyle='--')
    plt.tight_layout()
    if not COMBINED:
        plt.savefig(output_dir / 'exact_match.pdf', dpi=300, format='pdf')
    else:
        plt.savefig(output_dir / 'combined.pdf', dpi=300, format='pdf')
