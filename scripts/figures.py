"""Every figure, from scores.csv, in one place.

    python scripts/figures.py

The original had five plotting scripts totalling twelve hundred lines, each
loading its own JSON and defining its own shortening and colour helpers. They
are one file here because they draw the same numbers, and a change to how a
model is named should not have to be made five times.

Nothing here computes a metric. Figures read scores.csv, so a figure and a table
in the write-up cannot disagree.
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from settings import FIGURES_DIR, SCORES_PATH
from utils import model_slug, read_table, section

# One view is reported in the figures, and which one is stated on every caption
# rather than left to the reader to assume.
VIEW = 'judged'
SIZE = (9, 5)


# Define function to shorten a model identifier for an axis
def short(model_id):
    return model_slug(model_id).replace('-Instruct', '').replace('-4bit', '')


# Define function to save a figure under a name that says what it shows
def save(figure, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f'{name}.png'
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# Define function to draw how the methods compare, per model
def figure_methods(scores):
    overall = scores[(scores['direction'] == 'overall') &
                     (scores['view'] == VIEW)].copy()
    overall['label'] = overall['method'] + overall['shots'].replace('0', '')
    figure, axes = plt.subplots(figsize=SIZE)
    models = sorted(overall['model'].unique())
    labels = sorted(overall['label'].unique())
    width = 0.8 / max(len(models), 1)
    for index, model in enumerate(models):
        part = overall[overall['model'] == model].set_index('label')
        values = [float(part.loc[l, 'mcc']) if l in part.index else 0
                  for l in labels]
        axes.bar([i + index * width for i in range(len(labels))], values,
                 width, label=short(model))
    axes.set_xticks([i + width * (len(models) - 1) / 2
                     for i in range(len(labels))])
    axes.set_xticklabels(labels, rotation=20, ha='right')
    axes.set_ylabel('Matthews correlation')
    axes.set_title(f'Detection by method, {VIEW} replies only')
    axes.legend()
    axes.grid(axis='y', alpha=0.3)
    return save(figure, 'methods')


# Define function to draw how detection varies by translation direction, which
# is the question the multilingual framing exists to ask
def figure_directions(scores):
    per = scores[(scores['direction'] != 'overall') &
                 (scores['view'] == VIEW) &
                 (scores['method'] == 'baseline')]
    if per.empty:
        return None
    figure, axes = plt.subplots(figsize=(9, 7))
    order = (per.groupby('direction')['mcc'].apply(
        lambda s: s.astype(float).mean()).sort_values().index)
    for model in sorted(per['model'].unique()):
        part = per[per['model'] == model].set_index('direction')
        axes.plot([float(part.loc[d, 'mcc']) if d in part.index else None
                   for d in order], list(order), 'o-', label=short(model))
    axes.set_xlabel('Matthews correlation')
    axes.set_title(f'Detection by direction, baseline, {VIEW} replies only')
    axes.legend()
    axes.grid(axis='x', alpha=0.3)
    return save(figure, 'directions')


# Define function to draw how much the three views differ, which is the figure
# that shows whether refusals are doing any work
def figure_views(scores):
    overall = scores[scores['direction'] == 'overall']
    figure, axes = plt.subplots(figsize=SIZE)
    views = ['all', 'parsed', 'judged']
    for model in sorted(overall['model'].unique()):
        part = overall[(overall['model'] == model) &
                       (overall['method'] == 'baseline')].set_index('view')
        axes.plot(views, [float(part.loc[v, 'mcc']) if v in part.index else None
                          for v in views], 'o-', label=short(model))
    axes.set_ylabel('Matthews correlation')
    axes.set_title('The same replies under three views, baseline')
    axes.legend()
    axes.grid(axis='y', alpha=0.3)
    return save(figure, 'views')


# Define function to draw whether adding examples helps
def figure_shots(scores):
    few = scores[(scores['direction'] == 'overall') &
                 (scores['view'] == VIEW) &
                 (scores['method'].isin(['baseline', 'fewshot']))]
    if few.empty:
        return None
    figure, axes = plt.subplots(figsize=SIZE)
    for model in sorted(few['model'].unique()):
        part = few[few['model'] == model].sort_values('shots')
        axes.plot(part['shots'].astype(int), part['mcc'].astype(float),
                  'o-', label=short(model))
    axes.set_xlabel('Worked examples in the prompt')
    axes.set_ylabel('Matthews correlation')
    axes.set_title(f'Effect of in context examples, {VIEW} replies only')
    axes.legend()
    axes.grid(alpha=0.3)
    return save(figure, 'shots')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not SCORES_PATH.exists():
        raise SystemExit('No scores.csv. Run evaluate.py first.')
    scores = read_table(SCORES_PATH)

    section('Figures')
    for draw in [figure_methods, figure_directions, figure_views, figure_shots]:
        path = draw(scores)
        if path:
            print(f'  {path.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
