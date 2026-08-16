"""Reads the replies into labels, and the labels into metrics.

    python scripts/evaluate.py                   every model and method
    python scripts/evaluate.py --model <id>

Parsing, refusal detection and scoring live together because they are one
decision made in three steps: what did the model say, was it a judgement at all,
and how good was it.

Three views of the same data are reported, because dropping a case changes what
a score means and the reader should see all three rather than only the one that
flatters:

    all             every reply, an unparseable one counted as wrong
    parsed          replies that gave a label
    judged          replies that gave a label and were not refusals

A refusal is not a wrong answer, it is an absent one. Counting it as wrong
understates a model that declines where another guesses, which is a difference
worth seeing rather than hiding.
"""

import argparse
import re

import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

from settings import (HALLUCINATION, JUDGEMENTS_DIR, METHODS, MODELS,
                      NO_HALLUCINATION, PROMPTS_PATH, SCORES_PATH)
from utils import (announce, model_slug, read_lines, read_table, response_path,
                   section, write_table)

# The label line every method asks for. Matched at the end of the reply, since a
# model that reasons first will mention both labels on the way.
LABEL = re.compile(r'label\s*:\s*(no\s+hallucination|hallucination)',
                   re.IGNORECASE)

# A reply that declines rather than judges. The classifier in the panel decides
# when it is available; these carry the decision when it is not, and are kept
# deliberately narrow so that a reply merely discussing refusal is not caught.
REFUSALS = [
    r"\bi can(?:not|'t)\s+(?:assist|help|comply|provide|do that)\b",
    r"\bi(?:'m| am)\s+(?:unable|not able)\s+to\b",
    r"\bi\s+must\s+(?:decline|refuse)\b",
    r"\bi\s+won't\s+(?:assist|help|provide)\b",
    r"\b(?:sorry|apolog\w+)[, ]+but\s+i\s+can(?:not|'t)\b",
]
REFUSAL = re.compile('|'.join(REFUSALS), re.IGNORECASE)


# ----------------------------------------------------------------------------
# Reading one reply
# ----------------------------------------------------------------------------

# Define function to read the label out of a reply, returning nothing when it
# did not give one rather than guessing
def parse_label(reply):
    text = str(reply or '').strip()
    if not text:
        return ''
    found = LABEL.findall(text)
    if not found:
        return ''
    # the last one, since a reasoning chain states the alternatives on the way
    last = re.sub(r'\s+', ' ', found[-1]).lower()
    return NO_HALLUCINATION if last.startswith('no') else HALLUCINATION


# Define function to say whether a reply declined rather than judged
def is_refusal(reply):
    text = str(reply or '').strip()
    if not text:
        return False
    # a reply that gave a label was a judgement, whatever else it said
    return bool(REFUSAL.search(text)) and not parse_label(text)


# ----------------------------------------------------------------------------
# Reading a pass
# ----------------------------------------------------------------------------

# Define function to turn one model's replies into one row per judgement
def judge(model_id, method):
    replies = read_lines(response_path(model_id, method))
    if replies.empty:
        return pd.DataFrame()
    prompts = read_table(PROMPTS_PATH).set_index('prompt_id')

    rows = []
    for reply in replies.to_dict('records'):
        if str(reply.get('error') or '').strip():
            continue
        prompt_id = str(reply['prompt_id'])
        if prompt_id not in prompts.index:
            continue
        spec = prompts.loc[prompt_id]
        observed = parse_label(reply.get('response'))
        rows.append({
            'prompt_id': prompt_id,
            'model': model_id,
            'replicate': str(reply['replicate']),
            'method': method,
            'shots': spec['shots'],
            'direction': spec['direction'],
            'answer': spec['answer'],
            'observed': observed,
            'parsed': bool(observed),
            'refusal': is_refusal(reply.get('response')),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

# Define function to score one set of judgements. An unparsed reply is counted
# as the wrong answer in the all view, since a model that cannot be read has not
# answered, and dropping it silently would reward being unreadable.
def score(frame, view):
    if frame.empty:
        return None
    truth = (frame['answer'] == HALLUCINATION).astype(int)
    if view == 'all':
        opposite = 1 - truth
        told = frame['observed'].map({HALLUCINATION: 1, NO_HALLUCINATION: 0})
        told = told.fillna(pd.Series(opposite, index=frame.index)).astype(int)
    else:
        told = (frame['observed'] == HALLUCINATION).astype(int)
    if truth.nunique() < 2 or told.nunique() < 2:
        return None
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, told, average='binary', zero_division=0)
    return {'n': len(frame), 'mcc': matthews_corrcoef(truth, told),
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': float((truth == told).mean())}


# Define function to apply one view to a set of judgements
def view_of(frame, view):
    if view == 'all':
        return frame
    if view == 'parsed':
        return frame[frame['parsed']]
    return frame[frame['parsed'] & ~frame['refusal']]


VIEWS = ['all', 'parsed', 'judged']


# Define function to score every model, method and view, and every direction
# within them
def score_everything(judgements):
    rows = []
    groups = [('overall', judgements.assign(direction='overall'))]
    for direction, group in judgements.groupby('direction'):
        groups.append((direction, group))

    for direction, group in groups:
        for (model, method, shots), part in group.groupby(
                ['model', 'method', 'shots']):
            for view in VIEWS:
                measured = score(view_of(part, view), view)
                if not measured:
                    continue
                rows.append({'model': model, 'method': method, 'shots': shots,
                             'direction': direction, 'view': view,
                             'refusals': int(part['refusal'].sum()),
                             'unparsed': int((~part['parsed']).sum()),
                             **{k: round(v, 4) if isinstance(v, float) else v
                                for k, v in measured.items()}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='', help='one model, or all of them')
    arguments = parser.parse_args()

    wanted = ([arguments.model] if arguments.model
              else [entry['id'] for entry in MODELS.values()])

    section('Judgements')
    collected = []
    for model_id in wanted:
        for method in METHODS:
            judged = judge(model_id, method)
            if judged.empty:
                continue
            path = JUDGEMENTS_DIR / method / f'{model_slug(model_id)}.csv'
            write_table(judged, path)
            collected.append(judged)
            print(f'  {model_slug(model_id):<28} {method:<12} '
                  f'{len(judged):>6} replies, '
                  f'{int((~judged["parsed"]).sum()):>4} unparsed, '
                  f'{int(judged["refusal"].sum()):>4} refusals')

    if not collected:
        raise SystemExit('No replies to judge. Run run.py generate first.')

    judgements = pd.concat(collected, ignore_index=True)

    section('Agreement between replicates')
    agreed = (judgements.groupby(['prompt_id', 'model'])['observed']
              .nunique() == 1)
    print(f'  {agreed.mean():.1%} of cells where every replicate agreed')

    section('Scores')
    scores = score_everything(judgements)
    write_table(scores, SCORES_PATH)
    announce('rows', len(scores))

    headline = scores[(scores['direction'] == 'overall') &
                      (scores['view'] == 'judged')]
    print()
    print(f"  {'model':<24} {'method':<12} {'shots':>5} {'n':>6} {'mcc':>7}")
    for _, row in headline.sort_values(['model', 'method', 'shots']).iterrows():
        print(f"  {model_slug(row['model']):<24} {row['method']:<12} "
              f"{row['shots']:>5} {row['n']:>6} {row['mcc']:>7.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
