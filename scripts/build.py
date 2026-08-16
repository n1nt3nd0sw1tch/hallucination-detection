"""Turns the dataset into the two tables every later stage reads.

    python scripts/build.py

items.csv    one row per source and translation pair, with its gold label
prompts.csv  one row per item asked one way, with the rendered text

Building the prompts once and storing them means generation reads a file rather
than reconstructing a template, and the exact text sent to a model can be read
back from one place months later. It also makes the count of what a run will do
visible before the run starts, which the original pipeline could not tell you
without running it.
"""

import argparse
import random

import pandas as pd

import prompts as templates
from settings import (DIRECTIONS, GENERATION, HALLUCINATION, HALOMI_PATH,
                      ITEMS_PATH, ITEMS_PER_DIRECTION, ITEM_COLUMNS, METHODS,
                      NEGATIVE_GRADE, NO_HALLUCINATION, PROMPTS_PATH,
                      PROMPT_COLUMNS, SAMPLE_SEED, direction_name,
                      make_item_id, make_prompt_id)
from utils import (announce, make_directories, read_table, section, write_table)

# Examples for the few shot prompts are drawn once with this seed, so that every
# model and every replicate sees the same examples and a difference between them
# cannot be a difference in what they were shown.
EXAMPLE_SEED = 20260816


# ----------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------

# Define function to read HalOmi and keep only the directions configured
def build_items():
    raw = pd.read_csv(HALOMI_PATH, sep='\t')
    wanted = set(DIRECTIONS)
    rows = []
    for (source, target), group in raw.groupby(['src_lang', 'tgt_lang'],
                                               sort=True):
        if (source, target) not in wanted:
            continue
        for index, (_, row) in enumerate(group.iterrows()):
            grade = str(row['class_hall'])
            rows.append({
                'item_id': make_item_id(source, target, index),
                'source_language': source,
                'target_language': target,
                'direction': direction_name(source, target),
                'grade': grade,
                'answer': (NO_HALLUCINATION if grade == NEGATIVE_GRADE
                           else HALLUCINATION),
                'source_text': str(row['src_text']).strip(),
                'target_text': str(row['mt_text']).strip(),
            })
    return sample_items(pd.DataFrame(rows))


# Define function to take a fixed number of items from each direction, keeping
# the label balance of that direction rather than drawing at random over the
# whole frame, so that a direction which is mostly hallucinated stays that way
def sample_items(items):
    if not ITEMS_PER_DIRECTION:
        return items
    drawn = []
    for direction, group in items.groupby('direction', sort=True):
        wanted = min(ITEMS_PER_DIRECTION, len(group))
        share = (group['answer'] == HALLUCINATION).mean()
        positive = group[group['answer'] == HALLUCINATION]
        negative = group[group['answer'] == NO_HALLUCINATION]
        take_positive = min(len(positive), round(wanted * share))
        take_negative = min(len(negative), wanted - take_positive)
        drawn.append(pd.concat([
            positive.sample(take_positive, random_state=SAMPLE_SEED),
            negative.sample(take_negative, random_state=SAMPLE_SEED)]))
    return pd.concat(drawn).sort_values('item_id').reset_index(drop=True)


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

# Define function to draw the worked examples one direction's few shot prompts
# use. They are drawn from the same direction, balanced between the two labels,
# and never include the item being asked about.
def draw_examples(items, direction, count, chooser):
    pool = items[items['direction'] == direction]
    positive = pool[pool['answer'] == HALLUCINATION].to_dict('records')
    negative = pool[pool['answer'] == NO_HALLUCINATION].to_dict('records')
    drawn = []
    for index in range(count):
        source = positive if index % 2 == 0 else negative
        if source:
            drawn.append(chooser.choice(source))
    return drawn


# Define function to render every prompt the configuration asks for
def build_prompts(items):
    chooser = random.Random(EXAMPLE_SEED)
    examples = {}
    rows = []

    for method, spec in METHODS.items():
        # debate needs a first judgement to review, so its prompts are built
        # during generation rather than here
        if method == 'debate':
            continue
        shots_list = spec['shots'] if isinstance(spec['shots'], list) \
            else [spec['shots']]

        for shots in shots_list:
            for item in items.to_dict('records'):
                drawn = ()
                if shots:
                    key = (item['direction'], shots)
                    if key not in examples:
                        examples[key] = draw_examples(items, item['direction'],
                                                      shots, chooser)
                    drawn = [e for e in examples[key]
                             if e['item_id'] != item['item_id']]
                rows.append({
                    'prompt_id': make_prompt_id(item['item_id'], method, shots),
                    'item_id': item['item_id'],
                    'method': method,
                    'shots': shots,
                    'direction': item['direction'],
                    'answer': item['answer'],
                    'prompt': templates.build(method, item, drawn),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

# Define function to fail loudly on anything that would quietly spoil a run
def validate(items, prompts):
    assert items['item_id'].is_unique, 'an item identifier is repeated'
    assert prompts['prompt_id'].is_unique, 'a prompt identifier is repeated'
    assert not items['source_text'].eq('').any(), 'an item has no source text'
    assert not items['target_text'].eq('').any(), 'an item has no translation'
    assert not prompts['prompt'].eq('').any(), 'a prompt rendered empty'
    missing = set(prompts['item_id']) - set(items['item_id'])
    assert not missing, f'{len(missing)} prompts refer to unknown items'
    for method in prompts['method'].unique():
        rendered = prompts[prompts['method'] == method]['prompt']
        assert rendered.str.contains('Label:').all(), \
            f'{method} does not ask for the answer format'


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    make_directories()

    section('Items')
    items = build_items()
    write_table(items, ITEMS_PATH, ITEM_COLUMNS)
    announce('items', len(items))
    for direction, count in items.groupby('direction').size().items():
        share = (items[items['direction'] == direction]['answer']
                 == HALLUCINATION).mean()
        print(f'    {direction:<26} {count:>4}  {share:.0%} hallucinated')

    section('Prompts')
    prompts = build_prompts(items)
    validate(items, prompts)
    write_table(prompts, PROMPTS_PATH, PROMPT_COLUMNS)
    announce('prompts', len(prompts))
    for method, count in prompts.groupby('method').size().items():
        print(f'    {method:<12} {count:>6}')

    section('What a full pass will cost')
    if ITEMS_PER_DIRECTION:
        print(f'  sampled to {ITEMS_PER_DIRECTION} items a direction, '
              f'set items_per_direction to 0 in the config for all of them')
    replicates = GENERATION['replicates']
    calls = len(prompts) * replicates
    print(f'  {len(prompts):,} prompts by {replicates} replicates '
          f'= {calls:,} calls a model')
    print(f'  {calls * 2:,} across the two models in the panel')
    print(f'  debate adds a second round over the baseline prompts')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
