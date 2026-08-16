"""Generates the replies, and checks a specification before spending on one.

    python scripts/run.py check
    python scripts/run.py generate --model <id> --method baseline
    python scripts/run.py generate --model <id>            every method
    python scripts/run.py debate --model <id> --reviewer <id>

Every stage resumes. A run that stops part way loses nothing, and running it
again asks only for what is missing, because what has been collected is read
back from the file rather than tracked in memory.
"""

import argparse
import textwrap

import backends
from prompts import as_messages, build
from settings import (GENERATION, ITEMS_PATH, METHODS, MODELS, PROMPTS_PATH,
                      RESPONSE_COLUMNS)
from utils import (WORKERS, announce, collect, make_directories, model_slug,
                   outstanding, read_lines, read_table, response_path, section)


# ----------------------------------------------------------------------------
# Looking before spending
# ----------------------------------------------------------------------------

# Define function to show one rendered prompt in full, so that what a model will
# be sent can be read before thousands of calls are made on it
def run_check(arguments):
    prompts = read_table(PROMPTS_PATH)
    items = read_table(ITEMS_PATH)

    chosen = prompts
    if arguments.method:
        chosen = chosen[chosen['method'] == arguments.method]
    if arguments.prompt:
        chosen = chosen[chosen['prompt_id'] == arguments.prompt]
    if chosen.empty:
        raise SystemExit('nothing matches that method or prompt')
    row = chosen.iloc[0]
    item = items[items['item_id'] == row['item_id']].iloc[0]

    section('Specification')
    print(f'  prompt     {row["prompt_id"]}')
    print(f'  method     {row["method"]}, {row["shots"]} examples')
    print(f'  direction  {row["direction"]}')
    print(f'  grade      {item["grade"]}')
    print(f'  expected   {row["answer"]}')

    section('Sent')
    for line in row['prompt'].split('\n'):
        print(textwrap.fill(line, 78, initial_indent='  ',
                            subsequent_indent='    ') if line else '')

    section('Coverage')
    announce('prompts', len(prompts))
    for method, count in prompts.groupby('method').size().items():
        print(f'    {method:<12} {count:>6}')
    return 0


# ----------------------------------------------------------------------------
# Generating
# ----------------------------------------------------------------------------

# Define function to list every reply one model owes for one method
def wanted_for(model_id, method, prompts, replicates, temperature):
    return [{'prompt_id': row['prompt_id'], 'model': model_id,
             'replicate': replicate, 'runtime': backends.runtime_of(model_id),
             'temperature': temperature}
            for _, row in prompts.iterrows()
            for replicate in range(1, replicates + 1)]


# Define function to generate one method's replies for one model
def generate_method(model_id, method, arguments):
    prompts = read_table(PROMPTS_PATH)
    prompts = prompts[prompts['method'] == method]
    if prompts.empty:
        print(f'  {method}: no prompts, skipped')
        return 0

    cap = METHODS[method]['max_tokens']
    path = response_path(model_id, method)
    wanted = wanted_for(model_id, method, prompts, arguments.replicates,
                        arguments.temperature)
    pending = outstanding(wanted=wanted, collected=read_lines(path),
                          keys=['prompt_id', 'replicate'])
    if arguments.limit:
        pending = pending[:arguments.limit]
    if not pending:
        print(f'  {method}: nothing outstanding')
        return 0

    text = dict(zip(prompts['prompt_id'], prompts['prompt']))

    def produce(item):
        reply = backends.generate(model_id, as_messages(text[item['prompt_id']]),
                                  cap, item['temperature'])
        # a reply that ends without the label line has been cut off, which is a
        # different failure from a reply that gives the wrong label
        return {'response': reply, 'refusal': '',
                'truncated': 'Label:' not in reply}

    return collect(pending=pending, produce=produce, path=path,
                   label=f'{model_slug(model_id)} {method}',
                   workers=arguments.workers, meter=backends.spent)


# Define function to generate every method the panel asks for
def run_generation(arguments):
    section(f'Generating with {arguments.model}')
    methods = [arguments.method] if arguments.method else [
        name for name in METHODS if name != 'debate']
    failures = 0
    for method in methods:
        failures += generate_method(arguments.model, method, arguments)
    print(f'\n  {backends.spent()}')
    return failures


# ----------------------------------------------------------------------------
# Review
# ----------------------------------------------------------------------------

# Define function to run the second round, in which one model reads another's
# judgement and decides for itself. It is separate from the other methods
# because it cannot be built before the first round exists.
def run_debate(arguments):
    section(f'{arguments.model} reviewing {arguments.reviewer}')
    prompts = read_table(PROMPTS_PATH)
    prompts = prompts[prompts['method'] == 'baseline']
    items = read_table(ITEMS_PATH).set_index('item_id')

    first = read_lines(response_path(arguments.reviewer, 'baseline'))
    if first.empty:
        raise SystemExit(f'{arguments.reviewer} has no baseline replies to '
                         f'review. Generate those first.')
    said = {(row['prompt_id'], str(row['replicate'])): row['response']
            for _, row in first.iterrows()}

    cap = METHODS['debate']['max_tokens']
    path = response_path(arguments.model, 'debate')
    wanted = wanted_for(arguments.model, 'debate', prompts,
                        arguments.replicates, arguments.temperature)
    pending = outstanding(wanted=wanted, collected=read_lines(path),
                          keys=['prompt_id', 'replicate'])
    pending = [item for item in pending
               if (item['prompt_id'], str(item['replicate'])) in said]
    if arguments.limit:
        pending = pending[:arguments.limit]
    if not pending:
        print('  nothing outstanding')
        return 0

    by_prompt = prompts.set_index('prompt_id')

    def produce(item):
        row = by_prompt.loc[item['prompt_id']]
        other = said[(item['prompt_id'], str(item['replicate']))]
        prompt = build('debate', items.loc[row['item_id']].to_dict(),
                       other={'label': other.strip().splitlines()[-1]
                              if other.strip() else '',
                              'reasoning': other.strip()})
        reply = backends.generate(arguments.model, as_messages(prompt), cap,
                                  item['temperature'])
        return {'response': reply, 'refusal': '',
                'truncated': 'Label:' not in reply}

    return collect(pending=pending, produce=produce, path=path,
                   label=f'{model_slug(arguments.model)} debate',
                   workers=arguments.workers, meter=backends.spent)


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'generate', 'debate'])
    parser.add_argument('--model', default='',
                        help='the model identifier, as written in the panel')
    parser.add_argument('--reviewer', default='',
                        help='whose judgements to review, for the debate stage')
    parser.add_argument('--method', default='',
                        help='one method, or every method if left out')
    parser.add_argument('--prompt', default='',
                        help='one prompt to show, for the check stage')
    parser.add_argument('--replicates', type=int,
                        default=GENERATION['replicates'])
    parser.add_argument('--temperature', type=float,
                        default=GENERATION['temperature'])
    parser.add_argument('--workers', type=int, default=WORKERS,
                        help='calls in flight at once')
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many, for a trial run')
    arguments = parser.parse_args()
    make_directories()

    known = {entry['id'] for entry in MODELS.values()}
    if arguments.stage in ('generate', 'debate'):
        if not arguments.model:
            raise SystemExit(f'--model is needed to {arguments.stage}')
        if arguments.model not in known:
            raise SystemExit(f'{arguments.model} is not in the panel. '
                             f'Known: {", ".join(sorted(known))}')
    if arguments.stage == 'debate' and not arguments.reviewer:
        raise SystemExit('--reviewer is needed to say whose judgements to read')

    if arguments.stage == 'generate':
        failures = run_generation(arguments)
    elif arguments.stage == 'debate':
        failures = run_debate(arguments)
    else:
        failures = run_check(arguments)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
