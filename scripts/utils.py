"""Reading, writing, and the loop that every generating stage shares.

One collection loop rather than one per script. It resumes, it reports at a
fixed interval rather than per call, and it sends several requests at once where
the runtime allows, because a call spends nearly all of its time waiting rather
than sending.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pandas as pd

from settings import DATA_DIRS, ROOT

# How often to report, in seconds. A line per call would run to tens of
# thousands of lines and bury the failures worth seeing.
REPORT_EVERY = 60

# How many calls to have in flight at once. Raise it for an api that allows it,
# leave it at one for a local runtime, which is already using the whole machine.
WORKERS = 1

# Held while a line is written. Once calls run several at a time two workers can
# reach this together, and a response of several kilobytes is past the size at
# which an append arrives whole, so without this two half records interleave
# into one line that no later stage can parse.
_APPENDING = Lock()


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

# Define function to make every directory the pipeline writes into
def make_directories():
    for directory in DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# Define function to append one record to a JSON lines file
def append_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + '\n'
    with _APPENDING:
        with path.open('a', encoding='utf-8') as file:
            file.write(line)


# Define function to read a JSON lines file, returning an empty frame rather
# than raising when it does not exist yet
def read_lines(path):
    if not Path(path).exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8')
            .splitlines() if line.strip()]
    return pd.DataFrame(rows)


# Define function to read a table, keeping everything as written so that an
# identifier of digits is not silently turned into a number
def read_table(path, separator=','):
    return pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)


# Define function to write a table with its columns in the declared order, so
# that the file on disk matches what the settings say it holds
def write_table(frame, path, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns:
        frame = frame[columns]
    frame.to_csv(path, index=False)
    return path


# Define function to turn a model identifier into something that can be a
# filename, since most of them carry a slash
def model_slug(model_id):
    return str(model_id).split('/')[-1].replace(':', '-')


# Define function to name the file one model's responses to one method go in
def response_path(model_id, method):
    from settings import RESPONSES_DIR
    return RESPONSES_DIR / method / f'{model_slug(model_id)}.jsonl'


# Define function to read a key from the environment, or from a .env file that
# is not committed
def api_key(name):
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / '.env'
    if not env.exists():
        return ''
    for line in env.read_text().splitlines():
        if line.strip().startswith(f'{name}='):
            return line.split('=', 1)[1].strip().strip('"\'')
    return ''


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

# Define function to print a heading, so that a long run reads as stages rather
# than as one stream
def section(title):
    print(f'\n{title}')


# Define function to state how many records a file holds against how many were
# expected, which is the check worth making after every stage
def announce(label, count, expected=None):
    if expected is None:
        print(f'  {label}: {count:,}')
    else:
        mark = 'ok' if count == expected else f'expected {expected:,}'
        print(f'  {label}: {count:,} ({mark})')


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------

# Define function to list what is still to do, by removing what a file already
# holds. This is what makes every stage resumable: a run that stops part way
# loses nothing, and running it again asks only for what is missing.
def outstanding(wanted, collected, keys):
    if collected is None or collected.empty:
        return list(wanted)
    have = {tuple(str(row[key]) for key in keys)
            for _, row in collected.iterrows()}
    return [item for item in wanted
            if tuple(str(item[key]) for key in keys) not in have]


# Define function to work through a list of items, writing each result as it
# arrives so that an interruption costs only the call in flight
def collect(pending, produce, path, label='', workers=1, meter=None):
    started, spoke, failures, index = time.time(), time.time(), 0, 0

    def attempt(item):
        try:
            return produce(item), ''
        except Exception as problem:
            # one failure should not end an overnight run, so it is recorded and
            # the pass continues; running again retries whatever failed
            return {}, f'{type(problem).__name__}: {problem}'

    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        step = workers if pool else 1
        for start in range(0, len(pending), step):
            group = pending[start:start + step]
            if pool:
                outcomes = list(pool.map(attempt, group))
            else:
                outcomes = [attempt(group[0])]
            for item, (result, error) in zip(group, outcomes):
                append_line(path, {**item, **result, 'error': error})
                failures += 1 if error else 0
            index += len(group)

            if time.time() - spoke >= REPORT_EVERY or index == len(pending):
                spoke = time.time()
                rate = index / max(time.time() - started, 1)
                line = (f'  {label + "  " if label else ""}'
                        f'{index:,} of {len(pending):,}, '
                        f'{rate * 3600:,.0f} an hour, '
                        f'{(len(pending) - index) / rate / 3600:.1f} hours left, '
                        f'{failures} failed')
                if meter is not None:
                    measured = meter()
                    if measured:
                        line += f'\n     {measured}'
                print(line)
    finally:
        if pool:
            pool.shutdown()
    return failures
