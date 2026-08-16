"""Everything the pipeline reads from configuration, resolved once.

No script defines a path, a column list, or a model identifier of its own. They
are here, derived from config/settings.yml, so that a run is described by that
file rather than by whichever constant happened to be uncommented at the top of
whichever script was last edited.

The assertions at the bottom fail at import rather than midway through a run,
because a configuration that cannot produce a coherent experiment should not be
discovered after an hour of generation.
"""

from pathlib import Path

import yaml

# ----------------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'config' / 'settings.yml'
SETTINGS = yaml.safe_load(CONFIG_PATH.read_text())

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

DATA_DIR = ROOT / SETTINGS['paths']['data']
RESULTS_DIR = ROOT / SETTINGS['paths']['results']
FIGURES_DIR = ROOT / SETTINGS['paths']['figures']

HALOMI_DIR = DATA_DIR / 'halomi'
BENCHMARK_DIR = DATA_DIR / 'benchmark'
RESPONSES_DIR = RESULTS_DIR / 'responses'
JUDGEMENTS_DIR = RESULTS_DIR / 'judgements'

HALOMI_PATH = HALOMI_DIR / SETTINGS['dataset']['file']
ITEMS_PATH = BENCHMARK_DIR / 'items.csv'
PROMPTS_PATH = BENCHMARK_DIR / 'prompts.csv'
SCORES_PATH = RESULTS_DIR / 'scores.csv'

DATA_DIRS = [HALOMI_DIR, BENCHMARK_DIR, RESULTS_DIR, RESPONSES_DIR,
             JUDGEMENTS_DIR, FIGURES_DIR]

# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

GENERATION = SETTINGS['generation']
MODELS = SETTINGS['models']
REFUSAL = SETTINGS['refusal']
METHODS = SETTINGS['methods']
DATASET = SETTINGS['dataset']

# ----------------------------------------------------------------------------
# The label space
# ----------------------------------------------------------------------------

# The task is binary. The four grades HalOmi carries are kept alongside the
# label so that a severity breakdown is possible without re-reading the source.
HALLUCINATION = 'Hallucination'
NO_HALLUCINATION = 'No hallucination'
LABELS = [NO_HALLUCINATION, HALLUCINATION]

ITEMS_PER_DIRECTION = DATASET.get('items_per_direction', 0)
SAMPLE_SEED = DATASET.get('sample_seed', 0)

POSITIVE_GRADES = DATASET['positive_grades']
NEGATIVE_GRADE = DATASET['negative_grade']
GRADES = [NEGATIVE_GRADE] + POSITIVE_GRADES

# ----------------------------------------------------------------------------
# Directions
# ----------------------------------------------------------------------------

LANGUAGES = SETTINGS['languages']
ENGLISH = 'eng_Latn'


# Define function to name one language in the form a prompt or an axis can
# carry, so that neither has to parse a language code
def language_name(code):
    return {**LANGUAGES, ENGLISH: 'English'}.get(code, code)


# Define function to name a direction the same way
def direction_name(source, target):
    return f'{language_name(source)} to {language_name(target)}'


# Define function to list every direction the configuration asks for. Pairs are
# formed against English in both ways, plus whichever pairs are named directly.
def directions():
    found = []
    for code in LANGUAGES:
        found.append((code, ENGLISH))
        found.append((ENGLISH, code))
    for source, target in SETTINGS.get('extra_pairs', []):
        found.append((source, target))
    return found


DIRECTIONS = directions()

# ----------------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------------

# One item is one source and translation pair with its gold label.
ITEM_COLUMNS = ['item_id', 'source_language', 'target_language', 'direction',
                'grade', 'answer', 'source_text', 'target_text']

# One prompt is one item asked one way. shots is zero except for few shot, and
# the rendered text is stored so that what was sent can be read back from one
# file rather than reconstructed.
PROMPT_COLUMNS = ['prompt_id', 'item_id', 'method', 'shots', 'direction',
                  'answer', 'prompt']

# One response is one prompt answered once by one model. The flags sit before
# the text, so that scanning a file shows what happened to a call before it
# shows what came back.
RESPONSE_COLUMNS = ['prompt_id', 'model', 'replicate', 'runtime', 'temperature',
                    'error', 'refusal', 'truncated', 'response']

# One judgement is one response read into a label.
JUDGEMENT_COLUMNS = ['prompt_id', 'model', 'replicate', 'method', 'shots',
                     'direction', 'answer', 'observed', 'parsed', 'refusal']

# ----------------------------------------------------------------------------
# Identifiers
# ----------------------------------------------------------------------------

# Define function to name one item, stably, from the direction and its position
def make_item_id(source, target, index):
    return f'{source[:3]}{target[:3]}-{index:04d}'.lower()


# Define function to name one prompt, so that a method and a shot count are
# readable from the identifier without a join
def make_prompt_id(item_id, method, shots=0):
    return f'{item_id}-{method}' + (f'{shots}' if shots else '')


# ----------------------------------------------------------------------------
# What a coherent configuration looks like
# ----------------------------------------------------------------------------

assert GENERATION['replicates'] >= 1, 'at least one replicate is needed'
assert 0 <= GENERATION['temperature'] <= 2, 'temperature is out of range'
assert 0 < GENERATION['top_p'] <= 1, 'top_p is out of range'
assert MODELS, 'the panel is empty'
assert METHODS, 'no methods are configured'
assert NEGATIVE_GRADE not in POSITIVE_GRADES, \
    'a grade cannot be both positive and negative'
assert len(DIRECTIONS) == len(set(DIRECTIONS)), 'a direction is listed twice'
for name, method in METHODS.items():
    assert 'max_tokens' in method, f'{name} has no token cap'
    assert 'shots' in method, f'{name} does not say how many examples to give'
