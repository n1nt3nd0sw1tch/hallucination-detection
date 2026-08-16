"""The five ways of asking the same question.

Each method is one framing of a single task: given a source sentence and its
translation, does the translation introduce content the source does not support.
They share one builder and differ only in what it is told, so a sixth method is
a template here rather than a new eight hundred line script.

The instruction that fixes the output format is shared by every method, because
a difference in how the answer is requested would confound a difference in how
the question is asked.
"""

from settings import HALLUCINATION, NO_HALLUCINATION, language_name

# Every method ends with this, so that parsing is the same everywhere and a
# method is not advantaged by being asked for a tidier answer.
ANSWER_FORMAT = (f'End your reply with exactly one line:\n'
                 f'Label: {HALLUCINATION}\n'
                 f'or\n'
                 f'Label: {NO_HALLUCINATION}')

SYSTEM = ('You judge machine translations. A hallucination is content in the '
          'translation that the source does not support. Omission alone is not '
          'a hallucination.')

# ----------------------------------------------------------------------------
# The methods
# ----------------------------------------------------------------------------

TEMPLATES = {
    'baseline': (
        'Source ({source}): {source_text}\n'
        'Translation ({target}): {target_text}\n\n'
        'Does the translation contain hallucinated content?\n\n'
        f'{ANSWER_FORMAT}'),

    'fewshot': (
        '{examples}'
        'Source ({source}): {source_text}\n'
        'Translation ({target}): {target_text}\n\n'
        'Does the translation contain hallucinated content?\n\n'
        f'{ANSWER_FORMAT}'),

    # Chain of translation: render the non-English side in English first, so
    # that the judgement is made on text the model reads most reliably.
    'translate': (
        'Source ({source}): {source_text}\n'
        'Translation ({target}): {target_text}\n\n'
        'First, translate both the source and the translation into English.\n'
        'Then compare them and decide whether the translation introduces '
        'content the source does not support.\n\n'
        f'{ANSWER_FORMAT}'),

    # Span level: name the offending text before labelling, so that the label
    # follows from something stated rather than from an impression.
    'span': (
        'Source ({source}): {source_text}\n'
        'Translation ({target}): {target_text}\n\n'
        'List every span of the translation that is not supported by the '
        'source, one per line, prefixed with "Span:". If there are none, write '
        '"Span: none".\n'
        'Then give your label.\n\n'
        f'{ANSWER_FORMAT}'),

    # Review: a second model reads the first one's judgement and its reasoning,
    # and either agrees or does not.
    'debate': (
        'Source ({source}): {source_text}\n'
        'Translation ({target}): {target_text}\n\n'
        'Another reviewer judged this as: {other_label}\n'
        'Their reasoning: {other_reasoning}\n\n'
        'Do you agree? Judge it yourself, saying briefly where you differ if '
        'you do.\n\n'
        f'{ANSWER_FORMAT}'),
}

EXAMPLE = ('Example\n'
           'Source ({source}): {source_text}\n'
           'Translation ({target}): {target_text}\n'
           'Label: {answer}\n\n')


# ----------------------------------------------------------------------------
# Building one
# ----------------------------------------------------------------------------

# Define function to render the worked examples that precede a few shot prompt
def render_examples(examples):
    return ''.join(EXAMPLE.format(
        source=language_name(row['source_language']),
        target=language_name(row['target_language']),
        source_text=row['source_text'], target_text=row['target_text'],
        answer=row['answer']) for row in examples)


# Define function to build one prompt. Every method goes through here, so the
# only difference between them is the template and what it is given.
def build(method, item, examples=(), other=None):
    if method not in TEMPLATES:
        raise ValueError(f'{method} is not a method. Choose one of '
                         f'{", ".join(sorted(TEMPLATES))}.')
    fields = {
        'source': language_name(item['source_language']),
        'target': language_name(item['target_language']),
        'source_text': item['source_text'],
        'target_text': item['target_text'],
        'examples': render_examples(examples) if examples else '',
        'other_label': (other or {}).get('label', ''),
        'other_reasoning': (other or {}).get('reasoning', ''),
    }
    return TEMPLATES[method].format(**fields)


# Define function to wrap a prompt as the messages a chat model expects
def as_messages(prompt):
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': prompt}]
