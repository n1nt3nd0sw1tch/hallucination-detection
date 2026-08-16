# Multilingual Hallucination Detection

Detecting hallucinations in machine translation with LLM judges, on the
[HalOmi](https://github.com/facebookresearch/stopes/tree/main/demo/halomi)
dataset. A hallucination is content the translation introduces that the source
does not support; omission alone is not one.

The question is whether a judge's reliability depends on the languages it is
judging, and whether prompting it differently closes the gap.

## Design

**Five methods**, each a different way of asking the same question. Zero shot
binary, few shot at one, two and three examples, chain of translation, span
level chain of thought, and multi agent review.

**Eighteen directions**, eight languages against English both ways, plus the two
Spanish and Yoruba pairs, which are the only ones with no English anchor at all.

**Three replicates** per prompt, sampled stochastically at temperature 1.0 and
top_p 1.0 rather than replayed from a fixed seed. Three replicates separate
variation due to resampling from variation due to the method, and the share of
items where all three agree is reported as a stability measure in its own right.

**Three views** of every result, because dropping a case changes what a score
means:

| View | What it counts |
|---|---|
| `all` | every reply, an unreadable one counted as wrong |
| `parsed` | replies that gave a label |
| `judged` | replies that gave a label and were not refusals |

A refusal is not a wrong answer, it is an absent one. Counting it as wrong
understates a model that declines where another guesses.

## Layout

```
config/settings.yml      every setting, in one file
data/halomi/             the dataset
data/benchmark/          items.csv and prompts.csv, built from it
scripts/
  settings.py            configuration, paths, columns, assertions
  utils.py               reading, writing, and the collection loop
  backends.py            the runtimes a model is reached through
  prompts.py             the five methods as templates
  build.py               dataset to items and prompts
  run.py                 generate and check
  evaluate.py            parse, detect refusals, score
  figures.py             every figure, from scores.csv
notebooks/               exploration, one per stage
results/
  responses/<method>/<model>.jsonl
  judgements/<method>/<model>.csv
  scores.csv
```

Nothing is configured by editing a script. A run is described by
`config/settings.yml` and the command that started it.

## Running it

```bash
pip install -r requirements.txt

python scripts/build.py                                   # items and prompts
python scripts/run.py check --method span                 # read one prompt
python scripts/run.py generate --model <id> --limit 20    # a trial
python scripts/run.py generate --model <id>               # the pass
python scripts/run.py debate --model <a> --reviewer <b>   # the second round
python scripts/evaluate.py
python scripts/figures.py
```

Every stage resumes. A run that stops part way loses nothing, and running it
again asks only for what is missing.

`check` prints one rendered prompt in full before anything is spent, which is
worth doing after any change to a template.

## Scale

The full dataset across five methods at three replicates is over fifty thousand
calls a model, which is days on a laptop runtime. `items_per_direction` in the
configuration caps how many items each direction contributes, sampled with the
label balance of that direction preserved and a fixed seed. It defaults to 40,
giving 4,320 prompts and 12,960 calls a model. Set it to 0 for everything.

## Attribution

This is a refactor of
[mk1m/multilingual-hallucination-detect](https://github.com/mk1m/multilingual-hallucination-detect),
a group coursework project by Minoo Kim, Arina Bakulina, Harvey Yang, Frank H
and Erin Ju. The original is MIT licensed and that licence, with its copyright
notice, is retained here.

The dataset is HalOmi, from Dale et al., *HalOmi: A Manually Annotated Benchmark
for Multilingual Hallucination and Omission Detection in Machine Translation*,
EMNLP 2023.

## What changed in the refactor

The original had five evaluation scripts of 380 to 822 lines, ten functions
named `main`, and `load_halomi_data`, `parse_prediction`, `build_chat_prompt`
and `clean_model_output` each written five times. Configuration lived in
commented out blocks at the top of each script, so which languages a result
covered depended on which lines happened to be uncommented when it ran.

It is now eight modules with one definition of each thing. Prompts are rendered
once into a file, so what was sent to a model can be read back rather than
reconstructed. Generation resumes. Replicates, sampling and refusal handling are
new.
