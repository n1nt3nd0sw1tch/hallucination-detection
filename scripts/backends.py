"""The runtimes a model can be reached through, behind one call.

Every stage that generates asks for a reply the same way and does not know
whether it came from a local runtime or an api. Adding a provider is a branch
here and an entry in the panel, not a new script.

Sampling is sent where the runtime accepts it. Some models refuse the
parameters outright and some accept them and ignore them, so a model that
cannot be decoded as the design asks is declared in the panel rather than
discovered from a failed run.
"""

import json
import random
import time
import urllib.error
import urllib.request

from settings import GENERATION, MODELS, REFUSAL

TIMEOUT = 300
RETRIES = 5
BACKOFF = 2.0
RETRY_ON = {408, 409, 429, 500, 502, 503, 504}

# Tokens seen so far, so that a long run can report what it has consumed rather
# than only how far through it is.
USAGE = {'calls': 0, 'input': 0, 'output': 0}

# Loaded runtimes, kept so that a model is read from disk once rather than once
# per call.
_LOADED = {}


# ----------------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------------

# Define function to read one field from a model's entry in the panel
def panel_entry(model_id, field, fallback=None):
    for entry in MODELS.values():
        if entry['id'] == model_id:
            return entry.get(field, fallback)
    if REFUSAL['id'] == model_id:
        return REFUSAL.get(field, fallback)
    raise ValueError(f'{model_id} is not in the panel, so its {field} is '
                     f'unknown. Add it to config/settings.yml under models.')


# Define function to read which runtime a model is reached through
def runtime_of(model_id):
    return panel_entry(model_id, 'runtime')


# Define function to say whether a model accepts the sampling the design asks
# for. A model declared as provider runs at whatever the runtime decodes at.
def takes_sampling(model_id):
    return str(panel_entry(model_id, 'sampling', '')).lower() != 'provider'


# ----------------------------------------------------------------------------
# Local runtimes
# ----------------------------------------------------------------------------

# Define function to load a model once and keep it
def load_local(model_id, runtime):
    if model_id in _LOADED:
        return _LOADED[model_id]
    if runtime == 'mlx':
        from mlx_lm import load
        _LOADED[model_id] = load(model_id)
    elif runtime == 'transformers':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _LOADED[model_id] = (AutoModelForCausalLM.from_pretrained(model_id),
                             AutoTokenizer.from_pretrained(model_id))
    else:
        raise ValueError(f'{runtime} is not a runtime this pipeline knows')
    return _LOADED[model_id]


# Define function to generate one reply through MLX on this machine
def generate_mlx(model_id, messages, max_tokens, temperature):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load_local(model_id, 'mlx')
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False)
    sampler = make_sampler(temp=temperature, top_p=GENERATION['top_p'])
    reply = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                     sampler=sampler, verbose=False)
    USAGE['calls'] += 1
    USAGE['input'] += len(tokenizer.encode(prompt))
    USAGE['output'] += len(tokenizer.encode(reply))
    return reply


# Define function to generate one reply through vLLM, for a machine with a GPU
def generate_vllm(model_id, messages, max_tokens, temperature):
    from vllm import LLM, SamplingParams

    if model_id not in _LOADED:
        _LOADED[model_id] = LLM(model=model_id)
    engine = _LOADED[model_id]
    sampling = SamplingParams(temperature=temperature,
                              top_p=GENERATION['top_p'],
                              max_tokens=max_tokens)
    output = engine.chat([messages], sampling)[0]
    USAGE['calls'] += 1
    USAGE['input'] += len(output.prompt_token_ids)
    USAGE['output'] += len(output.outputs[0].token_ids)
    return output.outputs[0].text.strip()


# ----------------------------------------------------------------------------
# Over http
# ----------------------------------------------------------------------------

# Define function to build the body one provider expects
def build_payload(model_id, messages, max_tokens, temperature):
    payload = {'model': model_id, 'max_tokens': max_tokens,
               'messages': [{'role': m['role'], 'content': m['content']}
                            for m in messages]}
    if takes_sampling(model_id):
        payload['temperature'] = temperature
        payload['top_p'] = GENERATION['top_p']
    return payload


# Define function to read the reply out of a chat completions response
def read_reply(body):
    choices = body.get('choices') or []
    if not choices:
        return ''
    return (choices[0].get('message', {}).get('content') or '').strip()


# Define function to record what one call consumed
def record_usage(body):
    usage = body.get('usage', {}) or {}
    USAGE['calls'] += 1
    USAGE['input'] += int(usage.get('prompt_tokens', 0) or 0)
    USAGE['output'] += int(usage.get('completion_tokens', 0) or 0)


# Define function to generate one reply over http, retrying what passes on its
# own and giving up loudly on what does not
def generate_api(model_id, messages, max_tokens, temperature):
    url = panel_entry(model_id, 'url')
    key = panel_entry(model_id, 'key', '')
    from utils import api_key
    token = api_key(key) if key else ''

    payload = build_payload(model_id, messages, max_tokens, temperature)
    request = urllib.request.Request(
        url, method='POST', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': f'Bearer {token}'})

    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = json.loads(response.read())
            record_usage(body)
            return read_reply(body)
        except urllib.error.HTTPError as problem:
            detail = problem.read().decode('utf-8', 'replace')[:200]
            if problem.code not in RETRY_ON or attempt == RETRIES - 1:
                raise RuntimeError(f'{model_id} returned {problem.code}: '
                                   f'{detail}') from problem
        except urllib.error.URLError:
            if attempt == RETRIES - 1:
                raise
        time.sleep(BACKOFF * (2 ** attempt) * (0.5 + random.random()))


RUNTIMES = {'mlx': generate_mlx, 'vllm': generate_vllm, 'api': generate_api}


# Define function to add whatever a model spends before answering to the cap the
# method allows for the answer, so that a reasoning model is not silenced by a
# budget sized for a one word label
def budget_for(model_id, max_tokens):
    return max_tokens + int(panel_entry(model_id, 'reasoning_headroom', 0) or 0)


# Define function to generate one reply, whichever runtime the model uses
def generate(model_id, messages, max_tokens, temperature=None):
    temperature = GENERATION['temperature'] if temperature is None else temperature
    max_tokens = budget_for(model_id, max_tokens)
    runtime = runtime_of(model_id)
    if runtime not in RUNTIMES:
        raise ValueError(f'{runtime} is not a runtime this pipeline knows. '
                         f'Choose one of {", ".join(sorted(RUNTIMES))}.')
    return RUNTIMES[runtime](model_id, messages, max_tokens, temperature)


# Define function to report what has been consumed so far, for the progress line
def spent():
    return (f"{USAGE['calls']:,} calls, {USAGE['input']:,} input and "
            f"{USAGE['output']:,} output tokens")
