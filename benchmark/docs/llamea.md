# LLaMEA for the AEMEC benchmark

The `llamea` algorithm uses the actual [LLaMEA project](https://github.com/XAI-liacs/LLaMEA),
pinned to version **1.4.2**, to generate and evolve Python proposal functions.
It solves the same three-input, two-objective problem as the other algorithms.
**Every design is evaluated by the AEMEC CFD solver.** No proxy responses are
used for training, algorithm fitness, Pareto fronts or hypervolume.

## What is optimized?

There are two levels:

1. A generated proposal function chooses membrane thickness, water inlet
   temperature and anode PTL porosity. CFD returns cell voltage and gas crossover.
   Both physical objectives are minimized and their trade-off is retained.
2. LLaMEA generates/mutates those proposal functions and selects the one whose
   CFD observations give the greatest Pareto hypervolume. Hypervolume measures
   the quality of an entire objective set; it is not a weighted voltage/crossover
   response assigned to an individual design.

The adapter uses the project's `initialize_single()`, `evolve_solution()`,
`Solution.set_scores()`, `selection()` and `update_best()` methods. It steps a
**(1+1) elitist** evolution through the benchmark's ask/tell loop instead of calling
LLaMEA's blocking `run()` method. `evaluate_population=True` defers scoring to
our runner; no parallel evaluator or hidden CFD callback exists inside LLaMEA.
The model prompt, parser, mutation prompt and parent-selection behavior come
from the project, with an AEMEC-specific task/interface prompt.

## Budget and comparison protocol

For each seed:

- All benchmark algorithms receive exactly the same `--initial` CFD attempts.
- Each generated candidate gets those initial observations and up to
  `--llamea-candidate-budget` **additional** CFD attempts. Candidate histories
  include their own new responses only, so later candidates do not receive
  extra training data from earlier candidates.
- Candidate fitness is the hypervolume of its initial-plus-own successful
  responses, using the same fixed normalization/reference as the benchmark.
  The shared initial archive makes these scores directly comparable within a seed.
- At most `--llamea-candidates` candidates are generated. The effective count is
  also capped by how many equal candidate allowances fit in the remaining budget.
- After evolution, the best valid candidate uses all CFD observations collected
  by this LLaMEA run for the remaining attempts. New outcomes are private to this
  algorithm; competing benchmark methods never receive them.
- The reported Pareto archive includes all successful LLaMEA CFD evaluations,
  including evaluations from candidate code that was subsequently discarded.

For example, with `--budget 64 --initial 16 --llamea-candidates 8
--llamea-candidate-budget 4`, the allocation is **16 shared + 32 candidate + 16
winner-deployment attempts = 64**. LLM generation and mutation have additional
provider cost and runtime, but do not grant extra CFD calls. LLM time and worker
execution are included in `optimizer_s`.

If fewer adaptive attempts remain than one candidate allowance, the allowance is
shortened and only one candidate is evaluated. A small pilot can therefore test
integration without demonstrating meaningful evolution. CFD failures and
repeat designs count toward the allowance and total budget. Failed responses
are omitted from the objective archive, never replaced by penalty responses.

This compares **online algorithm development plus deployment** against the
other optimizers. It is not a generalization test of a frozen optimizer trained
on a separate task set. A seed controls local randomness, but model-service
responses need not reproduce exactly. Save the generated source and replies
when reporting results.

## Install separately from the older environment

The published LLaMEA 1.4.2 package requires **Python 3.11 or newer**. From the
repository root, create a new environment; preserve your existing environment:

```bash
python3.11 -m venv benchmark/.venv-llamea
source benchmark/.venv-llamea/bin/activate
python -m pip install -r benchmark/requirements-llamea.txt
```

Use `python3.12` instead of `python3.11` if that is the installed interpreter.
The existing four baselines and `ela_proxy` still work without this optional
package. Copy the full updated `benchmark/` folder and retain sibling `opt/`
on the CFD server.

Choose a provider and a model that you already have access to. No model name,
API credential, LLM response or optimizer performance is assumed by the code.

### Local Ollama

Have Ollama running with a downloaded code-capable model. Set `LLAMEA_MODEL` to
its exact installed model name. For example, inspect your installation with
`ollama list`, then set the variable to the chosen name. This environment
variable is only a convenience for the commands below; the CLI takes `--llamea-model`.

With OpenFOAM and `openFuelCell` available, run a 12-attempt pilot:

```bash
python benchmark/run_benchmark.py \
  --algorithms llamea \
  --llamea-provider ollama --llamea-model "$LLAMEA_MODEL" \
  --llamea-candidates 2 --llamea-candidate-budget 2 \
  --budget 12 --initial 4 --seeds 42 \
  --output benchmark/results/llamea_cfd_pilot
```

This uses four shared attempts, two candidates with two attempts each, and four
winner-deployment attempts. It makes at most two LLM requests and exactly twelve
CFD design attempts if it completes. The default Ollama endpoint is
`http://127.0.0.1:11434`; use `--llamea-base-url` to point to your Ollama server.

### OpenAI or a compatible Chat Completions endpoint

Set `OPENAI_API_KEY` in your environment and set `LLAMEA_MODEL` to a model available
to your account that supports Chat Completions. Then use the same command with
`--llamea-provider openai`. An optional `--llamea-base-url` selects your compatible
endpoint; `--llamea-api-key-env` selects an alternative environment-variable
**name**, never a literal key.

The adapter uses LLaMEA's code parser and a bounded API transport. It sends
`max_completion_tokens`, which includes reasoning and visible-output tokens;
allow sufficient headroom for complete Python code. Parameter support depends
on the chosen model/endpoint. See the [official API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Only task instructions, generated source, aggregate CFD fitness and error
feedback are sent to the configured model. Raw CFD files, repository contents
and numerical observation tables are not included in the prompts. The proposal
worker receives numerical history locally, without credentials.

### Compare against the existing baselines

Use your chosen provider/model in this command (Ollama is shown):

```bash
python benchmark/run_benchmark.py \
  --algorithms gp tpe nsga2 cmaes llamea \
  --llamea-provider ollama --llamea-model "$LLAMEA_MODEL" \
  --budget 64 --initial 16 --seeds 42 43 44 45 46 \
  --llamea-candidates 8 --llamea-candidate-budget 4 \
  --output benchmark/results/llamea_comparison_cfd
```

This comparison makes 1,280 actual CFD design attempts with shared initialization.
Add `ela_proxy` and its model-file options to include the sixth available method;
then the count is 1,520. The four original baselines remain the default, so LLaMEA
is only used when requested with `--algorithms llamea`.

Append `--dry-run` to preview configuration and attempt counts without CFD, LLM
requests or output files. As for other algorithms, a dry run skips runtime and
credential checks. A real run checks the optional package and required key before
CFD initialization, but does not contact the provider as a preflight check.

## Generated-code execution and errors

A candidate defines:

```python
def propose(x, y, attempts, reference, rng):
    # x: successful normalized designs, shape (n, 3)
    # y: corresponding fixed-normalized objectives, shape (n, 2)
    # attempts: all attempted designs, including failed CFD runs
    # reference: fixed normalized hypervolume reference, shape (2,)
    # rng: seeded NumPy Generator
    return rng.uniform(0.0, 1.0, 3)
```

The function is reconstructed for every call; its state is the supplied history
and RNG. Candidate comparisons use the same proposal seed schedule. During
winner deployment, the history grows with all observations from this LLaMEA run.

Generated code runs in a disposable subprocess with a numerical AST/API
allowlist, a clean environment, a temporary working directory, and a wall-time
limit. Linux workers also have a memory limit. The worker has no CFD callback.
NumPy/math numerical operations are supported; imports of other libraries,
file/network operations and introspection are rejected. The allowlist is in
`aemec_benchmark/llamea_worker.py`; it is intentionally a limited interface,
not a general-purpose security sandbox for arbitrary hostile Python programs.

`--llamea-proposal-timeout` defaults to 10 seconds per proposal;
`--llamea-request-timeout` defaults to 120 seconds per LLM request;
`--llamea-max-tokens` defaults to 4096. Provider retries are disabled.

Malformed code, invalid points and proposal timeouts reject that optimizer.
The rejection has an **algorithm-level score of -1**, below valid nonnegative
hypervolume. This score is never recorded as cell voltage or crossover. The
rejection consumes a candidate-generation slot but does not fabricate or charge
a CFD evaluation. If valid code was evaluated before the error, those CFD
attempts still count and remain in the reported archive. Any unused attempts
are available for the selected valid winner. If no valid candidate is found,
the run stops with a clear error and preserves completed results; it does not
silently switch to random search. A failing deployed winner similarly stops.

## Saved artifacts

In addition to the common evaluation/history/Pareto files, each
`seed_<seed>/llamea/code_evolution/` directory contains:

- `protocol.json`: effective candidate budget and method settings.
- `prompt.json`: AEMEC task/interface prompt and example.
- `llm_001.json`, etc.: request prompts, replies, token usage and sanitized errors.
- `candidate_001.py`, etc.: exact generated source.
- `candidates.json`: CFD attempt counts, fitness, errors and source hashes.
- `best_optimizer.py` and `best_optimizer.json`: best valid candidate and its score.

The main `details.json` maps each CFD proposal to candidate index, source hash,
phase and numerical seed. API keys are not put in these files. Upstream pickle
logging is disabled because it can serialize model-client credentials. These
artifacts support inspection, not exact automatic resume.

## Verification

After installing the optional requirements, run from the repository root:

```bash
PYTHONPATH=opt:benchmark:. MPLCONFIGDIR=/tmp/aemec-matplotlib \
  python -m unittest discover -s benchmark/tests -v
```

The integration tests use **real LLaMEA generation/mutation/selection** with
scripted model replies, execute the numerical worker, and mock the CFD adapter.
They test history isolation, hypervolume scoring, candidate rejection, failure
accounting, winner deployment, exact budgets, credential handling and timeouts.
They do not measure the quality of live LLM-generated optimizers or validate an
actual CFD solver run. On older environments without LLaMEA, native integration
tests are explicitly skipped while the numerical-worker/config tests still run.
