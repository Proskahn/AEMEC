"""A stepped LLaMEA adapter: evolve proposal code, assess it only with CFD.

LLaMEA's public generation/mutation/selection methods are used directly. We
step them between ask/tell calls instead of calling its blocking run() loop,
so the benchmark remains the sole owner of CFD calls and the attempt budget.
No LLM request or generated program runs at import time.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import numpy as np

from .llamea_worker import validate_code
from .metrics import hypervolume_2d

LLAMEA_VERSION = '1.4.2'
PROJECT_URL = 'https://github.com/XAI-liacs/LLaMEA'
EXAMPLE = '''import numpy as np

def propose(x, y, attempts, reference, rng):
    if len(x) < 4 or rng.uniform() < 0.3:
        return rng.uniform(0.0, 1.0, 3)
    weights = rng.uniform(0.1, 1.0, 2)
    weights = weights / np.sum(weights)
    parent = x[np.argmin(y @ weights)]
    return np.clip(parent + rng.normal(0.0, 0.15, 3), 0.0, 1.0)
'''


@dataclass(frozen=True)
class LLaMEASettings:
    """Separate code-generation controls from the common CFD attempt budget."""
    model: str
    provider: str = 'ollama'
    candidates: int = 8
    candidate_budget: int = 4
    proposal_timeout: float = 10.
    request_timeout: float = 120.
    max_tokens: int = 4096
    api_key_env: str = 'OPENAI_API_KEY'
    base_url: str | None = None

    def validate(self):
        if not self.model or not self.model.strip():
            raise ValueError('llamea requires --llamea-model (a model available from your provider)')
        if self.provider not in ('ollama', 'openai'):
            raise ValueError('LLaMEA provider must be ollama or openai')
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (self.candidates, self.candidate_budget, self.max_tokens)):
            raise ValueError('LLaMEA candidates, candidate budget and max tokens must be positive integers')
        if any(not np.isfinite(v) or v <= 0 for v in (self.proposal_timeout, self.request_timeout)):
            raise ValueError('LLaMEA timeouts must be finite and positive')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', self.api_key_env):
            raise ValueError('LLaMEA API key must be specified by an environment variable name')
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError('Use an HTTP(S) base URL without credentials, query or fragment')


def require_llamea(settings=None):
    """Check optional dependencies/credentials before starting expensive CFD."""
    if sys.version_info < (3, 11):
        raise ValueError('LLaMEA 1.4.2 requires Python 3.11+. Create a new environment and install '
                         'benchmark/requirements-llamea.txt; other algorithms can use the existing environment')
    try:
        if importlib.metadata.version('llamea') != LLAMEA_VERSION:
            raise ImportError('Unsupported LLaMEA version')
        import llamea
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ValueError('Install the optional integration with: python -m pip install '
                         '-r benchmark/requirements-llamea.txt') from exc
    if settings is not None:
        settings.validate()
        if settings.provider == 'openai' and not os.environ.get(settings.api_key_env):
            raise ValueError(f'Set {settings.api_key_env} in the environment before using the OpenAI provider')
    return llamea


def write_json(path, data):
    """Write audit files atomically without serializing clients or API keys."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def make_llm(module, settings, folder):
    """Use LLaMEA's parser with bounded provider requests and plain JSON logs.

    Logging pickled LLaMEA clients can serialize credentials, so upstream logging
    is disabled. Our audit contains only prompts, replies, usage, and settings.
    No request retries are hidden inside the candidate count.
    """
    if settings.provider == 'openai':
        from openai import OpenAI
        client = OpenAI(api_key=os.environ[settings.api_key_env],
                        base_url=settings.base_url, timeout=settings.request_timeout, max_retries=0)
    else:
        from ollama import Client
        client = Client(host=settings.base_url or 'http://127.0.0.1:11434', timeout=settings.request_timeout)

    class AuditedLLM(module.LLM):
        def __init__(self):
            super().__init__(api_key='', model=settings.model)
            self.calls = 0

        def query(self, session):
            self.calls += 1
            record = {'request': self.calls, 'provider': settings.provider,
                      'model': settings.model, 'messages': session}
            try:
                if settings.provider == 'openai':
                    reply = client.chat.completions.create(
                        model=settings.model, messages=session, max_completion_tokens=settings.max_tokens)
                    content = reply.choices[0].message.content
                    record['usage'] = reply.usage.model_dump() if reply.usage else None
                else:
                    reply = client.chat(model=settings.model, messages=session,
                                        options={'num_predict': settings.max_tokens})
                    content = reply.message.content
                    record['usage'] = {k: getattr(reply, k, None) for k in ('prompt_eval_count', 'eval_count')}
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Empty model response')
                record['response'] = content
                return content
            except Exception as exc:
                # Provider errors can contain HTTP details. Keep secrets out of
                # LLaMEA's next prompt and the audit by logging only the type.
                message = f'{settings.provider} request failed ({type(exc).__name__})'
                record['error'] = message
                raise RuntimeError(message) from None
            finally:
                write_json(folder/f'llm_{self.calls:03d}.json', record)

    return AuditedLLM()


def run_proposal(code, x, y, attempts, reference, seed, timeout):
    """Return a validated design; generated code receives neither solver nor key."""
    validate_code(code)
    payload = {'code': code, 'x': x, 'y': y, 'attempts': attempts,
               'reference': np.asarray(reference).tolist(), 'seed': int(seed)}
    # A minimal environment deliberately excludes API credentials and the
    # repository's Python path. -I also ignores user site-packages/PYTHONPATH.
    environment = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    environment['PATH'] = os.defpath
    worker = Path(__file__).with_name('llamea_worker.py')
    with tempfile.TemporaryDirectory(prefix='aemec-proposal-') as temp:
        try:
            result = subprocess.run([sys.executable, '-I', str(worker)],
                                    input=json.dumps(payload, allow_nan=False), text=True,
                                    capture_output=True, cwd=temp, env=environment, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f'Generated proposal exceeded {timeout:g} seconds') from exc
    try:
        response = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Generated proposal process failed (exit {result.returncode})') from exc
    if result.returncode or 'error' in response:
        raise ValueError(response.get('error', 'Generated proposal process failed'))
    point = np.asarray(response['point'], dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1):
        raise ValueError('Generated proposal must be a finite point in [0, 1]^3')
    return point


def protocol(settings, initial, budget):
    """Describe how algorithm development and deployment share the CFD budget."""
    steps = min(settings.candidate_budget, budget-initial)
    return {'implementation': 'LLaMEA stepped (1+1) elitist algorithm evolution',
            'project': PROJECT_URL, 'version': LLAMEA_VERSION, 'settings': asdict(settings),
            'candidate_budget_effective': steps,
            'candidate_limit_effective': min(settings.candidates, (budget-initial)//steps),
            'candidate_fitness': 'maximize fixed-reference HV of shared initial plus candidate CFD responses',
            'candidate_history': 'shared initialization plus own observations only',
            'deployment': 'best valid candidate uses all observations from this LLaMEA run',
            'cost': 'all candidate and deployment CFD attempts count toward the same total budget',
            'code_errors': 'no CFD response fabricated; reject candidate; no silent optimizer fallback',
            'llm_data': 'task prompt, generated code, aggregate CFD fitness and error feedback',
            'reproducibility': 'local RNGs seeded; remote model outputs not guaranteed reproducible; source/replies saved'}


class LLaMEAOptimizer:
    """Translate LLaMEA algorithm evolution into the runner's ask/tell protocol.

    Candidate fitness uses a common initial archive and equal trial lengths.
    Candidates cannot train on competitors' new CFD results. This avoids giving
    later code candidates a larger history merely because they were generated
    later. After code evolution, the winner uses the union of this run's data.
    All of these experiments count toward this optimizer's single CFD budget.
    """
    def __init__(self, seed, reference, initial, budget, settings, folder):
        settings.validate()
        self.settings, self.seed = settings, seed
        self.reference = np.asarray(reference, dtype=float)
        self.initial, self.budget = initial, budget
        self.steps = min(settings.candidate_budget, budget-initial)
        self.limit = min(settings.candidates, (budget-initial)//self.steps)
        self.folder = Path(folder)/'code_evolution'
        self.folder.mkdir()
        self.module = require_llamea(settings)
        self.engine = None
        self.np_state = np.random.RandomState(seed).get_state()
        self.py_state = random.Random(seed).getstate()
        self.attempts, self.x, self.y = [], [], []
        self.shared_attempts, self.shared_x, self.shared_y = [], [], []
        self.generated = 0
        self.active, self.best, self.pending = None, None, None
        self.records = []
        write_json(self.folder/'protocol.json', protocol(settings, initial, budget))

    @contextmanager
    def engine_rng(self):
        """Isolate upstream's global RNG use in this serial benchmark adapter."""
        old_np, old_py = np.random.get_state(), random.getstate()
        np.random.set_state(self.np_state)
        random.setstate(self.py_state)
        try:
            with redirect_stdout(io.StringIO()):
                yield
        finally:
            self.np_state, self.py_state = np.random.get_state(), random.getstate()
            np.random.set_state(old_np)
            random.setstate(old_py)

    def start_engine(self):
        task = f'''Evolve a numerical proposal function for three inputs in [0,1]^3,
ordered membrane thickness, water inlet temperature, and anode PTL porosity.
Two CFD objectives (cell voltage and gas crossover) are both minimized. Values
use a fixed affine normalization and may lie outside [0,1]. The reference is
{self.reference.tolist()}. Maximize fixed-reference Pareto hypervolume after
{self.steps} CFD attempts, starting from the same {self.initial} shared attempts.
Return exactly a function propose(x, y, attempts, reference, rng) -> length-3
finite array in [0,1]. x and y are arrays of successful designs/responses,
shape (n,3) and (n,2), possibly empty. attempts has ALL attempted designs,
including failures. rng is a seeded numpy Generator. Use the supplied history
and rng; the function is reconstructed in a fresh process at every proposal.
Only import numpy as np and/or math. No classes, decorators, from imports,
file/network access, external packages, introspection, or module-level execution.
Put constants inside functions. Use ordinary NumPy array operations, linalg,
and rng.uniform/normal/choice/integers/permutation; do not create a new RNG.
The function receives no objective callback. Return one point; the host alone
runs CFD. Aim to improve both objectives' trade-off and avoid repeat designs.
Errors/timeouts reject the candidate. Failed CFD attempts count toward its
allowance and supply no invented response. Example of the required interface:
'''
        llm = make_llm(self.module, self.settings, self.folder)
        with self.engine_rng():
            self.engine = self.module.LLaMEA(
                f=None, llm=llm, n_parents=1, n_offspring=1, elitism=True,
                task_prompt=task, example_prompt='\n```python\n'+EXAMPLE+'```\n',
                output_format_prompt='Return a short # Description: and one fenced python block containing the function.\n',
                evaluate_population=True, multi_objective=False, minimization=False,
                adaptive_prompt=False, HPO=False, log=False, max_workers=1,
                budget=self.limit)
        write_json(self.folder/'prompt.json', {'task': task, 'example': EXAMPLE})

    def new_candidate(self):
        if self.engine is None:
            self.start_engine()
        self.engine.generation = self.generated
        with self.engine_rng():
            solution = (self.engine.initialize_single() if not self.engine.population
                        else self.engine.evolve_solution(self.engine.population[0]))
        self.generated += 1
        # Use our numeric index for filenames; model-generated names are data.
        filename = f'candidate_{self.generated:03d}.py'
        code = solution.code or ''
        (self.folder/filename).write_text(code)
        self.active = {'solution': solution, 'index': self.generated, 'source': filename,
                       'sha256': hashlib.sha256(code.encode()).hexdigest(), 'steps': 0,
                       'x': list(self.shared_x), 'y': list(self.shared_y),
                       'attempts': list(self.shared_attempts)}
        try:
            if solution.error:
                raise ValueError(solution.error[:1000])
            validate_code(code)
        except (ValueError, SyntaxError) as exc:
            self.finish_candidate(str(exc))

    def finish_candidate(self, error=None):
        candidate = self.active
        solution = candidate['solution']
        # -1 is an ALGORITHM fitness for rejected code, never a physical response.
        # Valid HV is nonnegative. CFD failures simply leave the archive unchanged.
        fitness = -1. if error else hypervolume_2d(candidate['y'], self.reference)
        feedback = (f'CFD attempts: {candidate["steps"]}/{self.steps}. '
                    f'Pareto hypervolume including the common initial archive: {fitness:.12g}. '
                    + (f'Proposal rejected: {error}' if error else 'Higher is better. Both CFD objectives are minimized.'))
        solution.set_scores(float(fitness), feedback=feedback, error=ValueError(error) if error else None)
        record = {k: candidate[k] for k in ('index', 'source', 'sha256', 'steps')}
        record.update({'fitness': float(fitness), 'feedback': feedback, 'error': error,
                       'successful_new': len(candidate['y'])-len(self.shared_y)})
        self.records.append(record)
        with self.engine_rng():
            self.engine.run_history.append(solution)
            self.engine.population = self.engine.selection(self.engine.population, [solution])
            self.engine.update_best()
        if not error and (self.best is None or fitness > self.best['fitness']):
            self.best = {**record, 'code': solution.code}
            (self.folder/'best_optimizer.py').write_text(solution.code)
            write_json(self.folder/'best_optimizer.json', record)
        write_json(self.folder/'candidates.json', self.records)
        self.active = None

    def ask(self, fixed=None):
        if self.pending is not None:
            raise ValueError('Complete the pending LLaMEA proposal before asking again')
        if len(self.attempts) >= self.budget:
            raise ValueError('LLaMEA CFD budget is exhausted')
        if fixed is not None:
            if len(self.attempts) >= self.initial:
                raise ValueError('Fixed designs are only allowed during shared initialization')
            point = np.asarray(fixed, dtype=float).copy()
            details = {'proposal': 'llamea_shared_initial'}
        else:
            if len(self.attempts) < self.initial:
                raise ValueError('Replay all shared initial attempts before LLaMEA evolution')
            # Malformed code consumes a code-generation attempt but no CFD call.
            # The finite candidate limit prevents an endless repair loop.
            while True:
                if self.active is None and self.generated < self.limit:
                    self.new_candidate()
                    if self.active is None:
                        continue
                if self.active is not None:
                    candidate = self.active
                    x, y, attempts = candidate['x'], candidate['y'], candidate['attempts']
                    code = candidate['solution'].code
                    # Common random numbers improve candidate comparability.
                    draw_seed = (self.seed+104729*(candidate['steps']+1)) % (2**32-1)
                    phase = 'candidate'
                elif self.best is not None:
                    candidate = self.best
                    x, y, attempts, code = self.x, self.y, self.attempts, candidate['code']
                    draw_seed = (self.seed+104729*(len(self.attempts)+1)) % (2**32-1)
                    phase = 'deployment'
                else:
                    raise ValueError('LLaMEA generated no valid optimizer. Inspect code_evolution/candidates.json and '
                                     'llm_*.json; no fallback optimizer or synthetic evaluation was used')
                try:
                    point = run_proposal(code, x, y, attempts, self.reference,
                                         draw_seed, self.settings.proposal_timeout)
                except (ValueError, SyntaxError, OSError) as exc:
                    if phase == 'deployment':
                        raise ValueError(f'LLaMEA winner failed during deployment: {exc}') from exc
                    self.finish_candidate(str(exc))
                    continue
                details = {'proposal': f'llamea_{phase}', 'candidate': candidate['index'],
                           'code_sha256': candidate['sha256'], 'source': str(self.folder/candidate['source']),
                           'proposal_seed': draw_seed}
                break
        self.pending = (point.copy(), fixed is not None)
        return point, details

    def tell(self, x, y):
        if self.pending is None or not np.array_equal(x, self.pending[0]):
            raise ValueError('LLaMEA outcome does not match a pending proposal')
        if y is not None and (np.shape(y) != (2,) or not np.isfinite(y).all()):
            raise ValueError('Expected two finite CFD objectives or None for failure')
        point = np.asarray(x, dtype=float).tolist()
        values = None if y is None else np.asarray(y, dtype=float).tolist()
        initial = self.pending[1]
        self.attempts.append(point)
        if values is not None:
            self.x.append(point)
            self.y.append(values)
        if initial:
            self.shared_attempts.append(point)
            if values is not None:
                self.shared_x.append(point)
                self.shared_y.append(values)
        elif self.active is not None:
            self.active['attempts'].append(point)
            self.active['steps'] += 1
            if values is not None:
                self.active['x'].append(point)
                self.active['y'].append(values)
            if self.active['steps'] == self.steps:
                self.finish_candidate()
        self.pending = None
