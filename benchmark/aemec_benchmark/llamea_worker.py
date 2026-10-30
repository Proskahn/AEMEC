"""Run one generated numerical proposal in a disposable, time-limited process.

The parent supplies JSON, never a CFD callback or credentials. A small AST/API
allowlist rejects file, network, subprocess and introspection operations. This
is a restricted numerical runner, not a general-purpose Python security sandbox.
Keep it separate from the process that owns the LLM client and CFD evaluator.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import math
import sys

import numpy as np

MAX_CODE_BYTES = 65536
# Both NumPy functions and array/Generator methods are enumerated deliberately.
# In particular, load/save, ctypes, data/base, lib and private attributes are absent.
ATTRIBUTES = set('''abs absolute all allclose any append arange argmax argmin argpartition argsort
array asarray astype atleast_1d atleast_2d average choice clip column_stack concatenate
copy corrcoef cos count_nonzero cov cumsum diag diagonal diff dirichlet dot einsum empty empty_like exp eye flatten
float64 floor full full_like hstack inf int64 integers isclose isfinite isnan linalg
LinAlgError linspace log log10 log1p lstsq max maximum mean median min minimum multivariate_normal nan
nan_to_num nanmax nanmin ndim newaxis norm normal ones ones_like permutation pi pinv power prod ptp
quantile ravel repeat reshape round shape sin size solve sort sqrt square stack std
standard_normal sum T take tanh tile transpose tril triu uniform unique var vstack where zeros zeros_like'''.split())
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'int', 'len', 'list',
    'max', 'min', 'pow', 'range', 'reversed', 'round', 'set', 'sorted', 'sum', 'tuple',
    'zip', 'ValueError', 'ArithmeticError', 'ZeroDivisionError')}


def validate_code(code):
    """Check syntax and the allowed numerical interface without executing code."""
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > MAX_CODE_BYTES:
        raise ValueError('Expected nonempty Python source of at most 64 KiB')
    tree = ast.parse(code)
    if len(list(ast.walk(tree))) > 10000:
        raise ValueError('Generated program is too large')
    proposals = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'propose']
    if len(proposals) != 1:
        raise ValueError('Define exactly one top-level propose(x, y, attempts, reference, rng) function')
    args = proposals[0].args
    if (len(args.args) != 5 or args.posonlyargs or args.kwonlyargs or args.vararg
            or args.kwarg or args.defaults):
        raise ValueError('propose must take exactly five positional arguments')
    for node in tree.body:
        if not (isinstance(node, (ast.Import, ast.FunctionDef)) or
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            raise ValueError('Only imports, functions and docstrings are allowed at module level')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name not in ('numpy', 'math') or (a.asname or '').startswith('__') for a in node.names):
                raise ValueError('Only numpy and math imports are allowed')
        if isinstance(node, (ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal,
                             ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Await,
                             ast.Yield, ast.YieldFrom)):
            raise ValueError(f'{type(node).__name__} is outside the proposal interface')
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith('__')):
            raise ValueError('Decorators and special function names are not allowed')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Private names are not allowed')
        if isinstance(node, ast.arg) and node.arg.startswith('__'):
            raise ValueError('Private argument names are not allowed')
        if isinstance(node, ast.Attribute) and node.attr not in ATTRIBUTES:
            raise ValueError(f'Attribute {node.attr!r} is outside the numerical API')
    return tree


def numerical_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or fromlist or name not in ('numpy', 'math'):
        raise ImportError('Only numpy and math imports are allowed')
    return np if name == 'numpy' else math


def main():
    # The parent also enforces wall time and kills the process on timeout.
    # Linux address-space limits bound runaway allocations; macOS may not
    # support RLIMIT_AS reliably, so the timeout remains the portable limit.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if sys.platform.startswith('linux'):
            limit = 2 * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass
    try:
        raw = sys.stdin.read(4*1024**2+1)
        if len(raw) > 4*1024**2:
            raise ValueError('Proposal input is too large')
        data = json.loads(raw)
        tree = validate_code(data['code'])
        namespace = {'__builtins__': {**SAFE_BUILTINS, '__import__': numerical_import}}
        np.random.seed(data['seed'])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, '<llamea-proposal>', 'exec'), namespace)
            point = namespace['propose'](
                np.asarray(data['x'], dtype=float).reshape(-1, 3),
                np.asarray(data['y'], dtype=float).reshape(-1, 2),
                np.asarray(data['attempts'], dtype=float).reshape(-1, 3),
                np.asarray(data['reference'], dtype=float),
                np.random.default_rng(data['seed']))
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1):
            raise ValueError('propose must return three finite values in [0, 1]')
        print(json.dumps({'point': point.tolist()}, allow_nan=False))
    except Exception as exc:
        # Only input numerical arrays and source are in this process. Limit the
        # message so a generated exception cannot produce an unbounded log.
        print(json.dumps({'error': f'{type(exc).__name__}: {str(exc)[:500]}'}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
