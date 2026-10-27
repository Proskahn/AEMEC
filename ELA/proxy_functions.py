"""Safe, deterministic expression trees for ELA-matched scalar proxy functions.

Only NumPy is required at evaluation time. No Python eval, pickle, CFD, or GP
regression is used. The primitive vocabulary follows Table 1 of arXiv:2602.04529v1.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ARITY = {**dict.fromkeys(('add', 'sub', 'mul', 'div'), 2),
         **dict.fromkeys(('neg', 'rec', 'mult10', 'square', 'abs', 'sqrt', 'exp',
                          'ln', 'sin', 'cos', 'round', 'sum', 'mean', 'cum',
                          'prod', 'max'), 1)}


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple = ()
    value: float = None

    @property
    def depth(self):
        return 0 if not self.children else 1 + max(c.depth for c in self.children)

    @property
    def size(self):
        return 1 + sum(c.size for c in self.children)

    def to_dict(self):
        if self.op == 'constant':
            return {'op': self.op, 'value': self.value}
        return {'op': self.op, 'children': [c.to_dict() for c in self.children]}

    def expression(self):
        if self.op == 'constant':
            return repr(self.value)
        if self.op in ('x', 'index'):
            return 'u' if self.op == 'x' else 'index'
        return f"{self.op}({', '.join(c.expression() for c in self.children)})"


def parse_tree(data, level=0, budget=None):
    """Load a bounded, whitelisted AST; never execute serialized source code."""
    if budget is None:
        budget = [4096]
    budget[0] -= 1
    if level > 64 or budget[0] < 0 or not isinstance(data, dict):
        raise ValueError('Invalid or excessively large proxy expression')
    op = data.get('op')
    if op == 'constant':
        value = float(data['value'])
        if not np.isfinite(value):
            raise ValueError('Proxy constants must be finite')
        return Node(op, value=value)
    arity = 0 if op in ('x', 'index') else ARITY.get(op)
    children = data.get('children', [])
    if arity is None or not isinstance(children, list) or len(children) != arity:
        raise ValueError(f'Unknown primitive or invalid arity: {op}')
    return Node(op, tuple(parse_tree(c, level + 1, budget) for c in children))


def evaluate_tree(tree, x):
    """Evaluate rows independently; reduce a vector root by summing coordinates.

    x and index are vectors; constants have width one and broadcast. Vector
    reductions operate over coordinates, NEVER over rows in the design pool.
    Invalid arithmetic remains nonfinite and is penalized by the search.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError('Expression inputs must have shape (n, dimension)')

    def visit(node):
        if node.op == 'x':
            return x
        if node.op == 'index':
            return np.broadcast_to(np.arange(1, x.shape[1]+1, dtype=float), x.shape)
        if node.op == 'constant':
            return np.full((len(x), 1), node.value)
        args = [visit(c) for c in node.children]
        a = args[0]
        if node.op == 'add': return a + args[1]
        if node.op == 'sub': return a - args[1]
        if node.op == 'mul': return a * args[1]
        if node.op == 'div': return a / args[1]
        if node.op == 'neg': return -a
        if node.op == 'rec': return 1 / a
        if node.op == 'mult10': return 10 * a
        if node.op == 'square': return np.square(a)
        if node.op == 'abs': return np.abs(a)
        if node.op == 'sqrt': return np.sqrt(np.abs(a))
        if node.op == 'exp': return np.exp(a)
        if node.op == 'ln': return np.log(np.abs(a))
        if node.op == 'sin': return np.sin(2*np.pi*a)
        if node.op == 'cos': return np.cos(2*np.pi*a)
        if node.op == 'round': return np.rint(a)
        if node.op == 'sum': return np.sum(a, axis=1, keepdims=True)
        if node.op == 'mean': return np.mean(a, axis=1, keepdims=True)
        if node.op == 'cum': return np.cumsum(a, axis=1)
        if node.op == 'prod': return np.prod(a, axis=1, keepdims=True)
        if node.op == 'max': return np.max(a, axis=1, keepdims=True)
        raise ValueError(f'Unknown primitive: {node.op}')

    with np.errstate(all='ignore'):
        result = visit(tree)
        return np.sum(result, axis=1) if result.shape[1] != 1 else result[:, 0]


class ProxyFunction:
    """Callable scalar benchmark accepting physical AEMEC inputs.

    Returns normalized, dimensionless proxy values. These are not predictions
    of voltage/crossover; objective values and minimizers need not match CFD.
    """
    def __init__(self, artifact):
        if isinstance(artifact, (str, Path)):
            artifact = json.loads(Path(artifact).read_text())
        if artifact.get('schema_version') != 1:
            raise ValueError('Unsupported proxy artifact schema')
        self.tree = parse_tree(artifact['tree'])
        self.input_columns = list(artifact['input_columns'])
        self.lower = np.asarray(artifact['input_lower'], dtype=float)
        self.upper = np.asarray(artifact['input_upper'], dtype=float)
        self.offset = float(artifact['raw_offset'])
        self.scale = float(artifact['raw_scale'])
        self.landscape = artifact['landscape']
        if (self.lower.shape != (len(self.input_columns),) or self.upper.shape != self.lower.shape
                or not np.isfinite([self.lower, self.upper]).all()
                or np.any(self.upper <= self.lower)
                or not np.isfinite([self.offset, self.scale]).all() or self.scale <= 0):
            raise ValueError('Invalid proxy bounds or normalization')

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.lower) or not np.isfinite(x).all():
            raise ValueError('Expected finite physical designs with shape (n, 3) or (3,)')
        if np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError('Proxy input is outside the declared physical bounds')
        raw = evaluate_tree(self.tree, (x-self.lower)/(self.upper-self.lower))
        with np.errstate(all='ignore'):
            result = (raw-self.offset)/self.scale
        if not np.isfinite(result).all():
            raise ValueError('Proxy expression is nonfinite at this design')
        return float(result[0]) if single else result
