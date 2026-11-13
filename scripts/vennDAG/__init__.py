"""
VennDAG: Minimal reachability-preserving set reduction on DAGs.
"""

from .ir import DAGGraph, Node, Edge
from .dot_parser import parse_dot, parse_dot_graphs, DotParser
from .dot_writer import export_dot, render_graph, DotWriter
from .solvers import solve_graph, GreedySolver, MilpSolver

__all__ = [
    "DAGGraph",
    "Node",
    "Edge",
    "parse_dot",
    "parse_dot_graphs",
    "DotParser",
    "export_dot",
    "render_graph",
    "DotWriter",
    "solve_graph",
    "GreedySolver",
    "MilpSolver",
]
