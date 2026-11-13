"""
Solvers for the VennDAG set reduction problem.
Provides:
1. Greedy Solver (fast approximation with maximum utility heuristic and pruning)
2. Exact MILP Solver (optimal 0-1 Integer Linear Program using SciPy HiGHS)
"""

from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, Any
import time

from .ir import DAGGraph


class GreedySolver:
    """
    Greedy approximation solver for reachability-preserving set reduction on DAGs.
    
    Heuristic:
    Iteratively picks (element, cluster) that maximizes the ratio:
        Score = (number of newly satisfied active pairs) / (number of new node assignments)
    Followed by a pruning pass to eliminate any redundant element assignments.
    """

    def __init__(self, graph: DAGGraph):
        self.graph = graph

    def solve(self, prune_redundant: bool = True) -> DAGGraph:
        """
        Execute the greedy reduction algorithm.
        Returns a new DAGGraph with reduced member sets.
        """
        active_pairs = self.graph.get_active_overlap_pairs()
        
        # Initialize reduced sets as empty sets
        reduced_sets: Dict[str, Set[str]] = {nid: set() for nid in self.graph.nodes}
        
        # Track unsatisfied pairs and their valid candidates: (u, v) -> set of elements
        unsatisfied_pairs: Dict[Tuple[str, str], Set[str]] = {
            (u, v): set(common) for u, v, common in active_pairs
        }

        # Element to pairs mapping
        elem_to_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        for pair, common in unsatisfied_pairs.items():
            for elem in common:
                elem_to_pairs[elem].add(pair)

        while unsatisfied_pairs:
            best_elem: Optional[str] = None
            best_ratio = -1.0
            best_pairs_covered: Set[Tuple[str, str]] = set()
            best_nodes_to_add: Set[str] = set()

            # Iterate over elements that can satisfy remaining pairs
            for elem, pairs in elem_to_pairs.items():
                active_for_elem = pairs & unsatisfied_pairs.keys()
                if not active_for_elem:
                    continue

                # Find connected components of pairs for this element
                # Each component can be chosen independently for better granularity
                components = self._find_pair_components(active_for_elem)

                for comp_pairs in components:
                    nodes_needing_elem: Set[str] = set()
                    for u, v in comp_pairs:
                        if elem not in reduced_sets[u]:
                            nodes_needing_elem.add(u)
                        if elem not in reduced_sets[v]:
                            nodes_needing_elem.add(v)

                    cost = len(nodes_needing_elem)
                    if cost == 0:
                        # Already covered without additional cost
                        benefit = len(comp_pairs)
                        ratio = float('inf')
                    else:
                        benefit = len(comp_pairs)
                        ratio = benefit / cost

                    if ratio > best_ratio or (ratio == best_ratio and benefit > len(best_pairs_covered)):
                        best_ratio = ratio
                        best_elem = elem
                        best_pairs_covered = comp_pairs
                        best_nodes_to_add = nodes_needing_elem

            if best_elem is None or not best_pairs_covered:
                # Fallback if no candidate found
                break

            # Apply selection
            for nid in best_nodes_to_add:
                reduced_sets[nid].add(best_elem)

            for pair in best_pairs_covered:
                if pair in unsatisfied_pairs:
                    del unsatisfied_pairs[pair]

        # Post-processing: Prune any redundant elements
        if prune_redundant:
            self._prune_redundancies(reduced_sets, active_pairs)

        # Build output graph
        result_graph = self.graph.copy()
        for nid, members in reduced_sets.items():
            result_graph.nodes[nid].members = members
            result_graph.nodes[nid].original_members = set(self.graph.nodes[nid].members)

        return result_graph

    def _find_pair_components(self, pairs: Set[Tuple[str, str]]) -> List[Set[Tuple[str, str]]]:
        """Group pairs that share nodes into connected components."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        node_to_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        
        for u, v in pairs:
            adj[u].add(v)
            adj[v].add(u)
            node_to_pairs[u].add((u, v))
            node_to_pairs[v].add((u, v))

        visited_nodes: Set[str] = set()
        components: List[Set[Tuple[str, str]]] = []

        for node in adj:
            if node not in visited_nodes:
                comp_nodes: Set[str] = set()
                queue = [node]
                visited_nodes.add(node)
                while queue:
                    curr = queue.pop()
                    comp_nodes.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            queue.append(neighbor)

                comp_pairs: Set[Tuple[str, str]] = set()
                for n in comp_nodes:
                    comp_pairs.update(node_to_pairs[n])
                components.append(comp_pairs)

        return components if components else [pairs]

    def _prune_redundancies(self, reduced_sets: Dict[str, Set[str]],
                           active_pairs: List[Tuple[str, str, Set[str]]]) -> None:
        """
        Iteratively remove elements from nodes if doing so does not violate
        any active reachability overlap constraints.
        """
        # Node to associated active pairs
        node_incoming_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        node_outgoing_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for u, v, _ in active_pairs:
            node_outgoing_pairs[u].append((u, v))
            node_incoming_pairs[v].append((u, v))

        # Check each node and each element
        for nid, elements in list(reduced_sets.items()):
            for elem in list(elements):
                # Test removing elem
                can_remove = True

                # Check outgoing pairs (nid, v)
                for u, v in node_outgoing_pairs[nid]:
                    u_set = reduced_sets[u] - {elem}
                    v_set = reduced_sets[v]
                    if not (u_set & v_set):
                        can_remove = False
                        break

                if not can_remove:
                    continue

                # Check incoming pairs (u, nid)
                for u, v in node_incoming_pairs[nid]:
                    u_set = reduced_sets[u]
                    v_set = reduced_sets[v] - {elem}
                    if not (u_set & v_set):
                        can_remove = False
                        break

                if can_remove:
                    reduced_sets[nid].remove(elem)


class MilpSolver:
    """
    Exact Integer Linear Programming (ILP/MILP) solver using SciPy's HiGHS solver.
    Guarantees globally minimal total element count.
    """

    def __init__(self, graph: DAGGraph):
        self.graph = graph

    def solve(self) -> DAGGraph:
        """
        Formulate and solve 0-1 MILP.
        """
        try:
            import numpy as np
            from scipy.optimize import milp, LinearConstraint
            from scipy.sparse import dok_matrix, csc_matrix
        except ImportError as e:
            raise RuntimeError(f"SciPy is required for exact MILP solving: {e}")

        active_pairs = self.graph.get_active_overlap_pairs()
        nodes = list(self.graph.nodes.keys())

        # If no active pairs, return empty sets
        if not active_pairs:
            result_graph = self.graph.copy()
            for nid in result_graph.nodes:
                result_graph.nodes[nid].members = set()
                result_graph.nodes[nid].original_members = set(self.graph.nodes[nid].members)
            return result_graph

        # Indexing:
        # x_{v, e} : variable indicating element e is kept in node v
        # y_{u, v, e} : variable indicating element e satisfies pair (u, v)

        var_x_map: Dict[Tuple[str, str], int] = {}  # (node, elem) -> var_idx
        var_y_map: Dict[Tuple[str, str, str], int] = {}  # (u, v, elem) -> var_idx

        num_vars = 0
        for nid in nodes:
            for elem in sorted(self.graph.nodes[nid].members):
                var_x_map[(nid, elem)] = num_vars
                num_vars += 1

        for u, v, common in active_pairs:
            for elem in sorted(common):
                var_y_map[(u, v, elem)] = num_vars
                num_vars += 1

        # Objective: minimize sum of x_{v, e}
        c = np.zeros(num_vars, dtype=float)
        for (nid, elem), idx in var_x_map.items():
            c[idx] = 1.0

        # Constraints list
        # 1. Witness validity: y_{u, v, e} <= x_{u, e}  =>  y_{u, v, e} - x_{u, e} <= 0
        # 2. Witness validity: y_{u, v, e} <= x_{v, e}  =>  y_{u, v, e} - x_{v, e} <= 0
        # 3. Pair coverage: sum_{e in C(u, v)} y_{u, v, e} >= 1
        num_constraints = 2 * len(var_y_map) + len(active_pairs)
        A = dok_matrix((num_constraints, num_vars), dtype=float)
        lhs = np.zeros(num_constraints, dtype=float)
        rhs = np.zeros(num_constraints, dtype=float)

        row = 0
        # Constraints 1 & 2
        for (u, v, elem), y_idx in var_y_map.items():
            x_u_idx = var_x_map[(u, elem)]
            x_v_idx = var_x_map[(v, elem)]

            # y - x_u <= 0
            A[row, y_idx] = 1.0
            A[row, x_u_idx] = -1.0
            lhs[row] = -np.inf
            rhs[row] = 0.0
            row += 1

            # y - x_v <= 0
            A[row, y_idx] = 1.0
            A[row, x_v_idx] = -1.0
            lhs[row] = -np.inf
            rhs[row] = 0.0
            row += 1

        # Constraint 3: coverage
        for u, v, common in active_pairs:
            for elem in common:
                y_idx = var_y_map[(u, v, elem)]
                A[row, y_idx] = 1.0
            lhs[row] = 1.0
            rhs[row] = np.inf
            row += 1

        constraints = LinearConstraint(A.tocsc(), lhs, rhs)
        integrality = np.ones(num_vars, dtype=int)  # All variables binary (0 or 1)
        bounds_lb = np.zeros(num_vars, dtype=float)
        bounds_ub = np.ones(num_vars, dtype=float)
        from scipy.optimize import Bounds
        bounds = Bounds(bounds_lb, bounds_ub)

        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)

        if not res.success:
            raise RuntimeError(f"MILP solver failed: {res.status} - {res.message}")

        # Extract solution
        reduced_sets: Dict[str, Set[str]] = {nid: set() for nid in nodes}
        sol = res.x
        for (nid, elem), idx in var_x_map.items():
            if sol[idx] > 0.5:
                reduced_sets[nid].add(elem)

        result_graph = self.graph.copy()
        for nid, members in reduced_sets.items():
            result_graph.nodes[nid].members = members
            result_graph.nodes[nid].original_members = set(self.graph.nodes[nid].members)

        return result_graph


def solve_graph(graph: DAGGraph, method: str = "greedy", prune: bool = True) -> Tuple[DAGGraph, Dict[str, Any]]:
    """
    Solve set minimization problem on graph.
    method can be 'greedy' (default) or 'exact' (MILP).
    Returns (solved_graph, info_dict).
    """
    t0 = time.perf_counter()

    if method == "exact":
        solver = MilpSolver(graph)
        result = solver.solve()
    elif method == "greedy":
        solver = GreedySolver(graph)
        result = solver.solve(prune_redundant=prune)
    else:
        raise ValueError(f"Unknown solver method '{method}', must be 'greedy' or 'exact'")

    t1 = time.perf_counter()
    duration = t1 - t0

    is_valid, errors = result.verify_solution(graph)
    stats = result.get_statistics(original_graph=graph)
    stats["solver"] = method
    stats["solve_time_sec"] = duration
    stats["is_valid"] = is_valid
    stats["validation_errors"] = errors

    return result, stats
