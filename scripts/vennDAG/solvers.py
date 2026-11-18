"""
Solvers for the VennDAG set reduction problem.
Provides:
1. Greedy Solver (fast approximation with maximum utility heuristic and pruning)
2. Exact MILP Solver (optimal 0-1 Integer Linear Program using SciPy HiGHS)
"""

import heapq
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
    Features:
    - Inverted element-centric bitset representation (O(V) bitwise operations, minimal memory)
    - Universal / Heavy-element fast-path pre-pass
    - Lazy max-heap candidate scoring
    - Fast O(1)-bitmask redundancy pruning pass
    """

    def __init__(self, graph: DAGGraph):
        self.graph = graph

    def solve(self, prune_redundant: bool = True) -> DAGGraph:
        """
        Execute the greedy reduction algorithm.
        Returns a new DAGGraph with reduced member sets.
        """
        topo, node_to_idx, reach, rev_reach, orig_elem_masks, orig_node_elem_nodes = (
            self.graph.compute_reachability_bitmasks()
        )
        N = len(topo)

        # Build list of node indices for each element
        elem_node_list: Dict[str, List[int]] = {}
        for elem, mask in orig_elem_masks.items():
            lst = []
            cur = mask
            while cur:
                lsb = cur & -cur
                lst.append(lsb.bit_length() - 1)
                cur ^= lsb
            elem_node_list[elem] = lst

        # Unsatisfied reachable pairs mask per node
        unsatisfied_reach: List[int] = [reach[i] & orig_node_elem_nodes[i] for i in range(N)]
        total_unsatisfied = sum(unsatisfied_reach[i].bit_count() for i in range(N))

        # If no active pairs, return empty member sets
        if total_unsatisfied == 0:
            result_graph = self.graph.copy()
            for nid in result_graph.nodes:
                result_graph.nodes[nid].members = set()
            return result_graph

        reduced_elem_masks: Dict[str, int] = {elem: 0 for elem in orig_elem_masks}
        reduced_node_members: List[Set[str]] = [set() for _ in range(N)]

        # Maintain active node indices per candidate element
        elem_active_nodes: Dict[str, List[int]] = {
            elem: [i for i in nodes if unsatisfied_reach[i] & orig_elem_masks[elem]]
            for elem, nodes in elem_node_list.items()
        }
        elem_version: Dict[str, int] = {elem: 0 for elem in orig_elem_masks}

        def eval_elem(elem: str) -> Tuple[int, int, float]:
            nodes = elem_active_nodes.get(elem)
            if not nodes:
                return 0, 0, 0.0
            e_mask = orig_elem_masks[elem]
            gain = 0
            new_nodes = []
            for i in nodes:
                sat = unsatisfied_reach[i] & e_mask
                if sat:
                    gain += sat.bit_count()
                    new_nodes.append(i)
            elem_active_nodes[elem] = new_nodes
            if gain == 0:
                return 0, 0, 0.0
            cost = (e_mask & ~reduced_elem_masks[elem]).bit_count()
            ratio = float("inf") if cost == 0 else (gain / cost)
            return gain, cost, ratio

        # Strategy 3: Heavy / Universal Element Fast-Path Pre-Pass
        # For large elements with high gain/cost ratios, assign them upfront
        sorted_elems = sorted(orig_elem_masks.keys(), key=lambda e: len(elem_node_list[e]), reverse=True)
        for elem in sorted_elems:
            if total_unsatisfied == 0:
                break
            if len(elem_node_list[elem]) < 20:
                break
            gain, cost, ratio = eval_elem(elem)
            if gain > 0 and (cost == 0 or ratio >= 2.0):
                e_mask = orig_elem_masks[elem]
                reduced_elem_masks[elem] |= e_mask
                elem_version[elem] += 1
                touched_nodes = set()
                for i in elem_node_list[elem]:
                    reduced_node_members[i].add(elem)
                    sat = unsatisfied_reach[i] & e_mask
                    if sat:
                        unsatisfied_reach[i] ^= sat
                        total_unsatisfied -= sat.bit_count()
                        touched_nodes.add(i)
                if touched_nodes:
                    for other_elem, nodes in elem_active_nodes.items():
                        if other_elem != elem and any(i in touched_nodes for i in nodes):
                            elem_version[other_elem] += 1

        # Strategy 4 & 1: Lazy Priority-Queue Candidate Selection
        pq: List[Tuple[float, int, str, int]] = []
        for elem in orig_elem_masks:
            gain, cost, ratio = eval_elem(elem)
            if gain > 0:
                heapq.heappush(pq, (-ratio, -gain, elem, elem_version[elem]))

        while total_unsatisfied > 0 and pq:
            neg_ratio, neg_gain, best_elem, v = heapq.heappop(pq)
            if v != elem_version[best_elem]:
                gain, cost, ratio = eval_elem(best_elem)
                elem_version[best_elem] += 1
                if gain > 0:
                    heapq.heappush(pq, (-ratio, -gain, best_elem, elem_version[best_elem]))
                continue

            e_mask = orig_elem_masks[best_elem]
            reduced_elem_masks[best_elem] |= e_mask
            elem_version[best_elem] += 1

            touched_nodes = set()
            for i in elem_node_list[best_elem]:
                reduced_node_members[i].add(best_elem)
                sat = unsatisfied_reach[i] & e_mask
                if sat:
                    unsatisfied_reach[i] ^= sat
                    total_unsatisfied -= sat.bit_count()
                    touched_nodes.add(i)

            if touched_nodes:
                for other_elem, nodes in elem_active_nodes.items():
                    if other_elem != best_elem and any(i in touched_nodes for i in nodes):
                        elem_version[other_elem] += 1

        # Strategy 2: Fast Bitset Redundancy Pruning
        if prune_redundant:
            self._prune_redundancies_bitset(
                N=N,
                reach=reach,
                rev_reach=rev_reach,
                orig_node_elem_nodes=orig_node_elem_nodes,
                reduced_node_members=reduced_node_members,
                reduced_elem_masks=reduced_elem_masks
            )

        # Construct result graph
        result_graph = self.graph.copy()
        for i, nid in enumerate(topo):
            result_graph.nodes[nid].members = reduced_node_members[i]

        return result_graph

    def _prune_redundancies_bitset(
        self,
        N: int,
        reach: List[int],
        rev_reach: List[int],
        orig_node_elem_nodes: List[int],
        reduced_node_members: List[Set[str]],
        reduced_elem_masks: Dict[str, int]
    ) -> None:
        """
        Fast O(1)-bitmask redundancy pruning pass.
        Eliminates redundant element assignments without breaking reachability overlap constraints.
        """
        for i in range(N):
            mems = reduced_node_members[i]
            if len(mems) <= 1:
                continue
            out_needed = reach[i] & orig_node_elem_nodes[i]
            in_needed = rev_reach[i] & orig_node_elem_nodes[i]

            # Compute cov_1 and cov_2 bitmasks
            cov_1 = 0
            cov_2 = 0
            for elem in mems:
                mask = reduced_elem_masks[elem]
                cov_2 |= (cov_1 & mask)
                cov_1 |= mask

            for elem in list(mems):
                if len(mems) <= 1:
                    break
                unique_to_e = reduced_elem_masks[elem] & ~cov_2
                if (out_needed & unique_to_e == 0) and (in_needed & unique_to_e == 0):
                    mems.remove(elem)
                    reduced_elem_masks[elem] ^= (1 << i)
                    # Recompute cov_1, cov_2
                    cov_1 = 0
                    cov_2 = 0
                    for rem_elem in mems:
                        mask = reduced_elem_masks[rem_elem]
                        cov_2 |= (cov_1 & mask)
                        cov_1 |= mask


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

    # Free ephemeral graph caches
    graph.clear_cache()
    result.clear_cache()

    return result, stats
