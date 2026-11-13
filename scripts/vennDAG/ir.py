"""
Intermediate Representation (IR) for VennDAG.
Represents directed acyclic graphs with member sets associated with nodes,
reachability analysis, and solution validation.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any, Iterator


@dataclass
class Node:
    """Represents a node in the DAG with associated member set and attributes."""
    id: str
    members: Set[str] = field(default_factory=set)
    label: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    original_members: Optional[Set[str]] = None

    def copy(self) -> 'Node':
        return Node(
            id=self.id,
            members=set(self.members),
            label=self.label,
            attributes=dict(self.attributes),
            original_members=set(self.original_members) if self.original_members is not None else set(self.members)
        )


@dataclass
class Edge:
    """Represents a directed edge between two nodes."""
    src: str
    dst: str
    attributes: Dict[str, str] = field(default_factory=dict)


class DAGGraph:
    """
    Intermediate representation for a DAG with associated member sets per node.
    """

    def __init__(self, name: str = "G", is_directed: bool = True):
        self.name: str = name
        self.is_directed: bool = is_directed
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.graph_attributes: Dict[str, str] = {}
        self._adj: Dict[str, List[str]] = defaultdict(list)
        self._in_degree: Dict[str, int] = defaultdict(int)
        self._reachability_cache: Optional[Dict[str, Set[str]]] = None
        self._active_pairs_cache: Optional[List[Tuple[str, str, Set[str]]]] = None

    def add_node(self, node_id: str, members: Optional[Set[str]] = None,
                 label: Optional[str] = None, attributes: Optional[Dict[str, str]] = None) -> Node:
        """Add a node or update an existing node."""
        if node_id not in self.nodes:
            node = Node(
                id=node_id,
                members=set(members) if members is not None else set(),
                label=label,
                attributes=dict(attributes) if attributes is not None else {}
            )
            self.nodes[node_id] = node
            self._adj[node_id] = []
            self._in_degree[node_id] = 0
            self._invalidate_cache()
            return node
        else:
            node = self.nodes[node_id]
            if members is not None:
                node.members.update(members)
            if label is not None:
                node.label = label
            if attributes:
                node.attributes.update(attributes)
            self._invalidate_cache()
            return node

    def add_edge(self, src: str, dst: str, attributes: Optional[Dict[str, str]] = None) -> Edge:
        """Add a directed edge from src to dst."""
        self.add_node(src)
        self.add_node(dst)
        edge = Edge(src=src, dst=dst, attributes=dict(attributes) if attributes is not None else {})
        self.edges.append(edge)
        self._adj[src].append(dst)
        self._in_degree[dst] += 1
        self._invalidate_cache()
        return edge

    def _invalidate_cache(self) -> None:
        self._reachability_cache = None
        self._active_pairs_cache = None

    def copy(self) -> 'DAGGraph':
        """Create a deep copy of the graph."""
        new_g = DAGGraph(name=self.name, is_directed=self.is_directed)
        new_g.graph_attributes = dict(self.graph_attributes)
        for nid, node in self.nodes.items():
            new_g.nodes[nid] = node.copy()
        for edge in self.edges:
            new_g.add_edge(edge.src, edge.dst, edge.attributes)
        return new_g

    def successors(self, node_id: str) -> List[str]:
        return self._adj.get(node_id, [])

    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order.
        Raises ValueError if a cycle is detected.
        """
        in_deg = {n: 0 for n in self.nodes}
        for edge in self.edges:
            in_deg[edge.dst] = in_deg.get(edge.dst, 0) + 1

        queue = deque([n for n, d in in_deg.items() if d == 0])
        topo = []
        while queue:
            curr = queue.popleft()
            topo.append(curr)
            for nxt in self._adj.get(curr, []):
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)

        if len(topo) != len(self.nodes):
            raise ValueError(
                f"Graph contains cycles! Topologically sorted {len(topo)} of {len(self.nodes)} nodes."
            )
        return topo

    def is_acyclic(self) -> bool:
        """Check if the graph is a DAG."""
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def compute_reachability(self) -> Dict[str, Set[str]]:
        """
        Compute the transitive closure (reachability map).
        reachable[u] contains all nodes v reachable from u via paths of length >= 1.
        """
        if self._reachability_cache is not None:
            return self._reachability_cache

        try:
            topo = self.topological_sort()
            reachable: Dict[str, Set[str]] = {n: set() for n in self.nodes}
            # Compute reachable sets in reverse topological order
            for u in reversed(topo):
                for v in self._adj.get(u, []):
                    reachable[u].add(v)
                    reachable[u].update(reachable[v])
            self._reachability_cache = reachable
            return reachable
        except ValueError:
            # Fallback for graphs with cycles (DFS for each node)
            reachable = {n: set() for n in self.nodes}
            for u in self.nodes:
                visited: Set[str] = set()
                queue = deque([u])
                while queue:
                    curr = queue.popleft()
                    for nxt in self._adj.get(curr, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
                reachable[u] = visited
            self._reachability_cache = reachable
            return reachable

    def get_active_overlap_pairs(self) -> List[Tuple[str, str, Set[str]]]:
        """
        Find all active requirement pairs (u, v) where:
        1. u != v
        2. There is a directed path u -> ... -> v
        3. S(u) and S(v) have non-empty intersection
        Returns a list of tuples: (u, v, S(u) & S(v))
        """
        if self._active_pairs_cache is not None:
            return self._active_pairs_cache

        reach = self.compute_reachability()
        active_pairs = []
        for u, targets in reach.items():
            u_members = self.nodes[u].members
            if not u_members:
                continue
            for v in targets:
                v_members = self.nodes[v].members
                common = u_members & v_members
                if common:
                    active_pairs.append((u, v, common))

        self._active_pairs_cache = active_pairs
        return active_pairs

    def total_member_count(self) -> int:
        """Calculate total number of element instances across all nodes."""
        return sum(len(node.members) for node in self.nodes.values())

    def unique_elements(self) -> Set[str]:
        """Return the universe of unique elements in the graph."""
        res: Set[str] = set()
        for node in self.nodes.values():
            res.update(node.members)
        return res

    def verify_solution(self, original_graph: 'DAGGraph') -> Tuple[bool, List[str]]:
        """
        Verify that the current graph is a valid reduction of original_graph:
        1. S'(v) is a subset of S(v) for all v.
        2. For all pairs (u, v) with u ->...-> v and S(u) & S(v) != empty,
           S'(u) & S'(v) != empty.
        Returns (is_valid, list_of_error_messages).
        """
        errors = []

        # 1. Subset check
        for nid, node in self.nodes.items():
            if nid not in original_graph.nodes:
                errors.append(f"Node {nid} does not exist in original graph.")
                continue
            orig_node = original_graph.nodes[nid]
            if not node.members.issubset(orig_node.members):
                invalid_members = node.members - orig_node.members
                errors.append(
                    f"Node {nid} has elements {invalid_members} not present in original set {orig_node.members}."
                )

        # 2. Reachability overlap preservation check
        active_pairs = original_graph.get_active_overlap_pairs()
        for u, v, orig_common in active_pairs:
            u_red = self.nodes[u].members if u in self.nodes else set()
            v_red = self.nodes[v].members if v in self.nodes else set()
            new_common = u_red & v_red
            if not new_common:
                errors.append(
                    f"Overlap broken between '{u}' and '{v}': originally overlapped on {orig_common}, "
                    f"now S'({u})={u_red} and S'({v})={v_red}."
                )

        return (len(errors) == 0, errors)

    def get_statistics(self, original_graph: Optional['DAGGraph'] = None) -> Dict[str, Any]:
        """Compute summary statistics for the graph."""
        stats = {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "total_elements": self.total_member_count(),
            "unique_elements": len(self.unique_elements()),
            "active_pairs": len(self.get_active_overlap_pairs()),
        }
        if original_graph is not None:
            orig_total = original_graph.total_member_count()
            red_total = self.total_member_count()
            stats["original_elements"] = orig_total
            stats["reduced_elements"] = red_total
            stats["elements_saved"] = orig_total - red_total
            stats["reduction_percentage"] = (
                ((orig_total - red_total) / orig_total * 100.0) if orig_total > 0 else 0.0
            )
        return stats
