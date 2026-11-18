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
            original_members=set(self.original_members) if self.original_members is not None else None
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

    def clear_cache(self) -> None:
        """Clear cached transitive closure and active overlap pairs to free memory."""
        self._reachability_cache = None
        self._active_pairs_cache = None

    def _invalidate_cache(self) -> None:
        self.clear_cache()

    def copy(self) -> 'DAGGraph':
        """Create a deep copy of the graph."""
        new_g = DAGGraph(name=self.name, is_directed=self.is_directed)
        new_g.graph_attributes = dict(self.graph_attributes)
        new_g.nodes = {nid: node.copy() for nid, node in self.nodes.items()}
        new_g.edges = [Edge(src=e.src, dst=e.dst, attributes=dict(e.attributes)) for e in self.edges]
        new_g._adj = {nid: list(adj) for nid, adj in self._adj.items()}
        new_g._in_degree = dict(self._in_degree)
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

    def compute_reachability_bitmasks(self) -> Tuple[List[str], Dict[str, int], List[int], List[int], Dict[str, int], List[int]]:
        """
        Compute compact integer bitmasks for reachability and element support:
        Returns:
            (topo_nodes, node_to_idx, reach_mask, rev_reach_mask, elem_masks, node_elem_nodes)
        Where:
            topo_nodes: List of node IDs in topological order (or arbitrary order if cyclic)
            node_to_idx: Dict mapping node_id -> integer index [0 .. N-1]
            reach_mask: List of size N where reach_mask[i] has bit j set if i reaches j
            rev_reach_mask: List of size N where rev_reach_mask[j] has bit i set if i reaches j
            elem_masks: Dict mapping element -> bitmask of nodes that contain it
            node_elem_nodes: List of size N where node_elem_nodes[i] has bit j set if i and j share an element
        """
        try:
            topo = self.topological_sort()
        except ValueError:
            topo = list(self.nodes.keys())

        node_to_idx = {nid: i for i, nid in enumerate(topo)}
        N = len(topo)

        adj_idx: List[List[int]] = [[] for _ in range(N)]
        for edge in self.edges:
            if edge.src in node_to_idx and edge.dst in node_to_idx:
                adj_idx[node_to_idx[edge.src]].append(node_to_idx[edge.dst])

        # Forward reachability
        reach: List[int] = [0] * N
        for i in reversed(range(N)):
            mask = 0
            for j in adj_idx[i]:
                mask |= (1 << j) | reach[j]
            reach[i] = mask

        # Reverse reachability (nodes that reach node j)
        rev_reach: List[int] = [0] * N
        for i in range(N):
            i_mask = (1 << i) | rev_reach[i]
            for j in adj_idx[i]:
                rev_reach[j] |= i_mask

        # Element support bitmasks
        elem_masks: Dict[str, int] = {}
        for nid, node in self.nodes.items():
            i = node_to_idx[nid]
            for elem in node.members:
                elem_masks[elem] = elem_masks.get(elem, 0) | (1 << i)

        # Node element overlap mask: which nodes share >= 1 element with node i
        node_elem_nodes: List[int] = [0] * N
        for elem, e_mask in elem_masks.items():
            cur = e_mask
            while cur:
                lsb = cur & -cur
                i = (lsb.bit_length() - 1)
                node_elem_nodes[i] |= e_mask
                cur ^= lsb

        return topo, node_to_idx, reach, rev_reach, elem_masks, node_elem_nodes

    def count_active_overlap_pairs(self) -> int:
        """
        Count active requirement pairs (u, v) with reachability and member overlap
        using O(V) bitwise operations with zero pair tuple allocations.
        """
        if self._active_pairs_cache is not None:
            return len(self._active_pairs_cache)
        _, _, reach, _, _, node_elem_nodes = self.compute_reachability_bitmasks()
        return sum((reach[i] & node_elem_nodes[i]).bit_count() for i in range(len(reach)))

    def iter_active_overlap_pairs(self) -> Iterator[Tuple[str, str, Set[str]]]:
        """
        Generator yielding active requirement pairs (u, v, common_set)
        lazily without pre-allocating millions of tuples into memory.
        """
        if self._active_pairs_cache is not None:
            yield from self._active_pairs_cache
            return

        reach = self.compute_reachability()
        for u, targets in reach.items():
            u_members = self.nodes[u].members
            if not u_members:
                continue
            for v in targets:
                v_members = self.nodes[v].members
                common = u_members & v_members
                if common:
                    yield (u, v, common)

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

        active_pairs = list(self.iter_active_overlap_pairs())
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
        Uses fast bitset validation to verify millions of pairs in milliseconds.
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

        # 2. Fast Bitset Reachability Overlap Check
        topo, node_to_idx, reach, _, orig_elem_masks, orig_node_elem_nodes = original_graph.compute_reachability_bitmasks()
        N = len(topo)

        # Compute reduced element bitmasks
        red_node_elem_nodes: List[int] = [0] * N
        for nid, node in self.nodes.items():
            if nid not in node_to_idx:
                continue
            i = node_to_idx[nid]
            for elem in node.members:
                # Bitmask of all nodes in this reduction containing elem
                # Build per-elem masks
                pass

        # Build reduced node overlap masks
        red_elem_masks: Dict[str, int] = {}
        for nid, node in self.nodes.items():
            if nid in node_to_idx:
                i = node_to_idx[nid]
                for elem in node.members:
                    red_elem_masks[elem] = red_elem_masks.get(elem, 0) | (1 << i)

        for elem, e_mask in red_elem_masks.items():
            cur = e_mask
            while cur:
                lsb = cur & -cur
                i = (lsb.bit_length() - 1)
                red_node_elem_nodes[i] |= e_mask
                cur ^= lsb

        for i in range(N):
            needed = reach[i] & orig_node_elem_nodes[i]
            broken = needed & ~red_node_elem_nodes[i]
            if broken:
                u_id = topo[i]
                cur = broken
                while cur:
                    lsb = cur & -cur
                    j = (lsb.bit_length() - 1)
                    cur ^= lsb
                    v_id = topo[j]
                    orig_common = original_graph.nodes[u_id].members & original_graph.nodes[v_id].members
                    u_red = self.nodes[u_id].members if u_id in self.nodes else set()
                    v_red = self.nodes[v_id].members if v_id in self.nodes else set()
                    errors.append(
                        f"Overlap broken between '{u_id}' and '{v_id}': originally overlapped on {orig_common}, "
                        f"now S'({u_id})={u_red} and S'({v_id})={v_red}."
                    )

        return (len(errors) == 0, errors)

    def get_statistics(self, original_graph: Optional['DAGGraph'] = None) -> Dict[str, Any]:
        """Compute summary statistics for the graph."""
        stats = {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "total_elements": self.total_member_count(),
            "unique_elements": len(self.unique_elements()),
            "active_pairs": self.count_active_overlap_pairs(),
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
