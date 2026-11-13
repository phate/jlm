"""
DOT serializer and visualization generator for VennDAG.
Formats DAGGraph into DOT representation with visual styling options and image rendering.
"""

import subprocess
import shutil
import html
from typing import Dict, Set, List, Optional, Union, Sequence
from .ir import DAGGraph


class DotWriter:
    """Serializes one or more DAGGraph instances into DOT format with optional visual styling."""

    def __init__(self,
                 graphs: Union[DAGGraph, Sequence[DAGGraph]],
                 original_graphs: Optional[Union[DAGGraph, Sequence[DAGGraph], Dict[str, DAGGraph]]] = None):
        if isinstance(graphs, DAGGraph):
            self.graphs: List[DAGGraph] = [graphs]
            self.single_mode = True
        else:
            self.graphs = list(graphs)
            self.single_mode = False

        self.original_graphs = original_graphs

    def _get_orig_graph(self, idx: int, graph: DAGGraph) -> Optional[DAGGraph]:
        if self.original_graphs is None:
            return None
        if isinstance(self.original_graphs, dict):
            return self.original_graphs.get(graph.name)
        if isinstance(self.original_graphs, (list, tuple)):
            if idx < len(self.original_graphs):
                return self.original_graphs[idx]
            return None
        if isinstance(self.original_graphs, DAGGraph):
            return self.original_graphs
        return None

    def _graph_to_dot(self, graph: DAGGraph, orig_graph: Optional[DAGGraph],
                      mode: str = "standard", title: Optional[str] = None) -> str:
        lines = []
        graph_type = "digraph" if graph.is_directed else "graph"
        edge_op = "->" if graph.is_directed else "--"
        lines.append(f'{graph_type} "{graph.name}" {{')

        # Graph-level styling
        lines.append('  rankdir=TB;')
        lines.append('  node [fontname="Helvetica,Arial,sans-serif", fontsize=11, shape=box, style="filled,rounded", fillcolor="#F8FAFC", color="#64748B", penwidth=1.5];')
        lines.append('  edge [fontname="Helvetica,Arial,sans-serif", fontsize=9, color="#475569", penwidth=1.2, arrowsize=0.8];')

        if title:
            lines.append(f'  labelloc="t";')
            escaped_title = html.escape(title)
            lines.append(f'  label=<<FONT POINT-SIZE="14"><B>{escaped_title}</B></FONT>>;')

        # Nodes
        for nid, node in graph.nodes.items():
            members_sorted = sorted(list(node.members))
            members_str = ", ".join(members_sorted)
            members_attr_val = ",".join(members_sorted)

            if mode == "compare" and orig_graph and nid in orig_graph.nodes:
                orig_node = orig_graph.nodes[nid]
                orig_sorted = sorted(list(orig_node.members))
                retained = sorted(list(node.members))
                removed = sorted(list(orig_node.members - node.members))

                # Color coding based on reduction
                if removed:
                    fillcolor = "#ECFDF5"  # Light emerald/green
                    bordercolor = "#059669"
                else:
                    fillcolor = "#F8FAFC"
                    bordercolor = "#64748B"

                # HTML-like table label for comparison
                orig_str = html.escape("{" + ", ".join(orig_sorted) + "}" if orig_sorted else "∅")
                red_str = html.escape("{" + ", ".join(retained) + "}" if retained else "∅")
                escaped_nid = html.escape(str(nid))

                html_label = (
                    f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">'
                    f'<TR><TD><B>{escaped_nid}</B></TD></TR>'
                    f'<HR/>'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="#64748B">Initial: {orig_str}</FONT></TD></TR>'
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="{bordercolor}"><B>Reduced: {red_str}</B></FONT></TD></TR>'
                )
                if removed:
                    rem_str = html.escape("{" + ", ".join(removed) + "}")
                    html_label += f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="#DC2626"><S>Removed: {rem_str}</S></FONT></TD></TR>'
                html_label += '</TABLE>>'

                lines.append(
                    f'  "{nid}" [label={html_label}, fillcolor="{fillcolor}", color="{bordercolor}", members="{members_attr_val}"];'
                )

            elif mode == "minimal":
                lines.append(f'  "{nid}" [members="{members_attr_val}"];')

            else:  # standard
                set_repr = "{" + members_str + "}" if members_str else "∅"
                label_val = f"{nid}\\n{set_repr}"
                lines.append(
                    f'  "{nid}" [label="{label_val}", members="{members_attr_val}"];'
                )

        # Edges
        for edge in graph.edges:
            attr_parts = []
            for k, v in edge.attributes.items():
                if k not in ('label', 'members'):
                    attr_parts.append(f'{k}="{v}"')
            attr_str = f" [{', '.join(attr_parts)}]" if attr_parts else ""
            lines.append(f'  "{edge.src}" {edge_op} "{edge.dst}"{attr_str};')

        lines.append("}\n")
        return "\n".join(lines)

    def to_dot(self, mode: str = "standard", title: Optional[str] = None) -> str:
        """
        Export graph(s) to DOT format string.
        mode can be:
        - 'standard': Clean dot graph with members attribute and member label.
        - 'compare': Rich visual comparison showing initial vs reduced members.
        - 'minimal': Simple DOT with members attribute only.
        """
        chunks = []
        for idx, g in enumerate(self.graphs):
            orig_g = self._get_orig_graph(idx, g)
            t = title if (self.single_mode or idx == 0) else None
            chunks.append(self._graph_to_dot(g, orig_g, mode=mode, title=t))
        return "\n\n".join(chunks)

    def write_to_file(self, file_path: str, mode: str = "standard", title: Optional[str] = None) -> None:
        """Write DOT representation to file."""
        dot_str = self.to_dot(mode=mode, title=title)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(dot_str)

    def render(self, output_path: str, format: Optional[str] = None,
               mode: str = "standard", title: Optional[str] = None) -> str:
        """
        Render graph(s) to image file (SVG, PNG, PDF) using Graphviz `dot` CLI.
        Returns the output path on success.
        """
        if not shutil.which("dot"):
            raise RuntimeError("Graphviz 'dot' executable not found on system PATH.")

        if format is None:
            # Infer format from extension
            ext = output_path.split(".")[-1].lower()
            format = ext if ext in ("svg", "png", "pdf", "jpg", "jpeg") else "svg"

        dot_content = self.to_dot(mode=mode, title=title)
        cmd = ["dot", f"-T{format}", f"-o{output_path}"]

        proc = subprocess.run(
            cmd,
            input=dot_content.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )

        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Graphviz dot rendering failed (exit code {proc.returncode}):\n{err_msg}")

        return output_path


def export_dot(graphs: Union[DAGGraph, Sequence[DAGGraph]],
               file_path: Optional[str] = None,
               original_graph: Optional[Union[DAGGraph, Sequence[DAGGraph], Dict[str, DAGGraph]]] = None,
               mode: str = "standard", title: Optional[str] = None) -> str:
    """Convenience function to export DAGGraph(s) to DOT string or file."""
    writer = DotWriter(graphs, original_graphs=original_graph)
    dot_str = writer.to_dot(mode=mode, title=title)
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(dot_str)
    return dot_str


def render_graph(graphs: Union[DAGGraph, Sequence[DAGGraph]],
                 output_path: str,
                 original_graph: Optional[Union[DAGGraph, Sequence[DAGGraph], Dict[str, DAGGraph]]] = None,
                 format: Optional[str] = None,
                 mode: str = "standard", title: Optional[str] = None) -> str:
    """Convenience function to render DAGGraph(s) directly to image."""
    writer = DotWriter(graphs, original_graphs=original_graph)
    return writer.render(output_path, format=format, mode=mode, title=title)
