"""
Comprehensive test suite for VennDAG.
Tests IR, DOT parser, DOT serializer, graph visualization, greedy solver, exact MILP solver,
and edge cases (cycles, empty graphs, disconnected components, etc.).
"""

import unittest
import os
import tempfile
import sys

# Ensure package can be imported
pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from vennDAG.ir import DAGGraph, Node, Edge
from vennDAG.dot_parser import parse_dot, parse_dot_graphs, DotParser
from vennDAG.dot_writer import export_dot, render_graph, DotWriter
from vennDAG.solvers import solve_graph, GreedySolver, MilpSolver
from vennDAG.main import main as cli_main


class TestVennDAGIR(unittest.TestCase):
    def test_graph_creation_and_reachability(self):
        g = DAGGraph(name="test")
        g.add_node("A", members={"x", "y"})
        g.add_node("B", members={"y", "z"})
        g.add_node("C", members={"x", "w"})
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        self.assertTrue(g.is_acyclic())
        self.assertEqual(g.topological_sort(), ["A", "B", "C"])

        reach = g.compute_reachability()
        self.assertEqual(reach["A"], {"B", "C"})
        self.assertEqual(reach["B"], {"C"})
        self.assertEqual(reach["C"], set())

        active_pairs = g.get_active_overlap_pairs()
        # (A, B) overlap on {y}
        # (A, C) overlap on {x}
        # (B, C) no overlap
        self.assertEqual(len(active_pairs), 2)
        pairs_dict = {(u, v): common for u, v, common in active_pairs}
        self.assertEqual(pairs_dict[("A", "B")], {"y"})
        self.assertEqual(pairs_dict[("A", "C")], {"x"})

    def test_cycle_detection(self):
        g = DAGGraph()
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")
        self.assertFalse(g.is_acyclic())
        with self.assertRaises(ValueError):
            g.topological_sort()


class TestDotParser(unittest.TestCase):
    def test_parse_simple_dot(self):
        dot_code = """
        digraph Sample {
            // comment
            n1 [members="a, b", label="Node 1"];
            n2 [members="b, c"];
            /* multi line
               comment */
            n1 -> n2 [color=red];
        }
        """
        g = parse_dot(dot_code)
        self.assertEqual(g.name, "Sample")
        self.assertIn("n1", g.nodes)
        self.assertIn("n2", g.nodes)
        self.assertEqual(g.nodes["n1"].members, {"a", "b"})
        self.assertEqual(g.nodes["n2"].members, {"b", "c"})
        self.assertEqual(len(g.edges), 1)
        self.assertEqual(g.edges[0].src, "n1")
        self.assertEqual(g.edges[0].dst, "n2")

    def test_parse_multiple_graphs(self):
        dot_code = """
        digraph G1 {
            a [members="1, 2"];
            b [members="2, 3"];
            a -> b;
        }
        digraph G2 {
            c [members="X, Y"];
            d [members="Y, Z"];
            c -> d;
        }
        digraph G3 {
            e [members="M"];
        }
        """
        graphs = parse_dot_graphs(dot_code)
        self.assertEqual(len(graphs), 3)
        self.assertEqual(graphs[0].name, "G1")
        self.assertEqual(graphs[1].name, "G2")
        self.assertEqual(graphs[2].name, "G3")
        self.assertEqual(graphs[0].nodes["a"].members, {"1", "2"})
        self.assertEqual(graphs[1].nodes["c"].members, {"X", "Y"})
        self.assertEqual(graphs[2].nodes["e"].members, {"M"})

        # Test parse_dot with multiple=True
        graphs_via_parse_dot = parse_dot(dot_code, multiple=True)
        self.assertEqual(len(graphs_via_parse_dot), 3)

        # Test parse_dot with multiple=False returns first graph
        g1 = parse_dot(dot_code, multiple=False)
        self.assertEqual(g1.name, "G1")

    def test_parse_members_from_label(self):
        dot_code = """
        digraph {
            A [label="NodeA\\n{x, y}"];
            B [label="NodeB {y, z}"];
            A -> B;
        }
        """
        g = parse_dot(dot_code)
        self.assertEqual(g.nodes["A"].members, {"x", "y"})
        self.assertEqual(g.nodes["B"].members, {"y", "z"})

    def test_parse_edge_chains(self):
        dot_code = """
        digraph {
            1 [members="a"] -> 2 [members="a"] -> 3 [members="a"];
        }
        """
        g = parse_dot(dot_code)
        self.assertEqual(len(g.edges), 2)
        self.assertEqual(g.edges[0].src, "1")
        self.assertEqual(g.edges[0].dst, "2")
        self.assertEqual(g.edges[1].src, "2")
        self.assertEqual(g.edges[1].dst, "3")


class TestSolvers(unittest.TestCase):
    def test_greedy_solver_simple_chain(self):
        # n1: {A, B}, n2: {A, B, C}, n3: {B, C}
        # Paths: n1->n2 (overlap {A,B}), n2->n3 (overlap {B,C}), n1->n3 (overlap {B})
        # Picking B for n1, n2, n3 satisfies all 3 pairs with 3 total elements!
        g = DAGGraph()
        g.add_node("n1", members={"A", "B"})
        g.add_node("n2", members={"A", "B", "C"})
        g.add_node("n3", members={"B", "C"})
        g.add_edge("n1", "n2")
        g.add_edge("n2", "n3")

        solved_g, stats = solve_graph(g, method="greedy", prune=True)
        self.assertTrue(stats["is_valid"])
        self.assertEqual(solved_g.nodes["n1"].members, {"B"})
        self.assertEqual(solved_g.nodes["n2"].members, {"B"})
        self.assertEqual(solved_g.nodes["n3"].members, {"B"})
        self.assertEqual(solved_g.total_member_count(), 3)
        self.assertEqual(stats["original_elements"], 7)
        self.assertEqual(stats["reduced_elements"], 3)
        self.assertEqual(stats["elements_saved"], 4)

    def test_milp_solver_optimality(self):
        g = DAGGraph()
        g.add_node("n1", members={"A", "B"})
        g.add_node("n2", members={"A", "B", "C"})
        g.add_node("n3", members={"B", "C"})
        g.add_edge("n1", "n2")
        g.add_edge("n2", "n3")

        solved_g, stats = solve_graph(g, method="exact")
        self.assertTrue(stats["is_valid"])
        self.assertEqual(solved_g.total_member_count(), 3)

    def test_diamond_graph(self):
        # top: {x, y}, left: {x}, right: {y}, bottom: {x, y}
        # top -> left -> bottom
        # top -> right -> bottom
        # active pairs: (top, left, {x}), (top, right, {y}), (top, bottom, {x, y}),
        #               (left, bottom, {x}), (right, bottom, {y})
        g = DAGGraph()
        g.add_node("top", members={"x", "y"})
        g.add_node("left", members={"x"})
        g.add_node("right", members={"y"})
        g.add_node("bottom", members={"x", "y"})
        g.add_edge("top", "left")
        g.add_edge("top", "right")
        g.add_edge("left", "bottom")
        g.add_edge("right", "bottom")

        solved_g, stats = solve_graph(g, method="greedy")
        self.assertTrue(stats["is_valid"])
        # All required pairs must be satisfied:
        is_valid, errors = solved_g.verify_solution(g)
        self.assertTrue(is_valid, msg=f"Validation errors: {errors}")
        self.assertIn("x", solved_g.nodes["top"].members)
        self.assertIn("y", solved_g.nodes["top"].members)
        self.assertIn("x", solved_g.nodes["bottom"].members)
        self.assertIn("y", solved_g.nodes["bottom"].members)

    def test_no_overlap_graph(self):
        g = DAGGraph()
        g.add_node("n1", members={"A"})
        g.add_node("n2", members={"B"})
        g.add_edge("n1", "n2")

        solved_g, stats = solve_graph(g, method="greedy")
        self.assertTrue(stats["is_valid"])
        self.assertEqual(solved_g.total_member_count(), 0)

    def test_disconnected_components(self):
        g = DAGGraph()
        g.add_node("a1", members={"X", "Y"})
        g.add_node("a2", members={"X"})
        g.add_edge("a1", "a2")

        g.add_node("b1", members={"M", "N"})
        g.add_node("b2", members={"M"})
        g.add_edge("b1", "b2")

        solved_g, stats = solve_graph(g, method="greedy")
        self.assertTrue(stats["is_valid"])
        self.assertEqual(solved_g.nodes["a1"].members, {"X"})
        self.assertEqual(solved_g.nodes["a2"].members, {"X"})
        self.assertEqual(solved_g.nodes["b1"].members, {"M"})
        self.assertEqual(solved_g.nodes["b2"].members, {"M"})


class TestDotWriter(unittest.TestCase):
    def test_export_dot_standard_and_compare(self):
        g = DAGGraph(name="test")
        g.add_node("n1", members={"A", "B"})
        g.add_node("n2", members={"B"})
        g.add_edge("n1", "n2")

        orig_g = DAGGraph(name="test")
        orig_g.add_node("n1", members={"A", "B", "C"})
        orig_g.add_node("n2", members={"B", "D"})
        orig_g.add_edge("n1", "n2")

        dot_std = export_dot(g, mode="standard")
        self.assertIn('"n1"', dot_std)
        self.assertIn('"n2"', dot_std)
        self.assertIn('members="A,B"', dot_std)

        dot_cmp = export_dot(g, original_graph=orig_g, mode="compare")
        self.assertIn("Initial:", dot_cmp)
        self.assertIn("Reduced:", dot_cmp)
        self.assertIn("Removed:", dot_cmp)

    def test_export_multiple_graphs(self):
        g1 = DAGGraph(name="G1")
        g1.add_node("a", members={"A"})
        g2 = DAGGraph(name="G2")
        g2.add_node("b", members={"B"})

        dot_out = export_dot([g1, g2], mode="standard")
        self.assertIn('digraph "G1"', dot_out)
        self.assertIn('digraph "G2"', dot_out)
        self.assertIn('"a"', dot_out)
        self.assertIn('"b"', dot_out)

    def test_render_svg(self):
        g = DAGGraph(name="render_test")
        g.add_node("n1", members={"A"})
        g.add_node("n2", members={"A"})
        g.add_edge("n1", "n2")

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tf:
            svg_path = tf.name

        try:
            render_graph(g, output_path=svg_path, format="svg")
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<svg", content)
        finally:
            if os.path.exists(svg_path):
                os.remove(svg_path)

    def test_render_multiple_graphs_svg(self):
        g1 = DAGGraph(name="G1")
        g1.add_node("a", members={"A"})
        g2 = DAGGraph(name="G2")
        g2.add_node("b", members={"B"})

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tf:
            svg_path = tf.name

        try:
            render_graph([g1, g2], output_path=svg_path, format="svg")
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<svg", content)
        finally:
            if os.path.exists(svg_path):
                os.remove(svg_path)


class TestCliMultiGraph(unittest.TestCase):
    def test_cli_multi_graph(self):
        multi_dot = """
        digraph G1 {
            n1 [members="A, B"];
            n2 [members="B, C"];
            n1 -> n2;
        }
        digraph G2 {
            x1 [members="X, Y"];
            x2 [members="Y, Z"];
            x1 -> x2;
        }
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as tf_in:
            tf_in.write(multi_dot)
            input_path = tf_in.name

        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as tf_out:
            output_path = tf_out.name

        try:
            ret = cli_main([input_path, "-o", output_path, "--quiet"])
            self.assertEqual(ret, 0)

            with open(output_path, "r", encoding="utf-8") as f:
                out_content = f.read()

            parsed = parse_dot_graphs(out_content)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0].name, "G1")
            self.assertEqual(parsed[1].name, "G2")
            self.assertEqual(parsed[0].nodes["n1"].members, {"B"})
            self.assertEqual(parsed[0].nodes["n2"].members, {"B"})
            self.assertEqual(parsed[1].nodes["x1"].members, {"Y"})
            self.assertEqual(parsed[1].nodes["x2"].members, {"Y"})
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
