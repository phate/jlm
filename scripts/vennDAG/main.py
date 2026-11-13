#!/usr/bin/env python3
"""
VennDAG Command Line Interface.
Reads a DOT graph with node member sets, parses into IR, solves the minimal
reachability-preserving set reduction problem, and visualizes/outputs results.
"""

import sys
import os
import argparse
from typing import Optional

# Support importing within package or as standalone script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vennDAG.ir import DAGGraph
    from vennDAG.dot_parser import parse_dot, parse_dot_graphs
    from vennDAG.dot_writer import export_dot, render_graph
    from vennDAG.solvers import solve_graph
else:
    from .ir import DAGGraph
    from .dot_parser import parse_dot, parse_dot_graphs
    from .dot_writer import export_dot, render_graph
    from .solvers import solve_graph


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vennDAG",
        description="VennDAG: Minimal reachability-preserving set reduction on DAGs."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input DOT file path (or read from stdin if omitted or '-')"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output DOT file path (prints to stdout if omitted)"
    )
    parser.add_argument(
        "--render",
        default=None,
        metavar="IMAGE_PATH",
        help="Render visualization image (e.g., output.svg, output.png, output.pdf) using Graphviz dot"
    )
    parser.add_argument(
        "--solver",
        choices=["greedy", "exact"],
        default="greedy",
        help="Reduction algorithm (default: 'greedy')"
    )
    parser.add_argument(
        "--style",
        choices=["standard", "compare", "minimal"],
        default="compare",
        help="DOT visualization style: 'compare' (highlights changes), 'standard', or 'minimal' (default: 'compare')"
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable post-greedy redundant element pruning pass"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Display reduction and solver statistics (default: enabled)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_false",
        dest="stats",
        help="Disable printing statistics to stderr"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode: suppress all status/stats messages"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title for generated DOT/image visualization"
    )
    return parser


def main(args_list: Optional[list] = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(args_list)

    # 1. Read input
    if args.input is None or args.input == "-":
        if sys.stdin.isatty():
            parser.print_help(sys.stderr)
            return 1
        dot_content = sys.stdin.read()
    else:
        if not os.path.exists(args.input):
            sys.stderr.write(f"Error: Input file '{args.input}' not found.\n")
            return 1
        with open(args.input, "r", encoding="utf-8") as f:
            dot_content = f.read()

    # 2. Parse into IR
    try:
        graphs = parse_dot_graphs(dot_content)
    except Exception as e:
        sys.stderr.write(f"Error parsing DOT graph: {e}\n")
        return 1

    if not graphs:
        sys.stderr.write("Error: No valid graphs found in DOT input.\n")
        return 1

    # 3. Solve each graph
    prune = not args.no_prune
    solved_graphs = []
    all_stats = []
    overall_valid = True

    for i, g in enumerate(graphs):
        try:
            solved_g, stats = solve_graph(g, method=args.solver, prune=prune)
            solved_graphs.append(solved_g)
            all_stats.append(stats)
            if not stats["is_valid"]:
                overall_valid = False
        except Exception as e:
            sys.stderr.write(f"Error solving graph '{g.name}' (index {i}): {e}\n")
            return 1

    # 4. Report stats
    if args.stats and not args.quiet:
        if len(graphs) == 1:
            stats = all_stats[0]
            sys.stderr.write("\n=== VennDAG Reduction Results ===\n")
            sys.stderr.write(f"  Graph:              {graphs[0].name}\n")
            sys.stderr.write(f"  Solver:             {stats['solver']}\n")
            sys.stderr.write(f"  Nodes:              {stats['num_nodes']}\n")
            sys.stderr.write(f"  Edges:              {stats['num_edges']}\n")
            sys.stderr.write(f"  Active Pairs:       {stats['active_pairs']}\n")
            sys.stderr.write(f"  Original Elements:  {stats['original_elements']}\n")
            sys.stderr.write(f"  Reduced Elements:   {stats['reduced_elements']}\n")
            sys.stderr.write(f"  Elements Saved:     {stats['elements_saved']} ({stats['reduction_percentage']:.1f}%)\n")
            sys.stderr.write(f"  Solve Time:         {stats['solve_time_sec'] * 1000.0:.2f} ms\n")
            sys.stderr.write(f"  Solution Valid:     {'✓ YES' if stats['is_valid'] else '✗ NO'}\n")
            if not stats["is_valid"]:
                for err in stats["validation_errors"]:
                    sys.stderr.write(f"    - {err}\n")
            sys.stderr.write("=================================\n\n")
        else:
            total_nodes = sum(s['num_nodes'] for s in all_stats)
            total_edges = sum(s['num_edges'] for s in all_stats)
            total_pairs = sum(s['active_pairs'] for s in all_stats)
            total_orig = sum(s['original_elements'] for s in all_stats)
            total_red = sum(s['reduced_elements'] for s in all_stats)
            total_saved = sum(s['elements_saved'] for s in all_stats)
            total_time = sum(s['solve_time_sec'] for s in all_stats)
            overall_pct = (total_saved / total_orig * 100.0) if total_orig > 0 else 0.0

            sys.stderr.write(f"\n=== VennDAG Reduction Results ({len(graphs)} graphs) ===\n")
            for i, (g, s) in enumerate(zip(graphs, all_stats)):
                valid_str = '✓' if s['is_valid'] else '✗'
                sys.stderr.write(
                    f"  [{i+1}/{len(graphs)}] '{g.name}': {s['original_elements']} -> {s['reduced_elements']} elements "
                    f"({s['elements_saved']} saved, {s['reduction_percentage']:.1f}%) in {s['solve_time_sec']*1000.0:.2f}ms [{valid_str}]\n"
                )
                if not s["is_valid"]:
                    for err in s["validation_errors"]:
                        sys.stderr.write(f"      - {err}\n")
            sys.stderr.write("  ---------------------------------\n")
            sys.stderr.write(f"  Total Graphs:       {len(graphs)}\n")
            sys.stderr.write(f"  Total Nodes:        {total_nodes}\n")
            sys.stderr.write(f"  Total Edges:        {total_edges}\n")
            sys.stderr.write(f"  Total Active Pairs: {total_pairs}\n")
            sys.stderr.write(f"  Original Elements:  {total_orig}\n")
            sys.stderr.write(f"  Reduced Elements:   {total_red}\n")
            sys.stderr.write(f"  Total Saved:        {total_saved} ({overall_pct:.1f}%)\n")
            sys.stderr.write(f"  Total Solve Time:   {total_time * 1000.0:.2f} ms\n")
            sys.stderr.write(f"  All Valid:          {'✓ YES' if overall_valid else '✗ NO'}\n")
            sys.stderr.write("===============================================\n\n")

    # 5. Output DOT
    if len(graphs) == 1:
        s = all_stats[0]
        title = args.title or f"VennDAG ({s['reduction_percentage']:.1f}% reduced: {s['original_elements']} -> {s['reduced_elements']})"
    else:
        title = args.title

    output_dot = export_dot(
        solved_graphs,
        file_path=args.output,
        original_graph=graphs,
        mode=args.style,
        title=title
    )

    if not args.output:
        # If no output file specified and not rendering only, print DOT to stdout
        if not args.render:
            sys.stdout.write(output_dot)

    # 6. Render image if requested
    if args.render:
        try:
            render_path = render_graph(
                solved_graphs,
                output_path=args.render,
                original_graph=graphs,
                mode=args.style,
                title=title
            )
            if not args.quiet:
                sys.stderr.write(f"Visualization rendered to: {render_path}\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to render image with Graphviz: {e}\n")

    return 0 if overall_valid else 2


if __name__ == "__main__":
    sys.exit(main())
