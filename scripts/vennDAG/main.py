#!/usr/bin/env python3
"""
VennDAG Command Line Interface.
Reads DOT graphs with node member sets, parses into IR, solves the minimal
reachability-preserving set reduction problem, and visualizes/outputs results.
Supports single files, multigraph streams, and folders of DOT files with percentile summaries.
"""

import sys
import os
import glob
import math
import argparse
import time
import multiprocessing
from typing import Optional, List, Dict, Tuple, Any

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


def format_time_duration(seconds: float) -> str:
    """Format duration in seconds as mm:ss or hh:mm:ss."""
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_progress_bar(
    current: int,
    total: int,
    width: int = 30,
    elapsed: float = 0.0,
    unit: str = "files"
) -> str:
    """Render a terminal progress bar string with percentage, count, speed, and ETA."""
    if total <= 0:
        pct = 1.0
    else:
        pct = min(1.0, max(0.0, current / float(total)))

    filled_len = int(width * pct)
    if filled_len >= width:
        bar = "=" * width
    elif filled_len > 0:
        bar = "=" * (filled_len - 1) + ">" + " " * (width - filled_len)
    else:
        bar = " " * width

    pct_str = f"{pct * 100.0:5.1f}%"
    count_str = f"{current}/{total} {unit}"

    if elapsed > 0 and current > 0:
        rate = current / elapsed
        rem_sec = (total - current) / rate if rate > 0 else 0
        time_str = f"[{format_time_duration(elapsed)}<{format_time_duration(rem_sec)}, {rate:5.1f} {unit}/s]"
    else:
        time_str = f"[{format_time_duration(elapsed)}<?, ? {unit}/s]"

    return f"\r[{bar}] {pct_str} ({count_str}) {time_str}"


def calculate_percentiles(data: List[float], percentiles: List[int] = [10, 25, 50, 75, 90, 95, 99]) -> Dict[str, float]:
    """Compute exact percentile ranks, max, mean, sum, and count for a list of numeric values."""
    if not data:
        res = {f"{p}th": 0.0 for p in percentiles}
        res["Max"] = 0.0
        res["Mean"] = 0.0
        res["Sum"] = 0.0
        res["Count"] = 0
        return res

    arr = sorted(data)
    n = len(arr)
    res = {
        "Count": n,
        "Sum": float(sum(arr)),
        "Mean": sum(arr) / float(n),
        "Max": float(arr[-1])
    }
    for p in percentiles:
        if n == 1:
            res[f"{p}th"] = float(arr[0])
            continue
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            res[f"{p}th"] = float(arr[int(k)])
        else:
            d = k - f
            res[f"{p}th"] = float(arr[f] + d * (arr[c] - arr[f]))
    return res


def format_percentiles_table(title: str, rows_data: List[Tuple[str, List[float]]]) -> str:
    """Format metrics into a cleanly aligned text table with percentiles, max, mean, and sum."""
    percentile_keys = [10, 25, 50, 75, 90, 95, 99]
    header_cols = ["Metric"] + [f"{p}th" for p in percentile_keys] + ["Max", "Mean", "Sum"]

    table_rows = []
    for label, data in rows_data:
        stats = calculate_percentiles(data, percentile_keys)
        row = [label]
        for p in percentile_keys:
            row.append(f"{stats[f'{p}th']:.2f}")
        row.append(f"{stats['Max']:.2f}")
        row.append(f"{stats['Mean']:.2f}")
        row.append(f"{stats['Sum']:.2f}")
        table_rows.append(row)

    col_widths = [len(h) for h in header_cols]
    for row in table_rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    header_str = "  ".join(h.ljust(col_widths[0]) if i == 0 else h.rjust(col_widths[i]) for i, h in enumerate(header_cols))
    sep_str = "  ".join("-" * col_widths[i] for i in range(len(header_cols)))

    lines = [f"=== {title} ===", header_str, sep_str]
    for row in table_rows:
        row_str = "  ".join(val.ljust(col_widths[0]) if i == 0 else val.rjust(col_widths[i]) for i, val in enumerate(row))
        lines.append(row_str)
    lines.append("")
    return "\n".join(lines)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vennDAG",
        description="VennDAG: Minimal reachability-preserving set reduction on DAGs."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input DOT file path, folder with DOT files, or stdin if omitted / '-'"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output DOT file path or directory for batch processing"
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
        "--progress",
        action="store_true",
        default=True,
        help="Display progress bar during batch folder processing (default: enabled)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable progress bar during batch folder processing"
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
        "--summary",
        action="store_true",
        default=None,
        help="Print percentile summary tables for input and output files"
    )
    parser.add_argument(
        "--no-summary",
        action="store_false",
        dest="summary",
        help="Disable printing percentile summary tables"
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
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker processes for directory processing (default: all CPU cores)"
    )
    return parser


def _process_single_file_worker(task: Tuple[str, str, bool, Optional[str], str]) -> Dict[str, Any]:
    """
    Worker task to parse, solve, and collect metrics for a single DOT file.
    Runs inside a multiprocessing worker pool.
    """
    fpath, solver, prune, output_dir, style = task
    res: Dict[str, Any] = {
        "fpath": fpath,
        "success": False,
        "error": None,
        "graphs_count": 0,
        "nodes_count": 0,
        "edges_count": 0,
        "orig_elements": 0,
        "red_elements": 0,
        "is_valid": True,
        "input_elements_per_node": [],
        "input_elements_per_graph": [],
        "input_nodes_per_graph": [],
        "input_edges_per_graph": [],
        "input_active_pairs_per_graph": [],
        "output_elements_per_node": [],
        "output_elements_per_graph": [],
        "output_nodes_per_graph": [],
        "output_edges_per_graph": [],
        "output_active_pairs_per_graph": [],
    }

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        graphs = parse_dot_graphs(content)
    except Exception as e:
        res["error"] = f"Error reading/parsing '{fpath}': {e}"
        return res

    if not graphs:
        res["success"] = True
        return res

    solved_file_graphs = [] if output_dir else None
    file_orig_elems = 0
    file_red_elems = 0

    for g in graphs:
        try:
            solved_g, stats = solve_graph(g, method=solver, prune=prune)
            if solved_file_graphs is not None:
                solved_file_graphs.append(solved_g)

            if not stats["is_valid"]:
                res["is_valid"] = False

            res["graphs_count"] += 1
            res["nodes_count"] += stats["num_nodes"]
            res["edges_count"] += stats["num_edges"]
            file_orig_elems += stats["original_elements"]
            file_red_elems += stats["reduced_elements"]

            res["input_elements_per_graph"].append(stats["original_elements"])
            res["input_nodes_per_graph"].append(stats["num_nodes"])
            res["input_edges_per_graph"].append(stats["num_edges"])
            res["input_active_pairs_per_graph"].append(stats["active_pairs"])
            for n in g.nodes.values():
                count = len(n.members)
                if count > 0:
                    res["input_elements_per_node"].append(count)

            res["output_elements_per_graph"].append(stats["reduced_elements"])
            res["output_nodes_per_graph"].append(stats["num_nodes"])
            res["output_edges_per_graph"].append(stats["num_edges"])
            res["output_active_pairs_per_graph"].append(stats["active_pairs"])
            for n in solved_g.nodes.values():
                count = len(n.members)
                if count > 0:
                    res["output_elements_per_node"].append(count)

            # Clear internal caches immediately
            g.clear_cache()
            solved_g.clear_cache()

        except Exception as e:
            res["error"] = f"Error solving graph '{g.name}' in '{fpath}': {e}"
            res["is_valid"] = False

    res["orig_elements"] = file_orig_elems
    res["red_elements"] = file_red_elems
    res["success"] = True

    if output_dir and solved_file_graphs:
        try:
            out_file_path = os.path.join(output_dir, os.path.basename(fpath))
            export_dot(solved_file_graphs, file_path=out_file_path, original_graph=graphs, mode=style)
        except Exception as e:
            res["error"] = f"Error writing output DOT for '{fpath}': {e}"
            res["is_valid"] = False

    graphs.clear()
    if solved_file_graphs is not None:
        solved_file_graphs.clear()

    return res


def process_folder(folder_path: str, args: argparse.Namespace) -> int:
    """Process all DOT files in a directory in parallel and output percentile tables."""
    dot_files = sorted(glob.glob(os.path.join(folder_path, "*.dot")))
    if not dot_files:
        # Search recursively if none found in top directory
        dot_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.dot"), recursive=True))

    if not dot_files:
        sys.stderr.write(f"Error: No .dot files found in folder '{folder_path}'.\n")
        return 1

    prune = not args.no_prune

    # Metrics collections
    input_metrics = {
        "Elements / Node": [],
        "Elements / Graph": [],
        "Nodes / Graph": [],
        "Edges / Graph": [],
        "Active Pairs / Graph": [],
        "Elements / File": [],
    }
    output_metrics = {
        "Elements / Node": [],
        "Elements / Graph": [],
        "Nodes / Graph": [],
        "Edges / Graph": [],
        "Active Pairs / Graph": [],
        "Elements / File": [],
    }

    total_graphs = 0
    total_nodes = 0
    total_edges = 0
    total_orig_elems = 0
    total_red_elems = 0
    overall_valid = True
    start_time = time.time()
    total_files = len(dot_files)
    show_progress = args.progress and not args.quiet

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if show_progress:
        sys.stderr.write(render_progress_bar(0, total_files, elapsed=0.0))
        sys.stderr.flush()

    tasks = [
        (fpath, args.solver, prune, args.output, args.style)
        for fpath in dot_files
    ]

    num_workers = args.jobs
    if num_workers is not None and num_workers <= 0:
        num_workers = None

    max_procs = os.cpu_count() or 1
    if num_workers is not None:
        actual_procs = min(num_workers, total_files)
    else:
        actual_procs = min(max_procs, total_files)

    completed_files = 0

    def handle_result(res: Dict[str, Any]) -> None:
        nonlocal total_graphs, total_nodes, total_edges, total_orig_elems, total_red_elems, overall_valid
        if res["error"]:
            if show_progress:
                sys.stderr.write("\n")
            sys.stderr.write(f"{res['error']}\n")
        if not res["is_valid"]:
            overall_valid = False
        if res["success"] and res["graphs_count"] > 0:
            total_graphs += res["graphs_count"]
            total_nodes += res["nodes_count"]
            total_edges += res["edges_count"]
            total_orig_elems += res["orig_elements"]
            total_red_elems += res["red_elements"]

            input_metrics["Elements / File"].append(res["orig_elements"])
            output_metrics["Elements / File"].append(res["red_elements"])

            input_metrics["Elements / Node"].extend(res["input_elements_per_node"])
            input_metrics["Elements / Graph"].extend(res["input_elements_per_graph"])
            input_metrics["Nodes / Graph"].extend(res["input_nodes_per_graph"])
            input_metrics["Edges / Graph"].extend(res["input_edges_per_graph"])
            input_metrics["Active Pairs / Graph"].extend(res["input_active_pairs_per_graph"])

            output_metrics["Elements / Node"].extend(res["output_elements_per_node"])
            output_metrics["Elements / Graph"].extend(res["output_elements_per_graph"])
            output_metrics["Nodes / Graph"].extend(res["output_nodes_per_graph"])
            output_metrics["Edges / Graph"].extend(res["output_edges_per_graph"])
            output_metrics["Active Pairs / Graph"].extend(res["output_active_pairs_per_graph"])

    if actual_procs <= 1 or total_files <= 1:
        for task in tasks:
            res = _process_single_file_worker(task)
            completed_files += 1
            handle_result(res)
            if show_progress:
                elapsed = time.time() - start_time
                sys.stderr.write(render_progress_bar(completed_files, total_files, elapsed=elapsed))
                sys.stderr.flush()
    else:
        chunksize = max(1, min(16, total_files // (actual_procs * 4)))
        with multiprocessing.Pool(processes=actual_procs, maxtasksperchild=50) as pool:
            for res in pool.imap_unordered(_process_single_file_worker, tasks, chunksize=chunksize):
                completed_files += 1
                handle_result(res)
                if show_progress:
                    elapsed = time.time() - start_time
                    sys.stderr.write(render_progress_bar(completed_files, total_files, elapsed=elapsed))
                    sys.stderr.flush()

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    total_time = time.time() - start_time
    total_saved = total_orig_elems - total_red_elems
    overall_pct = (total_saved / total_orig_elems * 100.0) if total_orig_elems > 0 else 0.0

    if args.stats and not args.quiet:
        print(f"=== VennDAG Batch Reduction Summary ===")
        print(f"  Processed Files:    {len(dot_files)}")
        print(f"  Total Graphs:       {total_graphs}")
        print(f"  Total Nodes:        {total_nodes}")
        print(f"  Total Edges:        {total_edges}")
        print(f"  Original Elements:  {total_orig_elems}")
        print(f"  Reduced Elements:   {total_red_elems}")
        print(f"  Elements Saved:     {total_saved} ({overall_pct:.1f}%)")
        print(f"  Total Time:         {total_time:.2f} s")
        print(f"  All Valid:          {'✓ YES' if overall_valid else '✗ NO'}")
        print(f"=======================================\n")

    # Print summary tables (always printed for directory unless --no-summary or --quiet)
    show_summary = args.summary if args.summary is not None else (not args.quiet)
    if show_summary:
        in_rows = [
            ("Elements / Node", input_metrics["Elements / Node"]),
            ("Elements / Graph", input_metrics["Elements / Graph"]),
            ("Nodes / Graph", input_metrics["Nodes / Graph"]),
            ("Edges / Graph", input_metrics["Edges / Graph"]),
            ("Active Pairs / Graph", input_metrics["Active Pairs / Graph"]),
            ("Elements / File", input_metrics["Elements / File"]),
        ]
        out_rows = [
            ("Elements / Node", output_metrics["Elements / Node"]),
            ("Elements / Graph", output_metrics["Elements / Graph"]),
            ("Nodes / Graph", output_metrics["Nodes / Graph"]),
            ("Edges / Graph", output_metrics["Edges / Graph"]),
            ("Active Pairs / Graph", output_metrics["Active Pairs / Graph"]),
            ("Elements / File", output_metrics["Elements / File"]),
        ]

        print(format_percentiles_table("Input Files Percentiles", in_rows))
        print(format_percentiles_table("Output Files Percentiles", out_rows))
        sys.stdout.flush()

    return 0 if overall_valid else 2


def main(args_list: Optional[list] = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(args_list)

    # Check if input is a directory
    if args.input and os.path.isdir(args.input):
        return process_folder(args.input, args)

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

    # Prepare percentile rows relevant for single file
    in_node_elems = [len(n.members) for g in graphs for n in g.nodes.values() if len(n.members) > 0]
    out_node_elems = [len(n.members) for g in solved_graphs for n in g.nodes.values() if len(n.members) > 0]

    in_rows = [("Elements / Node", in_node_elems)]
    out_rows = [("Elements / Node", out_node_elems)]

    if len(graphs) > 1:
        in_rows.extend([
            ("Elements / Graph", [s["original_elements"] for s in all_stats]),
            ("Nodes / Graph", [s["num_nodes"] for s in all_stats]),
            ("Edges / Graph", [s["num_edges"] for s in all_stats]),
            ("Active Pairs / Graph", [s["active_pairs"] for s in all_stats]),
        ])
        out_rows.extend([
            ("Elements / Graph", [s["reduced_elements"] for s in all_stats]),
            ("Nodes / Graph", [s["num_nodes"] for s in all_stats]),
            ("Edges / Graph", [s["num_edges"] for s in all_stats]),
            ("Active Pairs / Graph", [s["active_pairs"] for s in all_stats]),
        ])

    # Summary tables if explicitly requested on stdout
    if args.summary:
        print(format_percentiles_table("Input File Percentiles", in_rows))
        print(format_percentiles_table("Output File Percentiles", out_rows))
        sys.stdout.flush()

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
            sys.stderr.write(format_percentiles_table("Input File Percentiles", in_rows) + "\n")
            sys.stderr.write(format_percentiles_table("Output File Percentiles", out_rows) + "\n\n")
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
            sys.stderr.write(format_percentiles_table("Input File Percentiles", in_rows) + "\n")
            sys.stderr.write(format_percentiles_table("Output File Percentiles", out_rows) + "\n\n")

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
