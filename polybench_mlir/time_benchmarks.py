#!/usr/bin/env python3
import subprocess
import os
import argparse
import csv
import sys
import time as pytime # Use pytime to avoid conflict if a module named 'time' is in the same dir

# Configuration
POLYBENCH_ROOT = "." # Script assumes it's run from usr/polybench
POLYBENCH_BUILD_REL = "build"
POLYBENCH_RESULTS_REL = "results"

POLYBENCH_SRC_FILES = [
    "linear-algebra/blas/gesummv/gesummv.c",
    "linear-algebra/blas/gemver/gemver.c",
    "linear-algebra/blas/trmm/trmm.c",
    "linear-algebra/kernels/atax/atax.c",
    "linear-algebra/kernels/mvt/mvt.c",
    "medley/floyd-warshall/floyd-warshall.c",
    "stencils/fdtd-2d/fdtd-2d.c",
    "stencils/jacobi-2d/jacobi-2d.c",
]

# BINARY_TYPES = ["polygeist", "polygeist_no_polymer", "jlm", "polygeist_jlm", "polygeist_jlm_no_polymer"]
BINARY_TYPES = ["jlm", "polygeist_jlm", "polygeist_jlm_no_polymer"]

def run_single_benchmark(binary_path, warmup_runs, timed_runs):
    """
    Runs a single benchmark binary with warm-up and timed iterations.
    Returns a list of recorded times for the timed runs, or None if critical error.
    """
    # Check if binary exists before proceeding
    if not os.path.exists(binary_path):
        print(f"  Error: Benchmark binary not found: {binary_path}", file=sys.stderr)
        return None # Indicates binary was not found

    print(f"  Warm-up runs for {os.path.normpath(binary_path)}...")
    for i in range(warmup_runs):
        try:
            # Suppress output of the benchmark itself during warm-up
            subprocess.run([binary_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"    Warning: Warm-up run {i+1} for {os.path.normpath(binary_path)} failed: {e}", file=sys.stderr)
        except FileNotFoundError: # Should have been caught above, but as a safeguard
            print(f"    Error: Benchmark binary not found during warm-up: {os.path.normpath(binary_path)}", file=sys.stderr)
            return None

    print(f"  Timed runs for {os.path.normpath(binary_path)}...")
    run_times = []
    for i in range(timed_runs):
        start_time = pytime.perf_counter()
        try:
            subprocess.run([binary_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            end_time = pytime.perf_counter()
            elapsed_time = end_time - start_time
            run_times.append(f"{elapsed_time:.6f}") # Store as string with precision
            print(f"    Run {i+1}/{timed_runs}: {elapsed_time:.6f}s")
        except subprocess.CalledProcessError as e:
            print(f"    Error: Timed run {i+1} for {os.path.normpath(binary_path)} failed: {e}", file=sys.stderr)
            run_times.append('Error') # Placeholder if run fails
        except FileNotFoundError:
             print(f"    Error: Benchmark binary not found during timed run: {os.path.normpath(binary_path)}", file=sys.stderr)
             return None # Critical failure if binary disappears mid-run

    return run_times

def main():
    parser = argparse.ArgumentParser(description="Run PolyBench benchmarks and record timings.")
    parser.add_argument(
        "--warmup-runs", type=int, default=1,
        help="Number of warm-up runs for each benchmark (default: 1)."
    )
    parser.add_argument(
        "--timed-runs", type=int, default=1,
        help="Number of timed runs for each benchmark (default: 1)."
    )
    parser.add_argument(
        "--output-file", type=str,
        default=os.path.join(POLYBENCH_ROOT, POLYBENCH_RESULTS_REL, "all_bench_timings_script.csv"),
        help="Path to the output CSV file."
    )
    args = parser.parse_args()

    polybench_build_dir = os.path.join(POLYBENCH_ROOT, POLYBENCH_BUILD_REL)
    
    # Ensure the base directory for the output file exists
    output_csv_path = os.path.abspath(args.output_file) # Make path absolute for clarity
    output_csv_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir, exist_ok=True)
        print(f"Created results directory: {output_csv_dir}")


    header = ["Benchmark", "BinaryType"] + [f"Run{i+1}" for i in range(args.timed_runs)]
    all_results_data = []

    for c_file_rel_path in POLYBENCH_SRC_FILES:
        bench_dir_in_src = os.path.dirname(c_file_rel_path) # e.g., "linear-algebra/blas/gesummv"
        bench_name = os.path.splitext(os.path.basename(c_file_rel_path))[0] # e.g., "gesummv"

        for bin_type in BINARY_TYPES:
            print(f"\nProcessing {bench_name} ({bin_type})...")
            
            # Construct binary path: e.g., ./build/linear-algebra/blas/gesummv/gesummv.polygeist
            binary_filename = f"{bench_name}.{bin_type}"
            # Path to binary is relative to build_dir, benchmark's own subdir structure is preserved
            binary_path = os.path.join(polybench_build_dir, bench_dir_in_src, binary_filename)
            
            current_run_times = run_single_benchmark(binary_path, args.warmup_runs, args.timed_runs)
            
            if current_run_times is None: # Binary not found or critical error
                times_to_write = ['N/A'] * args.timed_runs
            elif len(current_run_times) == args.timed_runs:
                times_to_write = current_run_times
            else: # Fallback for unexpected number of results
                times_to_write = (current_run_times + [''] * args.timed_runs)[:args.timed_runs]
            
            all_results_data.append([bench_name, bin_type] + times_to_write)

    try:
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_results_data)
        print(f"\nAll timing results written to {output_csv_path}")
    except IOError as e:
        print(f"\nError writing CSV file to {output_csv_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main() 