#! /usr/bin/env python3

import subprocess
import sys

DEBUG = True

def run(arguments, **kwargs):
    if DEBUG:
        print(" ".join(arguments))
    return subprocess.run(arguments, capture_output=True, text=True, check=True, **kwargs)

def run_jlm_opt(arguments):
    # List of arguments for jlm-opt when generating the mlir output
    mlirArguments = []
    # Since the mlir is written to the output we keep the original file path for when generating llvm
    outputFile = ""
    # We know that the output is the argument after '-o', this is a flag to know that '-o' has been seen
    outputFileNext = False
    # We only apply the transformation to jlm-opt that has both the input and output set to llvm
    llvmInputFormat = False
    llvmOutputFormat = False

    for argument in arguments:
        if argument == '--input-format=llvm':
            llvmInputFormat = True
            mlirArguments = mlirArguments + ['--input-format=llvm']
        elif argument == '--output-format=llvm':
            llvmOutputFormat = True
            mlirArguments = mlirArguments + ['--output-format=mlir']
        elif argument == '-o':
            outputFileNext = True
            mlirArguments = mlirArguments + [argument]
        elif outputFileNext == True:
            outputFileNext = False
            outputFile = argument
            mlirArguments = mlirArguments + [outputFile + '.mlir']
        else:
            # All other arguments are passed to jlm-opt as is
            mlirArguments = mlirArguments + [argument]

        # If both the input and output format is llvm then replace the
        # jlm-opt command with two commands that roundtrip through mlir
    if llvmInputFormat and llvmOutputFormat:
        # Generate the mlir output
        run(mlirArguments)
        # All optimizations etc. have already been performed when jlm-opt was
        # invoked to generat the mlir, so only convert the mlir file to llvm.
        llArguments = ['jlm-opt', '--input-format=mlir', '--output-format=llvm', '-o', outputFile, outputFile + '.mlir']
        run(llArguments)
    else:
        run(arguments)

def main():
    # Run jlc with -### to get the individual commands to be executed
    jlcArguments = ['jlc'] + sys.argv[1:] + ['-###']
    jlcOutput = run(jlcArguments)

    for command in jlcOutput.stdout.splitlines():
        commandArguments = command.split()
        if commandArguments[0] =='jlm-opt':
            run_jlm_opt(commandArguments)
        else:
            run(commandArguments)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)