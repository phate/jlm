#! /usr/bin/env python3

import subprocess
import sys

# Run jlc with -### to get the individual commands to be executed
jlcArguments = ['jlc'] + sys.argv[1:] + ['-###']
jlcOutput = subprocess.run(jlcArguments, capture_output=True, text=True)

# List of arguments for jlm-opt when generating the mlir output
mlirArguments = []
# Since the mlir is written to the output we keep the original file path for when generating llvm
outputFile = ""
# We know that the output is the argument after '-o', this is a flag to know that '-o' has been seen
outputFileNext = False
# We only apply the transformation to jlm-opt that has both the input and output set to llvm
llvmInputFormat = False
llvmOutputFormat = False

for command in jlcOutput.stdout.splitlines():
    commandArguments = command.split()
    if commandArguments[0] =='jlm-opt':
        for argument in commandArguments:
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
            output = subprocess.run(mlirArguments, capture_output=True, text=True)
            if output.stderr:
                print(output.stderr)
                exit(1)
            # All optimizations etc. have already been performed when jlm-opt was 
            # invoked to generat the mlir, so only convert the mlir file to llvm.
            llArguments = ['jlm-opt', '--input-format=mlir', '--output-format=llvm', '-o', outputFile, outputFile + '.mlir']
            output = subprocess.run(llArguments, capture_output=True, text=True)
            if output.stderr:
                print(output.stderr)
                exit(1)
        else:
            output = subprocess.run(commandArguments, capture_output=True, text=True)
    else:
        # Else we run the command as is
        output = subprocess.run(commandArguments, capture_output=True, text=True)