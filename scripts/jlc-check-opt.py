#! /usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description='The script tries different jlc optimizations to find if any of them, or a combination of them, cause errors. The cfile is first compiled without any optimizations and assumed to result in correct execution and used as a reference, which all later runs are compared against.')

parser.add_argument('cfile', 
        help='the C-file to try the optimizations on.')
parser.add_argument('-jlc', 
        help='full path to the jlc compiler (not needed if jlc is in your PATH)',
        dest='jlc',
        default='jlc', required=False)

args = parser.parse_args()

opts = ["cne", "dne", "iln", "inv", "psh", "pll", "red", "ivt", "url"]

import subprocess

def compile_and_run(command, foutput):
    print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
#    print(output.decode())
#    print(p_status)

    # If the compilation was successful and a reference output is provided
    # then try to run the compiled file
    if p_status ==0:
        cmd = "./" + args.cfile + ".out > "  + foutput
#        cmd = "./" + sys.argv[2] + ".out > "  + foutput
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
#        print(output.decode())
#        print(p_status)
        # Check if the execution exited without an error
        if p_status != 0:
            print("The execution of the compiled file returned with an error")
            exit()

def compare(freference, foutput):
    cmd = "diff " + freference + " " + foutput
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
#   print(output.decode())
#   print(p_status)
    if p_status != 0:
        print("The output differs from the reference")
        exit()


jlc = args.jlc
cfile = args.cfile
efile = args.cfile + ".out "

# We first generate a reference output
# This is done by compiling the file with no optimizations
freference = cfile + ".reference_output"
cmd = jlc + " -o " + efile + cfile
compile_and_run(cmd, freference)

# Name of the file where outputs will be written to
foutput = cfile + ".output"

# We then start with checking individual optimizations
for opt in opts:
    # Try to compile the file
    cmd = jlc + " -J" + opt + " -o " + efile + cfile
    compile_and_run(cmd, foutput)
    compare(freference, foutput)

# Next we check the optimizations for O3
O3 = ["iln",  "inv",  "red", "dne", "ivt", "inv", "dne", "psh", "inv", "dne", "red", "cne", "dne",  "pll", "inv", "dne", "url", "inv"];
for i in range(len(O3)):
    opts = ""
    for j in range(i):
        opts += " -J" + O3[j]
    cmd = jlc + opts + " -o " + efile + cfile
    compile_and_run(cmd, foutput)
    compare(freference, foutput)
