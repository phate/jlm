#! /usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description='The script expects a file with statistics from the compilation of multiple files and will print out the average, minimum, and maximum fraction (%) that each optimization takes relative all optimizations, as well as the file requiring the longest runtime for each optimization.')

parser.add_argument('statfile',
        help='the file with statistics results to be parsed')

args = parser.parse_args()

f = open(args.statfile)

files = 0

stats = dict()
# Array: total time, max-time, max-file, %-avg, %-min, %-max, max-file
stats["inv"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["dne"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["ivt"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["cne"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["pll"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["url"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["opt"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["ano"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["cfr"] = [0,0, "max-path", 0,-1,0, "%-path"]
stats["agr"] = [0,0, "max-path", 0,-1,0, "%-path"]
#stats["opt_total"] = 0

# Variables for storing temporary stats for each optimization
# Array: numer of invocations, total time [time for individual passes]
inv = [0, 0]
dne = [0, 0, 0, 0]
ivt = [0, 0]
cne = [0, 0, 0, 0]
pll = [0, 0]
unroll = [0, 0]
opt = [0, 0] # Total time for all optimizations
ano = [0, 0] # Annotation
cfr = [0, 0] # CFR 
agr = [0, 0] # Aggregation

optimizations = {'inv': inv, 'dne': dne, 'ivt': ivt, 'cne': cne, 'pll': pll, 'url': unroll}
other_stats = {'ano': ano, 'cfr': cfr, 'agr': agr}
all_stats = optimizations.copy()
all_stats.update(other_stats)

def parse(line):
    items = line.split(" ")
    fname = None

    if items[0] == "INV":
       inv[0] += 1 
       inv[1] += int(items[5]) 
    elif items[0] == "DNE":
       dne[0] += 1 
       dne[1] += int(items[5]) + int(items[6])
       dne[2] += int(items[5])
       dne[3] += int(items[6])
    elif items[0] == "IVT":
       ivt[0] += 1 
       ivt[1] += int(items[5])
    elif items[0] == "CNE":
       cne[0] += 1 
       cne[1] += int(items[5]) + int(items[6])
       cne[2] += int(items[5])
       cne[3] += int(items[6])
    elif items[0] == "PULL":
       pll[0] += 1 
       pll[1] += int(items[3])
    elif items[0] == "UNROLL":
       unroll[0] += 1 
       unroll[1] += int(items[3])
    elif items[0] == "RVSDGOPTIMIZATION":
        return items[1]
    elif items[0] == "ANNOTATIONTIME":
        ano[0] = 1
        ano[1] += int(items[4])
    elif items[0] == "CFRTIME":
        cfr[0] = 1
        cfr[1] += int(items[4])
    elif items[0] == "AGGREGATIONTIME":
        agr[0] = 1
        agr[1] += int(items[4])
    else:
        print("Unknown case: ", items[0])
        exit()

    return None

def clear():
    total = 0
    for name in all_stats:
        clear_opt(all_stats[name])

def clear_opt(opt):
    for i in range(len(opt)):
        opt[i] = 0

def update_def(stat, name, fname, time):
    # Check if optimization did run otherwise we are done
    if stat[0] == 0:
        return
    # Calculate the average runtime of the optimization
    average = stat[1] / stat[0]
    stats[name][0] += average 
    # Max time
    if stats[name][1] < average:
        stats[name][1] = average
        stats[name][2] = fname
    # Calculate the % of total optimization time
    percentage = average / time
    # Average
    stats[name][3] += percentage
    # Min
    if stats[name][4] > percentage:
        stats[name][4] = percentage
    elif stats[name][4] == -1:
        stats[name][4] = percentage
    # Max
    if stats[name][5] < percentage:
        stats[name][5] = percentage
        stats[name][6] = fname

def print_stats(name):
    # Check if optimization has any statistics otherwise we are done
    if stats[name][0] == 0:
        return

    print(name, ":\t",
        format(stats[name][3]/files * 100, '.1f'), "%\t",
        format(stats[name][4] * 100, '.1f'), "%\t",
        format(stats[name][5] * 100, '.1f'), "%\t")
    print("\t max-time:\t", format(stats[name][1], '.0f'),
            "\tfile:\t", stats[name][2].split("llvm-test-suite.git/")[1])
    print("\t %-max:\t\t", format(stats[name][5] * 100, '.1f'), "%",
            "\tfile:\t", stats[name][6].split("llvm-test-suite.git/")[1])

for line in f:
    fname = parse(line.strip())
    # The statistics for a file ends with RVSDGOPTIMIZATION, which includes  file
    # So if a file name is returned then we know that we have all the statistics
    if fname:
        # Keep track of the number of files that has been optimized
        files += 1
        # Calculate the total time of the optimizations
        time = 0
        for name in optimizations:
            time += optimizations[name][1] / optimizations[name][0]
        # Collect the stats for each optimization
        for name in optimizations:
            update_def(optimizations[name], name, fname, time)
        opt_time = time
        # Calculate the time for all statistics that are collected
        for name in other_stats:
            time += other_stats[name][1]
        # Collect the stats for all other phases, i.e., not optimizations
        for name in other_stats:
            update_def(other_stats[name], name, fname, time)
        # Add stats for total time of all optimizations, i.e., sum of all opt
        opt[0] = 1
        opt[1] = opt_time
        update_def(opt, "opt", fname, time)
        clear()

print("name:\t avg-%\t min-%\t max-%")
print("Optimizations - % is relative to all optimizations, i.e., the sum of avg = 100%")
for name in optimizations:
    print_stats(name)

print("")
print("Other statistics - % is relative to all collected statistics, i.e., optimizations plus all other statistics")
print_stats("opt")
for name in other_stats:
    print_stats(name)

f.close()
