import sys

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
NC = '\033[0m' # No Color
prefixes = ["==", "end", "begin"]

tolerance = 0.01
def main():
    success = True
    with open(sys.argv[1], "r") as f1, open(sys.argv[2], "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        if len(lines1) != len(lines2):
            print(f"Files have different number of lines: {len(lines1)} vs {len(lines2)}")
            success = False
        else:
            for line_num, (line1, line2) in enumerate(zip(lines1, lines2), 1):
                if any(line1.strip().startswith(prefix) for prefix in prefixes):
                    continue
                if not success:
                    break
                floats1 = [float(x) for x in line1.strip().split()]
                floats2 = [float(x) for x in line2.strip().split()]
                if len(floats1) != len(floats2):
                    print(f"Line {line_num}: Different number of values ({len(floats1)} vs {len(floats2)})")
                    success = False
                    break
                for f_val1, f_val2 in zip(floats1, floats2):
                    if f_val1 == 0 and f_val2 == 0:
                        continue
                    error = abs(f_val1 - f_val2) / max(abs(f_val1), abs(f_val2))
                    if error > tolerance:
                        print(f"Line {line_num}: Different values ({f_val1} vs {f_val2}), relative error {error}")
                        success = False
                        break
    if success:
        print(f"    {GREEN}SUCCESS{NC}")
        return 0
    else: # Explicitly handle failure case for printing
        print(f"    {RED}FAILURE{NC}")
    return 1

if __name__ == "__main__":
    main()