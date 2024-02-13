#include <assert.h>
#include <stdlib.h>

// test local memories and aliasing with a pseudo processor without branching/jumps

void
run(int * program, int * data)
{
  int end = 0;
  int pc = 0;
  int reg[256];
  int rv;
  do
  {
    int insn = program[pc];
    int op = insn & 0xFF;
    int rs1 = (insn >> 8) & 0xFF;
    int rs2 = (insn >> 16) & 0xFF;
    int rd = (insn >> 24) & 0xFF;
    int wr = 1;

    int irs1 = reg[rs1];
    int irs2 = reg[rs2];
    int ird = 0xDEADBEEF;
    switch (op)
    {
    case 0x00: // NOP
      wr = 0;
      break;
    case 0x01: // ADD
      ird = irs1 + irs2;
      break;
    case 0x02: // SUB
      ird = irs1 - irs2;
      break;
    case 0x03: // INC
      ird = irs1 + 1;
      break;
    case 0x04: // END
      wr = 0;
      rv = rs1;
      end = 1;
      break;
    case 0x05: // LOAD
      wr = 1;
      rv = data[rs1];
      break;
    case 0x06: // STORE
      wr = 0;
      data[rs1] = rs2;
      break;
    default: // skip
      wr = 0;
      ird = 0xC0FEBABE;
      break;
    }
    if (wr && rd)
    {
      reg[rd] = ird;
    }
    pc++;
  } while (!end);
}

int
main(int argc, char ** argv)
{
  int data[32];
  int program[12] = {
    0x01000103, // INC 1
    0x02000203, // INC 2
    0x01000103, // INC 1
    0x02000203, // INC 2
    0x01010101, // ADD 1, 1
    0x02000203, // INC 2
    0x01010101, // ADD 1, 1
    0x02000203, // INC 2
    0x01010101, // ADD 1, 1
    0x02000203, // INC 2
    0x01010101, // ADD 1, 1
    0x00000104, // END 1
  };
  run(program, data);
  // TODO: use loads and stores to check correctness
  return 0;
}
