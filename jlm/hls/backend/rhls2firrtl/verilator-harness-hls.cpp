/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>

#include <sstream>

namespace jlm::hls
{

// The number of cycles before a load is ready
static constexpr int MEMORY_RESPONSE_LATENCY = 10;

std::string
ConvertToCType(const rvsdg::Type * type)
{
  if (auto t = dynamic_cast<const rvsdg::bittype *>(type))
  {
    return "int" + util::strfmt(t->nbits()) + "_t";
  }
  if (jlm::rvsdg::is<llvm::PointerType>(*type))
  {
    return "void*";
  }
  if (auto ft = dynamic_cast<const llvm::FloatingPointType *>(type))
  {
    switch (ft->size())
    {
    case llvm::fpsize::flt:
      return "float";
    case llvm::fpsize::dbl:
      return "double";
    default:
      throw std::logic_error(type->debug_string() + " not implemented!");
    }
  }
  if (auto t = dynamic_cast<const llvm::VectorType *>(type))
  {
    return ConvertToCType(&t->type()) + " __attribute__((vector_size("
         + std::to_string(JlmSize(type) / 8) + ")))";
  }
  if (auto t = dynamic_cast<const llvm::ArrayType *>(type))
  {
    return ConvertToCType(&t->element_type()) + "*";
  }

  JLM_UNREACHABLE("Unimplemented C type");
}

/**
 * Takes an HLS kernel and determines the return type of the original C function.
 * If the function did not have a return value, i.e., returns "void", nullopt is returned.
 * @param kernel the lambda node representing the kernel
 * @return the return type of the kernel as written in C, or nullopt if it has no return value.
 */
std::optional<std::string>
GetReturnTypeAsC(const rvsdg::LambdaNode & kernel)
{
  const auto & results = kernel.GetOperation().type().Results();

  if (results.empty())
    return std::nullopt;

  const auto & type = results.front();

  if (rvsdg::is<rvsdg::StateType>(type))
    return std::nullopt;

  return ConvertToCType(type.get());
}

/**
 * Takes an HLS kernel and determines the parameters of the original C function.
 * Returns a tuple, the first element of which is the number of parameters.
 * The second element is a string defining the C parameters, like "int32_t a0, void* a1, void* a2".
 * The third element is a string for calling the C function, like "a0, a1, a2".
 * @param kernel the lambda node representing the kernel
 * @return a tuple (number of parameters, string of parameters, string of call arguments)
 */
std::tuple<size_t, std::string, std::string>
GetParameterListAsC(const rvsdg::LambdaNode & kernel)
{
  size_t argument_index = 0;
  std::ostringstream parameters;
  std::ostringstream arguments;

  for (auto & argType : kernel.GetOperation().type().Arguments())
  {
    if (rvsdg::is<rvsdg::StateType>(argType))
      continue;
    if (rvsdg::is<bundletype>(argType))
      continue;

    if (argument_index != 0)
    {
      parameters << ", ";
      arguments << ", ";
    }

    parameters << ConvertToCType(argType.get()) << " a" << argument_index;
    arguments << "a" << argument_index;
    argument_index++;
  }

  return std::make_tuple(argument_index, parameters.str(), arguments.str());
}

std::string
VerilatorHarnessHLS::GetText(llvm::RvsdgModule & rm)
{
  std::ostringstream cpp;
  const auto & kernel = *get_hls_lambda(rm);
  const auto & function_name =
      dynamic_cast<llvm::LlvmLambdaOperation &>(kernel.GetOperation()).name();

  // The request and response parts of memory queues
  const auto mem_reqs = get_mem_reqs(kernel);
  const auto mem_resps = get_mem_resps(kernel);
  JLM_ASSERT(mem_reqs.size() == mem_resps.size());

  // All inputs that are not memory queues
  const auto reg_args = get_reg_args(kernel);

  // Extract info about the kernel's function signature in C
  const auto c_return_type = GetReturnTypeAsC(kernel);
  const auto [num_c_params, c_params, c_call_args] = GetParameterListAsC(kernel);

  cpp << R"(
#define TRACE_CHUNK_SIZE 100000
#define TIMEOUT 10000000

#include <verilated.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>
#ifdef FST
#include "verilated_fst_c.h"
#else
#include "verilated_vcd_c.h"
#endif
#define xstr(s) str(s)
#define str(s) #s
)" << std::endl;

  cpp << "#include \"V" << VerilogFile_.base() << ".h\"" << std::endl;
  cpp << "#define V_NAME V" << VerilogFile_.base() << std::endl;

  cpp << R"(
// ======== Global variables used for simulating the model ========
// The verilated model being simulated
V_NAME *top;

// Current simulation time, in number of cycles
uint64_t main_time = 0;

// Can be set from signal handlers, to trigger gracefull early termination
bool terminate = false;


// ======== Global variables imported from other modules ========
)";

  for (const auto arg : rm.Rvsdg().GetRootRegion().Arguments())
  {
    const auto graphImport = util::AssertedCast<llvm::GraphImport>(arg);
    cpp << "extern \"C\" char " << graphImport->Name() << ";" << std::endl;
  }
  cpp << R"(

// ======== Tracing accesses to main memory ==========
struct mem_access {
    void * addr;
    bool write;
    uint8_t width; // 2^width bytes
    void* data;

    bool operator==(const mem_access & other) const {
        return addr == other.addr && write == other.write && width == other.width && !memcmp(data, other.data, 1<<width);
    }
};

// A log of memory accesses made by the kernel
std::vector<mem_access> memory_accesses;
// Accesses to regions in this vector of (start, length) pairs are not traced
std::vector<std::pair<void*, size_t>> ignored_memory_regions;

static void ignore_memory_region(void* start, size_t length) {
    ignored_memory_regions.emplace_back(start, length);
}

static bool in_ignored_region(void* addr) {
    for (auto [start, length] : ignored_memory_regions) {
        if (addr >= start && addr < (char*)start + length)
            return true;
    }
    return false;
}

static void* instrumented_load(void* addr, uint8_t width) {
    void * data = malloc(1 << width);
    memcpy(data, addr, 1 << width);
    if (!in_ignored_region(addr))
        memory_accesses.push_back({addr, false, width, data});
    return data;
}

static void instrumented_store(void* addr, void *data, uint8_t width) {
    void * data_copy = malloc(1 << width);
    memcpy(data_copy, data, 1 << width);
    memcpy(addr, data_copy, 1 << width);
    if(!in_ignored_region(addr))
        memory_accesses.push_back({addr, true, width, data_copy});
}

uint32_t dummy_data[16] = {
        0xDEADBEE0,
        0xDEADBEE1,
        0xDEADBEE2,
        0xDEADBEE3,
        0xDEADBEE4,
        0xDEADBEE5,
        0xDEADBEE6,
        0xDEADBEE7,
        0xDEADBEE8,
        0xDEADBEE9,
        0xDEADBEEA,
        0xDEADBEEB,
        0xDEADBEEC,
        0xDEADBEED,
        0xDEADBEEE,
        0xDEADBEEF,
};
// ======== Implementation of external memory queues, adding latency to loads ========
class MemoryQueue {
    struct Response {
        uint64_t request_time;
        void* data;
        uint8_t size;
        uint8_t id;
    };
    int latency;
    int width;
    std::deque<Response> responses;

public:
    MemoryQueue(int latency, int width) : latency(latency), width(width) {}

    // Called right before posedge, can only read from the model
    void accept_request(uint8_t req_ready, uint8_t req_valid, uint8_t req_write, uint64_t req_addr, uint8_t req_size, void* req_data, uint8_t req_id, uint8_t res_valid, uint8_t res_ready) {
        if (top->reset) {
            responses.clear();
            return;
        }

        // If a response was consumed this cycle, remove it
        if (res_ready && res_valid) {
          assert(!responses.empty());
          responses.pop_front();
        }

        if (!req_ready || !req_valid)
            return;

        if (req_write) {
            // Stores are performed immediately
            instrumented_store((void*) req_addr, req_data, req_size);
        } else {
            // Loads are performed immediately, but their response is placed in the queue
            void* data = instrumented_load((void*) req_addr, req_size);
            responses.push_back({main_time, data, req_size, req_id});
        }
    }

    // Called right after posedge, can only write to the model
    void produce_response(uint8_t& req_ready, uint8_t& res_valid, void* res_data, uint8_t& res_id) {
        if (!responses.empty() && responses.front().request_time + latency <= main_time + 1) {
            res_valid = 1;
            memcpy(res_data, responses.front().data, 1<<responses.front().size);
            res_id = responses.front().id;
        } else {
            res_valid = 0;
            memcpy(res_data, dummy_data, width);
            res_id = 0;
        }

        // Always ready for requests
        req_ready = 1;
    }

    bool empty() const {
      return responses.empty();
    }
};
)" << std::endl;

  cpp << "MemoryQueue memory_queues[] = {";
  for (size_t i = 0; i < mem_reqs.size(); i++)
  {
    auto bundle = dynamic_cast<const bundletype *>(mem_resps[i]->Type().get());
    auto size = JlmSize(&*bundle->get_element_type("data")) / 8;
    //    int width =
    cpp << "{" << MEMORY_RESPONSE_LATENCY << ", " << size << "}, ";
  }
  cpp << "};" << R"(

// ======== Variables and functions for tracing the verilated model ========
#ifdef TRACE_SIGNALS
#ifdef FST
VerilatedFstC *tfp;
#else
VerilatedVcdC *tfp;
#endif
#endif

static void init_tracing() {
    #ifdef TRACE_SIGNALS
    #ifdef FST
    tfp = new VerilatedFstC;
    top->trace(tfp, 99);   // Trace 99 levels of hierarchy
    tfp->open(xstr(V_NAME) ".fst");
    #else
    tfp = new VerilatedVcdC;
    top->trace(tfp, 99);   // Trace 99 levels of hierarchy
    tfp->open(xstr(V_NAME) ".vcd");
    #endif
    #endif
}

// Saves the current state of all wires and registers at the given timestep
static void capture_trace(uint64_t time) {
    #ifdef TRACE_SIGNALS
    tfp->dump(time);
    #ifdef VCD_FLUSH
    tfp->flush();
    #endif
    #endif
}

static void finish_trace() {
    //  Coverage analysis (since test passed)
#if VM_COVERAGE
    Verilated::mkdir("logs");
    VerilatedCov::write("logs/coverage.dat");
#endif
#ifdef TRACE_SIGNALS
    tfp->close();
#endif
}

// ======== Setup and execution of the verilated model ========
static void posedge();
static void negedge();
static void verilator_finish();

// Called by $time in Verilog. Converts to real, to match SystemC
double sc_time_stamp() {
    return main_time;
}

// Called once to initialize the verilated model
static void verilator_init(int argc, char **argv) {
    // set up signaling so we can kill the program and still get waveforms
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = [](int sig){ terminate = true; };
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGKILL, &action, NULL);
    sigaction(SIGINT, &action, NULL);

    atexit(verilator_finish);

    // Set debug level, 0 is off, 9 is highest presently used
    // May be overridden by commandArgs
    Verilated::debug(0);

    // Randomization reset policy
    // May be overridden by commandArgs
    Verilated::randReset(2);

    // Verilator must compute traced signals
    Verilated::traceEverOn(true);

    // Pass arguments so Verilated code can see them, e.g., $value$plusargs
    // This needs to be called before you create any model
    Verilated::commandArgs(argc, argv);

    // Construct the Verilated model
    top = new V_NAME;
    main_time = 0;

    init_tracing();

    top->clk = 0;
    top->reset = 1;
    top->i_valid = 0;
    top->o_ready = 0;
)" << std::endl;

  // Zero out all kernel inputs, except for context variables
  size_t first_ctx_var = reg_args.size() - kernel.GetContextVars().size();
  for (size_t i = 0; i < first_ctx_var; i++)
  {
    cpp << "    top->i_data_" << i << " = 0;" << std::endl;
  }
  for (const auto & ctx : kernel.GetContextVars())
  {
    // Context variables should always be external symbols imported by name
    const auto import = util::AssertedCast<rvsdg::GraphImport>(ctx.input->origin());
    cpp << "    top->i_data_" << first_ctx_var << " = (uint64_t) &" << import->Name() << ";"
        << std::endl;
    first_ctx_var++;
  }

  cpp << R"(
    // Run some cycles with reset set HIGH
    posedge();
    negedge();
    posedge();
    negedge();
    posedge();
    negedge();
    posedge();
    negedge();
    posedge();
    negedge();
    posedge();
    top->reset = 0;
    negedge();
}

// Model outputs should be read right before posedge()
// Model inputs should be set right after posedge()
static void posedge() {
    if (terminate) {
        std::cout << "terminating\n";
        exit(-1);
    }
    assert(!Verilated::gotFinish());
    assert(top->clk == 0);

    // Read memory requests just before the rising edge
)";

  // Emit calls to MemoryQueue::accept_request()
  for (size_t i = 0; i < mem_reqs.size(); i++)
  {
    const auto req_bt = util::AssertedCast<const bundletype>(&mem_reqs[i]->type());
    const auto has_write = req_bt->get_element_type("write") != nullptr;

    cpp << "    memory_queues[" << i << "].accept_request(";
    cpp << "top->mem_" << i << "_req_ready, ";
    cpp << "top->mem_" << i << "_req_valid, ";
    if (has_write)
      cpp << "    top->mem_" << i << "_req_data_write, ";
    else
      cpp << "0, ";
    cpp << "top->mem_" << i << "_req_data_addr, ";
    cpp << "top->mem_" << i << "_req_data_size, ";
    if (has_write)
      cpp << "&top->mem_" << i << "_req_data_data, ";
    else
      cpp << "nullptr, ";
    cpp << "top->mem_" << i << "_req_data_id, ";
    cpp << "top->mem_" << i << "_res_ready, ";
    cpp << "top->mem_" << i << "_res_valid);" << std::endl;
  }

  cpp << R"(
    top->clk = 1;
    top->eval();
    // Capturing the posedge trace here would make external inputs appear on negedge
    // capture_trace(main_time * 2);
}

static void negedge() {
    assert(!Verilated::gotFinish());
    assert(top->clk == 1);

    // Memory responses are ready before the negedge
)";

  // Emit calls to MemoryQueue::produce_response
  for (size_t i = 0; i < mem_reqs.size(); i++)
  {
    cpp << "    memory_queues[" << i << "].produce_response(";
    cpp << "top->mem_" << i << "_req_ready, ";
    cpp << "top->mem_" << i << "_res_valid, ";
    cpp << "&top->mem_" << i << "_res_data_data, ";
    cpp << "top->mem_" << i << "_res_data_id);" << std::endl;
  }

  cpp << R"(
    top->eval();

    // Capturing the posedge trace here makes external inputs appear to update with the posedge
    capture_trace(main_time * 2);

    top->clk = 0;
    top->eval();
    capture_trace(main_time * 2 + 1);
    main_time++;
}

static void verilator_finish() {
  if (!top)
    return;
  top->final();
  finish_trace();
  // delete top;
}

static )"
      << c_return_type.value_or("void") << " run_hls(" << std::endl;
  cpp << c_params << R"(
) {
    if(!top) {
        verilator_init(0, NULL);
    }
    int start = main_time;

    // Run cycles until i_ready becomes HIGH
    for (int i = 0; i < TIMEOUT && !top->i_ready; i++) {
        posedge();
        negedge();
    }
    if (!top->i_ready) {
        std::cout << "i_ready was not set within TIMEOUT" << std::endl;
        exit(-1);
    }

    posedge();

    // Pass in input data for one cycle
    top->i_valid = 1;
)";

  for (size_t i = 0; i < num_c_params; i++)
  {
    if (auto ft = dynamic_cast<const jlm::llvm::FloatingPointType *>(
            kernel.GetOperation().type().Arguments()[i].get()))
    {
      if (ft->size() == llvm::fpsize::flt)
        cpp << "top->i_data_" << i << " = *(uint32_t*) &a" << i << ";" << std::endl;
      else if (ft->size() == llvm::fpsize::dbl)
        cpp << "top->i_data_" << i << " = *(uint64_t*) &a" << i << ";" << std::endl;
    }
    else
    {
      cpp << "top->i_data_" << i << " = (uint64_t) a" << i << ";" << std::endl;
    }
  }

  cpp << R"(
    negedge();
    posedge();

    top->o_ready = 1;
    top->i_valid = 0;
)";

  // Zero out the kernel inputs again
  for (size_t i = 0; i < num_c_params; i++)
  {
    cpp << "top->i_data_" << i << " = 0;" << std::endl;
  }

  cpp << R"(
    negedge();

    // Cycle until o_valid becomes HIGH
    for (int i = 0; i < TIMEOUT && !top->o_valid; i++) {
        posedge();
        negedge();
    }
    if (!top->o_valid) {
        std::cout << "o_valid was not set within TIMEOUT" << std::endl;
        exit(-1);
    }

    std::cout << "finished - took " << (main_time - start) << " cycles" << std::endl;

  // Ensure all memory queues are empty
)";
  for (size_t i = 0; i < mem_reqs.size(); i++)
    cpp << "assert(memory_queues[" << i << "].empty());" << std::endl;

  if (c_return_type.has_value())
    cpp << "return *(" << c_return_type.value() << "*)&top->o_data_0;" << std::endl;

  cpp << R"(
}


// ======== Running the kernel compiled as C, with intrumentation ========
extern "C" )"
      << c_return_type.value_or("void") << " instrumented_ref(" << c_params << ");" << R"(

extern "C" void reference_load(void* addr, uint64_t width) {
    instrumented_load(addr, width);
}

extern "C" void reference_store(void* addr, uint64_t width) {
    instrumented_store(addr, addr, width);
}

extern "C" void reference_alloca(void* start, uint64_t length) {
    ignore_memory_region(start, length);
}

std::vector<mem_access> ref_memory_accesses;

// Calls instrumented_ref in a forked process and stores its memory accesses
static void run_ref(
)" << c_params
      << R"(
) {
    int fd[2]; // channel 0 for reading and 1 for writing
    size_t tmp = pipe(fd);
    int pid = fork();
    if(pid == 0) { // child
        close(fd[0]); // close fd[0] since child will only write

        instrumented_ref()"
      << c_call_args << R"();

        // Send all memory accesses to the parent
        size_t cnt = memory_accesses.size();
        tmp = write(fd[1], &cnt, sizeof(size_t));
        for (auto & access : memory_accesses){
            tmp = write(fd[1], &access, sizeof(mem_access));
            tmp = write(fd[1], access.data, 1<< access.width);
        }

        close(fd[1]);
        exit(0);
    } else { // parent
        close(fd[1]); // close fd[1] since parent will only read

        // Retrieve all memory_accesses from the child
        size_t cnt;
        tmp = read(fd[0], &cnt, sizeof(size_t));
        ref_memory_accesses.resize(cnt);
        for (auto & access : ref_memory_accesses) {
            tmp = read(fd[0], &access, sizeof(mem_access));
            access.data = malloc(1 << access.width);
            tmp = read(fd[0], access.data, 1 << access.width);
        }

        close(fd[0]);
    }
}

// Checks that memory_accesses and ref_memory_accesses are identical within each address
static void compare_memory_accesses() {
    assert (memory_accesses.size() == ref_memory_accesses.size());

    // Stable sort the memory accesses by only address, keeping order within each address.
    auto addr_sort = [](const mem_access & a, const mem_access & b) {
        return a.addr < b.addr;
    };
    std::stable_sort(memory_accesses.begin(), memory_accesses.end(), addr_sort);
    std::stable_sort(ref_memory_accesses.begin(), ref_memory_accesses.end(), addr_sort);
    assert(memory_accesses == ref_memory_accesses);
}

static void empty_mem_acces_vector(std::vector<mem_access> &vec){
    for (auto &m: vec) {
        free(m.data);
    }
    vec.erase(vec.begin(), vec.end());
}

// ======== Entry point for calling kernel from host device (C code) ========
extern "C" )"
      << c_return_type.value_or("void") << " " << function_name << "(" << c_params << ")" << R"(
{
    // Execute instrumented version of kernel compiled for the host in a fork
    run_ref()"
      << c_call_args << R"();

    // Execute the verilated model in this process
    )";
  if (c_return_type.has_value())
    cpp << "auto result = ";
  cpp << "run_hls(" << c_call_args << ");" << std::endl;

  cpp << R"(
    // Compare traced memory accesses
    compare_memory_accesses();

    // Reset structures used for tracing memory operations
    empty_mem_acces_vector(memory_accesses);
    empty_mem_acces_vector(ref_memory_accesses);
    ignored_memory_regions.clear();
)";

  if (c_return_type.has_value())
    cpp << "    return result;" << std::endl;

  cpp << "}" << std::endl;

  return cpp.str();
}

} // namespace jlm::hls
