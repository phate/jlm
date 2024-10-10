/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>

namespace jlm::hls
{

std::string
VerilatorHarnessHLS::get_text(llvm::RvsdgModule & rm)
{
  std::ostringstream cpp;
  auto ln = get_hls_lambda(rm);
  auto function_name = ln->name();
  auto file_name = get_base_file_name(rm);

  auto mem_reqs = get_mem_reqs(ln);
  auto mem_resps = get_mem_resps(ln);
  JLM_ASSERT(mem_reqs.size() == mem_resps.size());
  cpp << "#define TRACE_CHUNK_SIZE 100000\n"
         //		"#define HLS_MEM_DEBUG 1\n"
         "\n"
         "#include <verilated.h>\n"
         "#include <cassert>\n"
         "#include <iostream>\n"
         "#include <signal.h>\n"
         "#include <stdio.h>\n"
         "#include <stdlib.h>\n"
         "#include <string.h>\n"
         "#include <unistd.h>\n"
         "#include <sstream>\n"
         "#include <iomanip>\n"
         "#include <queue>\n"
         "#ifdef FST\n"
         "#include \"verilated_fst_c.h\"\n"
         "#else\n"
         "#include \"verilated_vcd_c.h\"\n"
         "#endif\n"
         // Include the Verilator generated header, which provides access to Verilog signals
         // The name of the header is based on the Verilog filename used as input to Verilator
         "#include \"V"
      << GetVerilogFileName().base() << ".h\"\n"
      << "#define V_NAME V" << GetVerilogFileName().base() << "\n"
      << "#define TIMEOUT 10000000\n"
         "#define xstr(s) str(s)\n"
         "#define str(s) #s\n"
         "void clock_cycle();\n"
         "\n"
         "\n"
         "typedef struct MemAccess{\n"
         "    void * addr;\n"
         "    uint64_t data;\n"
         "    uint64_t width;\n"
         "    uint64_t ctr;\n"
         "\n"
         "    bool operator==(const MemAccess& rhs) const {\n"
         "        return addr == rhs.addr && data == rhs.data && width == rhs.width;\n"
         "    }\n"
         "} mem_access;\n\n"
         "\n"
         "uint64_t mem_access_ctr = 0;"
         "\n"
         "std::vector<mem_access> ref_loads;\n"
         "std::vector<mem_access> ref_stores;\n"
         "std::vector<std::pair<void *, uint64_t>> ref_allocas;\n"
         "std::vector<mem_access> hls_loads;\n"
         "std::vector<mem_access> hls_stores;\n"
         "std::map<void*, std::deque<mem_access>> load_map;\n"
         "std::map<void*, std::deque<mem_access>> store_map;\n"
         "\n"
         "void access_mem_load(mem_access access){\n"
         "    hls_loads.push_back(access);\n"
         "    auto find = load_map.find(access.addr);\n"
         "    if(find == load_map.end()){\n"
         "        throw std::logic_error(\"unexpected load address\");\n"
         "    }\n"
         "    if(find->second.empty()){\n"
         "        throw std::logic_error(\"too many loads to address\");\n"
         "    }\n"
         "    if(!find->second.front().operator==(access)){\n"
         "        throw std::logic_error(\"wrong type of load to address\");\n"
         "    }\n"
         "    find->second.pop_front();\n"
         "}\n"
         "\n"
         "void access_mem_store(mem_access access){\n"
         "    hls_stores.push_back(access);\n"
         "    auto find = store_map.find(access.addr);\n"
         "    if(find == store_map.end()){\n"
         "        throw std::logic_error(\"unexpected store address\");\n"
         "    }\n"
         "    if(find->second.empty()){\n"
         "        throw std::logic_error(\"too many stores to address\");\n"
         "    }\n"
         "    if(!find->second.front().operator==(access)){\n"
         "        throw std::logic_error(\"wrong type of store to address\");\n"
         "    }\n"
         "    find->second.pop_front();\n"
         "}\n"
         "// Current simulation time (64-bit unsigned)\n"
         "vluint64_t main_time = 0;\n"
         "// Called by $time in Verilog\n"
         "double sc_time_stamp() {\n"
         "    return main_time;  // Note does conversion to real, to match SystemC\n"
         "}\n"
         "V_NAME *top;\n"
         "#ifdef TRACE_SIGNALS\n"
         "#ifdef FST\n"
         "VerilatedFstC *tfp;\n"
         "#else\n"
         "VerilatedVcdC *tfp;\n"
         "#endif\n"
         "#endif\n"
         "bool terminate = false;\n"
         "\n"
         "void term(int signum) {\n"
         "    terminate = true;\n"
         "}\n"
         "\n"
         "void verilator_finish() {\n"
         "    // Final model cleanup\n"
         "#ifdef TRACE_SIGNALS\n"
         "    tfp->dump(main_time * 2);\n"
         "#endif\n"
         "    top->final();\n"
         "\n"
         "    //  Coverage analysis (since test passed)\n"
         "#if VM_COVERAGE\n"
         "    Verilated::mkdir(\"logs\");\n"
         "    VerilatedCov::write(\"logs/coverage.dat\");\n"
         "#endif\n"
         "#ifdef TRACE_SIGNALS\n"
         "    tfp->close();\n"
         "#endif\n"
         "    // Destroy model\n"
         //		"    delete top;\n"
         //		"    top = NULL;\n"
         "}\n"
         "\n"
         "typedef struct mem_resp_struct {\n"
         "    bool valid = false;\n"
         "    uint64_t data = 0xDEADBEEF;\n"
         "    uint8_t id = 0;\n"
         "} mem_resp_struct;\n"
         "\n"
         "std::queue<mem_resp_struct>* mem_resp["
      << mem_resps.size()
      << "];\n"
         "const uint64_t mem_latency["
      << mem_resps.size() << "] = {";
  for (size_t i = 0; i < mem_resps.size(); ++i)
  {
    if (i != 0)
    {
      cpp << ", ";
    }
    cpp << "10";
  }
  cpp << "};\n";
  cpp << "\n"
         "void verilator_init(int argc, char **argv) {\n"
         "    // set up signaling so we can kill the program and still get waveforms\n"
         "    struct sigaction action;\n"
         "    memset(&action, 0, sizeof(struct sigaction));\n"
         "    action.sa_handler = term;\n"
         "    sigaction(SIGTERM, &action, NULL);\n"
         "    sigaction(SIGKILL, &action, NULL);\n"
         "    sigaction(SIGINT, &action, NULL);\n";

  for (size_t i = 0; i < mem_resps.size(); ++i)
  {
    cpp << "    mem_resp[" << i << "] = new std::queue<mem_resp_struct>();\n";
  }
  for (size_t i = 0; i < mem_resps.size(); ++i)
  {
    cpp << "    for (size_t i = 0; i < mem_latency[" << i
        << "]; ++i) {\n"
           "       mem_resp["
        << i
        << "]->emplace();\n"
           "    }\n";
  }
  cpp << "\n"
         "	atexit(verilator_finish);\n"
         "\n"
         "    // Set debug level, 0 is off, 9 is highest presently used\n"
         "    // May be overridden by commandArgs\n"
         "    Verilated::debug(0);\n"
         "\n"
         "    // Randomization reset policy\n"
         "    // May be overridden by commandArgs\n"
         "    Verilated::randReset(2);\n"
         "\n"
         "    // Verilator must compute traced signals\n"
         "    Verilated::traceEverOn(true);\n"
         "\n"
         "    // Pass arguments so Verilated code can see them, e.g. $value$plusargs\n"
         "    // This needs to be called before you create any model\n"
         "    Verilated::commandArgs(argc, argv);\n"
         "\n"
         "    // Construct the Verilated model, from Vtop.h generated from Verilating \"top.v\"\n"
         "    top = new V_NAME;  // Or use a const unique_ptr, or the VL_UNIQUE_PTR wrapper\n"
         "#ifdef TRACE_SIGNALS\n"
         "#ifdef FST\n"
         "    tfp = new VerilatedFstC;\n"
         "    top->trace(tfp, 99);   // Trace 99 levels of hierarchy\n"
         "    tfp->open(xstr(V_NAME)\".fst\");\n"
         "#else\n"
         "    tfp = new VerilatedVcdC;\n"
         "    top->trace(tfp, 99);   // Trace 99 levels of hierarchy\n"
         "    tfp->open(xstr(V_NAME)\".vcd\");\n"
         "#endif\n"
         "#endif\n"
         "\n"
         "    top->i_valid = 0;\n";
  // reset all data inputs to zero
  for (size_t i = 0; i < ln->ninputs(); ++i)
  {
    cpp << "    top->i_data_" << i << " = 0;\n";
  }
  cpp << "    top->reset = 1;\n"
         "\n";
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    cpp << "    top->mem_" << i
        << "_req_ready = false;\n"
           "    top->mem_"
        << i
        << "_res_valid = false;\n"
           "    top->mem_"
        << i << "_res_data_data = 0xDEADBEEF;\n";
  }
  cpp << "    clock_cycle();\n"
         "    clock_cycle();\n"
         "    top->reset = 0;\n"
         "    clock_cycle();\n"
         "}\n"
         "\n"
         "void posedge() {\n"
         "    if (terminate) {\n"
         "        std::cout << \"terminating\\n\";\n"
         "        verilator_finish();\n"
         "        exit(-1);\n"
         "    }\n"
         "    assert(!Verilated::gotFinish());\n"
         "    top->clk = 1;\n"
         "    top->eval(); //eval here to get a clean posedge with the old inputs\n"
         "}\n"
         "\n"
         "void finish_clock_cycle() {\n";
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    cpp << "    top->mem_" << i << "_res_data_data = mem_resp[" << i
        << "]->front().data;\n"
           "    top->mem_"
        << i << "_res_data_id = mem_resp[" << i
        << "]->front().id;\n"
           "    top->mem_"
        << i << "_res_valid = mem_resp[" << i
        << "]->front().valid;\n"
           "    top->mem_"
        << i << "_req_ready = true;\n";
  }
  cpp << "    top->eval();\n"
         "    // dump before trying to access memory\n"
         "#ifdef TRACE_SIGNALS\n"
         "    tfp->dump(main_time * 2);\n"
         "#ifdef VCD_FLUSH\n"
         "    tfp->flush();\n"
         "#endif\n"
         "#endif\n";
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    cpp << "    if (top->mem_" << i << "_res_valid && top->mem_" << i
        << "_res_ready) {\n"
           "        mem_resp["
        << i
        << "]->pop();\n"
           "    } else if (!mem_resp["
        << i
        << "]->front().valid){\n"
           "        mem_resp["
        << i
        << "]->pop();\n"
           "    }\n";
  }
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    cpp << "    // mem_" << i
        << "\n"
           "    if (!top->reset && top->mem_"
        << i << "_req_valid && top->mem_" << i
        << "_req_ready) {\n"
           "        mem_resp["
        << i
        << "]->emplace();\n"
           "        mem_resp["
        << i
        << "]->back().valid = true;\n"
           "        mem_resp["
        << i << "]->back().id = top->mem_" << i
        << "_req_data_id;\n"
           "        void *addr = (void *) top->mem_"
        << i
        << "_req_data_addr;\n"
           "        uint64_t size = top->mem_"
        << i
        << "_req_data_size;\n"
           "        uint64_t data;\n";
    auto req_bt = dynamic_cast<const bundletype *>(&mem_reqs[i]->type());
    auto has_write = req_bt->get_element_type("write") != nullptr;
    if (has_write)
    {
      cpp << "        data = top->mem_" << i
          << "_req_data_data;\n"
             "        if (top->mem_"
          << i
          << "_req_data_write) {\n"
             "#ifdef HLS_MEM_DEBUG\n"
             "            std::cout << \"mem_"
          << i
          << " writing \" << data << \" to \" << addr << \"\\n\";\n"
             "#endif\n"
             "            access_mem_store({addr, data, size, mem_access_ctr++});\n"
             "            switch (size) {\n"
             "                case 0:\n"
             "                    *(uint8_t *) addr = data;\n"
             "                    break;\n"
             "                case 1:\n"
             "                    *(uint16_t *) addr = data;\n"
             "                    break;\n"
             "                case 2:\n"
             "                    *(uint32_t *) addr = data;\n"
             "                    break;\n"
             "                case 3:\n"
             "                    *(uint64_t *) addr = data;\n"
             "                    break;\n"
             "                default:\n"
             "                    assert(false);\n"
             "            }\n"
             "            mem_resp["
          << i
          << "]->back().data = 0xFFFFFFFF;\n"
             "        } else {\n";
    }
    else
    {
      cpp << "        if (true) {\n";
    }
    cpp << "#ifdef HLS_MEM_DEBUG\n"
           "            std::cout << \"mem_"
        << i
        << " reading from \" << addr << \"\\n\";\n"
           "#endif\n"
           "            switch (size) {\n"
           "                case 0:\n"
           "                    data = *(uint8_t *) addr;\n"
           "                    break;\n"
           "                case 1:\n"
           "                    data = *(uint16_t *) addr;\n"
           "                    break;\n"
           "                case 2:\n"
           "                    data = *(uint32_t *) addr;\n"
           "                    break;\n"
           "                case 3:\n"
           "                    data = *(uint64_t *) addr;\n"
           "                    break;\n"
           "                default:\n"
           "                    assert(false);\n"
           "            }\n"
           "            mem_resp["
        << i
        << "]->back().data = data;\n"
           "            access_mem_load({addr, data, size, mem_access_ctr++});\n"
           "        }\n"
           "    } else if (mem_resp["
        << i << "]->size()<mem_latency[" << i
        << "]){\n"
           "        mem_resp["
        << i
        << "]->emplace();\n"
           "    }\n";
  }
  cpp << "    assert(!Verilated::gotFinish());\n"
         "    top->clk = 0;\n"
         "    top->eval();\n"
         "#ifdef TRACE_SIGNALS\n"
         "    tfp->dump(main_time * 2 + 1);\n"
         "#ifdef VCD_FLUSH\n"
         "    tfp->flush();\n"
         "#endif\n"
         "#endif\n"
         "    main_time++;\n"
         "}\n"
         "\n"
         "void clock_cycle() {\n"
         "    posedge();\n"
         "    finish_clock_cycle();\n"
         "}\n"
         "\n";

  cpp << "extern \"C\"\n" // TODO: parameter for linkage type here
         "{\n";
  // imports
  auto root = rm.Rvsdg().root();
  for (size_t i = 0; i < root->narguments(); ++i)
  {
    if (auto graphImport = dynamic_cast<const llvm::GraphImport *>(root->argument(i)))
    {
      if (dynamic_cast<const jlm::llvm::PointerType *>(&graphImport->type()))
      {
        cpp << "extern " << convert_to_c_type(&graphImport->type()) << " " << graphImport->Name()
            << ";\n";
      }
      else
      {
        throw util::error("unexpected impport type");
      }
    }
  }

  get_function_header(cpp, ln, "instrumented_ref");
  cpp << ";\n";
  cpp << "bool in_alloca(void *addr){\n"
         "    for (auto a: ref_allocas) {\n"
         "        if(addr >= a.first && addr < ((uint8_t*)a.first)+a.second){\n"
         "            return true;\n"
         "        }\n"
         "    }\n"
         "    return false;\n"
         "}\n"
         "\n"
         "void reference_load(void *addr, uint64_t width) {\n"
         "    if(in_alloca(addr)){\n"
         "        return;\n"
         "    }\n"
         "    uint64_t data;\n"
         "    switch (width) {\n"
         "        case 0:\n"
         "            data = *(uint8_t *) addr;\n"
         "            break;\n"
         "        case 1:\n"
         "            data = *(uint16_t *) addr;\n"
         "            break;\n"
         "        case 2:\n"
         "            data = *(uint32_t *) addr;\n"
         "            break;\n"
         "        case 3:\n"
         "            data = *(uint64_t *) addr;\n"
         "            break;\n"
         "        default:\n"
         "            assert(false);\n"
         "    }\n"
         "    ref_loads.push_back({addr, data, width, mem_access_ctr++});\n"
         "}\n"
         "\n"
         "void reference_store(void *addr, uint64_t data, uint64_t width) {\n"
         "    if(in_alloca(addr)){\n"
         "        return;\n"
         "    }\n"
         "    ref_stores.push_back({addr, data, width, mem_access_ctr++});\n"
         "}\n"
         "\n"
         "void reference_alloca(void *addr, uint64_t size) {\n"
         "    std::cout << \"alloca \" << std::hex << addr << \" \" << std::dec << size << "
         "std::endl;\n"
         "    ref_allocas.emplace_back(addr, size);\n"
         "}\n"
         "\n";
  get_function_header(cpp, ln, "run_hls");
  cpp << " {\n";
  cpp << "	if(!top){\n"
         "		verilator_init(0, NULL);\n"
         "	}\n";
  size_t register_ix = 0;
  for (size_t i = 0; i < ln->type().NumArguments(); ++i)
  {
    if (dynamic_cast<const rvsdg::StateType *>(&ln->type().ArgumentType(i)))
    {
      register_ix++;
      continue;
    }
    else if (dynamic_cast<const bundletype *>(&ln->type().ArgumentType(i)))
    {
      continue;
    }
    cpp << "    top->i_data_" << i << " = (uint64_t) a" << i << ";\n";
    register_ix++;
  }
  for (const auto & ctxvar : ln->GetContextVars())
  {
    std::string name;
    if (auto graphImport = dynamic_cast<const llvm::GraphImport *>(ctxvar.input->origin()))
    {
      name = graphImport->Name();
    }
    else
    {
      throw util::error("Unsupported cvarg origin type type");
    }
    cpp << "    top->i_data_" << register_ix++ << " = (uint64_t) &" << name << ";\n";
    cpp << "#ifdef HLS_MEM_DEBUG\n";
    cpp << "    std::cout << \"" << name << ": \" << &" << name << " << \"\\n\";\n";
    cpp << "#endif\n";
  }
  cpp << "    int start = main_time;\n"
         "    for (int i = 0; i < TIMEOUT && !top->i_ready; i++) {\n"
         "        clock_cycle();\n"
         "    }\n"
         "    if (!top->i_ready) {\n"
         "        std::cout << \"i_ready not set\\n\";\n"
         "        verilator_finish();\n"
         "        exit(-1);\n"
         "    }\n"
         "    posedge();\n"
         "    top->i_valid = 1;\n"
         "    top->o_ready = 1;\n"
         "    finish_clock_cycle();\n"
         "    posedge();\n"
         "    top->i_valid = 0;\n";
  for (size_t i = 0; i < ln->type().NumArguments(); ++i)
  {
    if (!dynamic_cast<const bundletype *>(&ln->type().ArgumentType(i)))
    {
      cpp << "    top->i_data_" << i << " = 0;\n";
    }
  }
  cpp << "    finish_clock_cycle();\n"
         "    for (int i = 0; i < TIMEOUT && !top->o_valid; i++) {\n"
         "        clock_cycle();\n"
         "    }\n"
         "    if (!top->o_valid) {\n"
         "        std::cout << \"o_valid not set\\n\";\n"
         "        //verilator_finish();\n"
         "        exit(-1);\n"
         "    }\n"
         "\n"
         "    for (auto &pair: store_map) {\n"
         "        assert(pair.second.empty());\n"
         "    }\n"
         "    for (auto &pair: load_map) {\n"
         "        assert(pair.second.empty());\n"
         "    }\n"
         "    std::cout << \"finished - took \" << (main_time - start) << \"cycles\\n\";\n"
         "\n"
         "    // empty loads and stores\n"
         "    ref_loads.erase(ref_loads.begin(), ref_loads.end());\n"
         "    ref_stores.erase(ref_stores.begin(), ref_stores.end());\n"
         "    hls_loads.erase(hls_loads.begin(), hls_loads.end());\n"
         "    hls_stores.erase(hls_stores.begin(), hls_stores.end());\n"
         "    mem_access_ctr = 0;\n";
  if (ln->type().NumResults() && !dynamic_cast<const rvsdg::StateType *>(&ln->type().ResultType(0)))
  {
    cpp << "    return top->o_data_0;\n";
  }
  cpp << "}\n";

  get_function_header(cpp, ln, "run_ref");
  cpp << " {\n";
  cpp << "    int fd[2]; // channel 0 for reading and 1 for writing\n"
         "    size_t tmp = pipe(fd);\n"
         "    int pid = fork();\n"
         "    if(pid == 0) { // child\n"
         "        close(fd[0]); // close fd[0] since child will only write\n"
         "        ";
  call_function(cpp, ln, "instrumented_ref");
  cpp << "\n"
         "        size_t cnt = ref_loads.size();\n"
         "        tmp = write(fd[1], &cnt, sizeof(size_t));\n"
         "        for (auto load:ref_loads) {\n"
         "            tmp = write(fd[1], &load, sizeof(mem_access));\n"
         "        }\n"
         "        cnt = ref_stores.size();\n"
         "        tmp = write(fd[1], &cnt, sizeof(size_t));\n"
         "        for (auto store:ref_stores) {\n"
         "            tmp = write(fd[1], &store, sizeof(mem_access));\n"
         "        }\n"
         "        close(fd[1]);\n"
         "        exit(0);\n"
         "    } else { // parent\n"
         "        close(fd[1]); // close fd[1] since parent will only read\n"
         "        size_t cnt;\n"
         "        size_t tmp = read(fd[0], &cnt, sizeof(size_t));\n"
         "        for (size_t i = 0; i < cnt; ++i) {\n"
         "            mem_access load;\n"
         "            tmp = read(fd[0], &load, sizeof(mem_access));\n"
         "            ref_loads.push_back(load);\n"
         "            if(load_map.find(load.addr) == load_map.end()){\n"
         "                load_map.emplace(load.addr, std::deque<mem_access>());\n"
         "            }\n"
         "            load_map.find(load.addr)->second.push_back(load);\n"
         "        }\n"
         "        tmp = read(fd[0], &cnt, sizeof(size_t));\n"
         "        for (size_t i = 0; i < cnt; ++i) {\n"
         "            mem_access store;\n"
         "            tmp = read(fd[0], &store, sizeof(mem_access));\n"
         "            ref_stores.push_back(store);\n"
         "            if(store_map.find(store.addr) == store_map.end()){\n"
         "                store_map.emplace(store.addr, std::deque<mem_access>());\n"
         "            }\n"
         "            store_map.find(store.addr)->second.push_back(store);\n"
         "        }\n"
         "        close(fd[0]);\n"
         "    }\n";

  if (ln->type().NumResults() && !dynamic_cast<const rvsdg::StateType *>(&ln->type().ResultType(0)))
  {
    cpp << "    return 0;\n";
  }
  cpp << "}\n";
  get_function_header(cpp, ln, function_name);
  cpp << " {\n"
         "    ";
  call_function(cpp, ln, "run_ref");
  cpp << "\n";
  if (ln->type().NumResults() && !dynamic_cast<const rvsdg::StateType *>(&ln->type().ResultType(0)))
  {
    cpp << "    return ";
  }
  else
  {
    cpp << "    ";
  }
  call_function(cpp, ln, "run_hls");
  cpp << "\n";
  cpp << "}\n";
  cpp << "}\n";
  return cpp.str();
}

void
VerilatorHarnessHLS::call_function(
    std::ostringstream & cpp,
    const jlm::llvm::lambda::node * ln,
    const std::string & function_name)
{
  cpp << function_name << "(";
  for (size_t i = 0; i < ln->type().NumArguments(); ++i)
  {
    if (dynamic_cast<const rvsdg::StateType *>(&ln->type().ArgumentType(i)))
    {
      continue;
    }
    else if (dynamic_cast<const bundletype *>(&ln->type().ArgumentType(i)))
    {
      continue;
    }
    if (i != 0)
    {
      cpp << " ,";
    }
    cpp << "a" << i;
  }
  cpp << ");";
}

void
VerilatorHarnessHLS::get_function_header(
    std::ostringstream & cpp,
    const jlm::llvm::lambda::node * ln,
    const std::string & function_name)
{
  std::string return_type;
  if (ln->type().NumResults() == 0)
  {
    return_type = "void";
  }
  else
  {
    auto type = &ln->type().ResultType(0);
    if (dynamic_cast<const rvsdg::StateType *>(type))
    {
      return_type = "void";
    }
    else if (dynamic_cast<const bundletype *>(type))
    {
      return_type = "void";
    }
    else
    {
      return_type = convert_to_c_type(type);
    }
  }
  cpp << return_type << " " << function_name << "(\n";
  for (size_t i = 0; i < ln->type().NumArguments(); ++i)
  {
    if (dynamic_cast<const rvsdg::StateType *>(&ln->type().ArgumentType(i)))
    {
      continue;
    }
    else if (dynamic_cast<const bundletype *>(&ln->type().ArgumentType(i)))
    {
      continue;
    }
    if (i != 0)
    {
      cpp << ",\n";
    }
    cpp << "    " << convert_to_c_type(&ln->type().ArgumentType(i)) << " a" << i
        << convert_to_c_type_postfix(&ln->type().ArgumentType(i));
  }
  cpp << "\n"
         ")";
}

std::string
VerilatorHarnessHLS::convert_to_c_type(const jlm::rvsdg::Type * type)
{
  if (auto t = dynamic_cast<const jlm::rvsdg::bittype *>(type))
  {
    return "int" + util::strfmt(t->nbits()) + "_t";
  }
  else if (jlm::rvsdg::is<llvm::PointerType>(*type))
  {
    return "void*";
  }
  else if (auto t = dynamic_cast<const llvm::arraytype *>(type))
  {
    return convert_to_c_type(&t->element_type());
  }
  else
  {
    throw std::logic_error(type->debug_string() + " not implemented!");
  }
}

std::string
VerilatorHarnessHLS::convert_to_c_type_postfix(const jlm::rvsdg::Type * type)
{
  if (auto t = dynamic_cast<const llvm::arraytype *>(type))
  {
    return util::strfmt("[", t->nelements(), "]", convert_to_c_type(&t->element_type()));
  }
  else
  {
    return "";
  }
}

} // namespace jlm::hls
