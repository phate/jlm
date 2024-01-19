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
  assert(mem_reqs.size() == mem_resps.size());
  cpp << "#define TRACE_CHUNK_SIZE 100000\n"
#ifndef HLS_USE_VCD
         "#define FST 1\n"
#endif
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
         "#include \"V"
      << file_name << ".h\"\n"
      << "#define V_NAME V" << file_name << "\n"
      << "#define TIMEOUT 10000000\n"
         "#define xstr(s) str(s)\n"
         "#define str(s) #s\n"
         "void clock_cycle();\n"
         "\n"
         "// Current simulation time (64-bit unsigned)\n"
         "vluint64_t main_time = 0;\n"
         "// Called by $time in Verilog\n"
         "double sc_time_stamp() {\n"
         "    return main_time;  // Note does conversion to real, to match SystemC\n"
         "}\n"
         "V_NAME *top;\n"
         "#ifdef FST\n"
         "VerilatedFstC *tfp;\n"
         "#else\n"
         "VerilatedVcdC *tfp;\n"
         "#endif\n"
         "bool terminate = false;\n"
         "\n"
         "void term(int signum) {\n"
         "    terminate = true;\n"
         "}\n"
         "\n"
         "void verilator_finish() {\n"
         "    // Final model cleanup\n"
         "    tfp->dump(main_time * 2);\n"
         "    top->final();\n"
         "\n"
         "    //  Coverage analysis (since test passed)\n"
         "#if VM_COVERAGE\n"
         "    Verilated::mkdir(\"logs\");\n"
         "    VerilatedCov::write(\"logs/coverage.dat\");\n"
         "#endif\n"
         "    tfp->close();\n"
         "    // Destroy model\n"
         "    delete top;\n"
         "    top = NULL;\n"
         "}\n"
         "\n"
         "#define MEM_LATENCY 10\n"
         "typedef struct mem_resp_struct {\n"
         "    bool valid = false;\n"
         "    uint64_t data = 0xDEADBEEF;\n"
         "    uint8_t id = 0;\n"
         "} mem_resp_struct;\n"
         "\n"
         "std::queue<mem_resp_struct>* mem_resp["
      << mem_resps.size()
      << "];"
         "\n"
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
  cpp << "    for (size_t i = 0; i < MEM_LATENCY; ++i) {\n";
  for (size_t i = 0; i < mem_resps.size(); ++i)
  {
    cpp << "    mem_resp[" << i << "]->emplace();\n";
  }
  cpp << "    }\n";
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
         "#ifdef FST\n"
         "    tfp = new VerilatedFstC;\n"
         "    top->trace(tfp, 99);   // Trace 99 levels of hierarchy\n"
         "    tfp->open(xstr(V_NAME)\".fst\");\n"
         "#else\n"
         "    tfp = new VerilatedVcdC;\n"
         "    top->trace(tfp, 99);   // Trace 99 levels of hierarchy\n"
         "    tfp->open(xstr(V_NAME)\".vcd\");\n"
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
         "    tfp->dump(main_time * 2);\n";
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
        << i << "_req_data_addr;\n";
    auto req_bt = dynamic_cast<const jlm::hls::bundletype *>(&mem_reqs[i]->type());
    auto has_write = req_bt->get_element_type("write") != nullptr;
    if (has_write)
    {
      cpp << "        uint64_t data = top->mem_" << i
          << "_req_data_data;\n"
             "        if (top->mem_"
          << i
          << "_req_data_write) {\n"
             "#ifdef HLS_MEM_DEBUG\n"
             "            std::cout << \"mem_"
          << i
          << " writing \" << data << \" to \" << addr << \"\\n\";\n"
             "#endif\n"
             "            switch (top->mem_"
          << i
          << "_req_data_size) {\n"
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
           "        }\n"
           "        switch (top->mem_"
        << i
        << "_req_data_size) {\n"
           "            case 0:\n"
           "                mem_resp["
        << i
        << "]->back().data = *(uint8_t *) addr;\n"
           "                break;\n"
           "            case 1:\n"
           "                mem_resp["
        << i
        << "]->back().data = *(uint16_t *) addr;\n"
           "                break;\n"
           "            case 2:\n"
           "                mem_resp["
        << i
        << "]->back().data = *(uint32_t *) addr;\n"
           "                break;\n"
           "            case 3:\n"
           "                mem_resp["
        << i
        << "]->back().data = *(uint64_t *) addr;\n"
           "                break;\n"
           "            default:\n"
           "                assert(false);\n"
           "        }\n"
           "    } else if (mem_resp["
        << i
        << "]->size()<MEM_LATENCY){\n"
           "        mem_resp["
        << i
        << "]->emplace();\n"
           "    }\n";
  }
  cpp << "    assert(!Verilated::gotFinish());\n"
         "    top->clk = 0;\n"
         "    top->eval();\n"
         "    tfp->dump(main_time * 2 + 1);\n"
         "//    tfp->flush();\n"
         "    main_time++;\n"
         "}\n"
         "\n"
         "void clock_cycle() {\n"
         "    posedge();\n"
         "    finish_clock_cycle();\n"
         "}\n"
         "\n"
         "extern \"C\"\n" // TODO: parameter for linkage type here
         "{\n";
  // imports
  auto root = rm.Rvsdg().root();
  for (size_t i = 0; i < root->narguments(); ++i)
  {
    if (auto rvsdgImport = dynamic_cast<const llvm::impport *>(&root->argument(i)->port()))
    {
      JLM_ASSERT(jlm::rvsdg::is<llvm::PointerType>(rvsdgImport->type()));
      cpp << "extern " << convert_to_c_type(&rvsdgImport->GetValueType()) << " "
          << rvsdgImport->name() << ";\n";
    }
  }
  std::string return_type;
  if (ln->type().NumResults() == 0)
  {
    return_type = "void";
  }
  else
  {
    auto type = &ln->type().ResultType(0);
    if (dynamic_cast<const jlm::rvsdg::statetype *>(type))
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
    if (dynamic_cast<const jlm::rvsdg::statetype *>(&ln->type().ArgumentType(i)))
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
         ") {\n"
         "	if(!top){\n"
         "		verilator_init(0, NULL);\n"
         "	}\n";
  for (size_t i = 0; i < ln->type().NumArguments(); ++i)
  {
    if (dynamic_cast<const jlm::rvsdg::statetype *>(&ln->type().ArgumentType(i)))
    {
      continue;
    }
    cpp << "    top->i_data_" << i << " = (uint64_t) a" << i << ";\n";
  }
  for (size_t i = 0; i < ln->ncvarguments(); ++i)
  {
    size_t ix = ln->cvargument(i)->input()->argument()->index();
    std::string name;
    if (auto a = dynamic_cast<jlm::rvsdg::argument *>(ln->input(i)->origin()))
    {
      if (auto ip = dynamic_cast<const llvm::impport *>(&a->port()))
      {
        name = ip->name();
      }
    }
    else
    {
      throw util::error("Unsupported cvarg origin type type");
    }
    cpp << "    top->i_data_" << ix << " = (uint64_t) &" << name << ";\n";
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
    cpp << "    top->i_data_" << i << " = 0;\n";
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
         "    std::cout << \"finished - took \" << (main_time - start) << \"cycles\\n\";\n";
  if (ln->type().NumResults()
      && !dynamic_cast<const jlm::rvsdg::statetype *>(&ln->type().ResultType(0)))
  {
    cpp << "    return top->o_data_0;\n";
  }
  cpp << "}\n}\n";
  return cpp.str();
}

std::string
VerilatorHarnessHLS::convert_to_c_type(const jlm::rvsdg::type * type)
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
VerilatorHarnessHLS::convert_to_c_type_postfix(const jlm::rvsdg::type * type)
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
}
