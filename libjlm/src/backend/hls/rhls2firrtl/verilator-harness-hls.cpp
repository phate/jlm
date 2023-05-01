/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/backend/hls/rhls2firrtl/verilator-harness-hls.hpp>

std::string
jlm::hls::VerilatorHarnessHLS::get_text(jlm::RvsdgModule &rm) {
	std::ostringstream cpp;
	auto ln = get_hls_lambda(rm);
	auto function_name = ln->name();
	auto file_name = get_base_file_name(rm);
	cpp <<
		"#define TRACE_CHUNK_SIZE 100000\n"
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
		"#ifdef FST\n"
		"#include \"verilated_fst_c.h\"\n"
		"#else\n"
		"#include \"verilated_vcd_c.h\"\n"
		"#endif\n"
		"#include \"V" << file_name << ".h\"\n" <<
		"#define V_NAME V" << file_name << "\n" <<
		"#define TIMEOUT 10000000\n"
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
		"void verilator_init(int argc, char **argv) {\n"
		"    // set up signaling so we can kill the program and still get waveforms\n"
		"    struct sigaction action;\n"
		"    memset(&action, 0, sizeof(struct sigaction));\n"
		"    action.sa_handler = term;\n"
		"    sigaction(SIGTERM, &action, NULL);\n"
		"    sigaction(SIGKILL, &action, NULL);\n"
		"    sigaction(SIGINT, &action, NULL);\n"
		"\n"
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
	for (size_t i = 0; i < ln->ninputs(); ++i) {
		cpp << "    top->i_data" << i << " = 0;\n";
	}
	cpp <<
		"    top->reset = 1;\n"
		"\n"
		"    top->mem_req_ready = true;\n"
		"    top->mem_res_valid = false;\n"
		"    top->mem_res_data = 1111111;\n"
		"    clock_cycle();\n"
		"    clock_cycle();\n"
		"    top->reset = 0;\n"
		"    clock_cycle();\n"
		"}\n"
		"\n"
		"bool mem_resp_valid;\n"
		"uint64_t mem_resp_data;\n"
		"\n"
		"void clock_cycle() {\n"
		"    if (terminate) {\n"
		"        std::cout << \"terminating\\n\";\n"
		"        verilator_finish();\n"
		"        exit(-1);\n"
		"    }\n"
		"    assert(!Verilated::gotFinish());\n"
		"    top->clk = 1;\n"
		"    top->eval();\n"
		"    top->mem_res_valid = mem_resp_valid;\n"
		"    top->mem_res_data = mem_resp_data;\n"
		"    // dump before trying to access memory\n"
        "    tfp->dump(main_time * 2);\n"
		"    if (!top->reset && top->mem_req_valid) {\n"
		"        mem_resp_valid = true;\n"
		"        void *addr = (void *) top->mem_req_addr;\n"
		"        uint64_t data = top->mem_req_data;\n"
		"        if (top->mem_req_write) {\n"
		"#ifdef HLS_MEM_DEBUG\n"
		"            std::cout << \"writing \" << data << \" to \" << addr << \"\\n\";\n"
		"#endif\n"
		"            switch (top->mem_req_width) {\n"
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
		"        } else {\n"
		"#ifdef HLS_MEM_DEBUG\n"
		"            std::cout << \"reading from \" << addr << \"\\n\";\n"
		"#endif\n"
		"        }\n"
		"        switch (top->mem_req_width) {\n"
		"            case 0:\n"
		"                mem_resp_data = *(uint8_t *) addr;\n"
		"                break;\n"
		"            case 1:\n"
		"                mem_resp_data = *(uint16_t *) addr;\n"
		"                break;\n"
		"            case 2:\n"
		"                mem_resp_data = *(uint32_t *) addr;\n"
		"                break;\n"
		"            case 3:\n"
		"                mem_resp_data = *(uint64_t *) addr;\n"
		"                break;\n"
		"            default:\n"
		"                assert(false);\n"
		"        }\n"
		"    } else {\n"
		"        mem_resp_valid = false;\n"
		"    }\n"
		"    assert(!Verilated::gotFinish());\n"
		"    top->clk = 0;\n"
		"    top->eval();\n"
		"    tfp->dump(main_time * 2 + 1);\n"
		"    main_time++;\n"
		"}\n"
		"extern \"C\"\n"// TODO: parameter for linkage type here
		"{\n";
	// imports
	auto root = rm.Rvsdg().root();
	for (size_t i = 0; i < root->narguments(); ++i)
  {
    if (auto rvsdgImport = dynamic_cast<const impport*>(&root->argument(i)->port()))
    {
      JLM_ASSERT(is<PointerType>(rvsdgImport->type()));
      cpp << "extern " << convert_to_c_type(&rvsdgImport->GetValueType()) << " " << rvsdgImport->name() << ";\n";
    }
	}
	std::string return_type;
	if (ln->type().NumResults() == 0) {
		return_type = "void";
	} else {
		auto type = &ln->type().ResultType(0);
		if (dynamic_cast<const jive::statetype *>(type)) {
			return_type = "void";
		} else {
			return_type = convert_to_c_type(type);
		}
	}
	cpp << return_type << " " << function_name << "(\n";
	for (size_t i = 0; i < ln->type().NumArguments(); ++i) {
		if (dynamic_cast<const jive::statetype *>(&ln->type().ArgumentType(i))) {
			continue;
		}
		if (i != 0) {
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
	for (size_t i = 0; i < ln->type().NumArguments(); ++i) {
		if (dynamic_cast<const jive::statetype *>(&ln->type().ArgumentType(i))) {
			continue;
		}
		cpp << "    top->i_data" << i << " = (uint64_t) a" << i << ";\n";
	}
	for (size_t i = 0; i < ln->ncvarguments(); ++i) {
		size_t ix = ln->cvargument(i)->input()->argument()->index();
		std::string name;
		if(auto a = dynamic_cast<jive::argument *>(ln->input(i)->origin())){
			if(auto ip = dynamic_cast<const impport*>(&a->port())){
				name = ip->name();
			}
		} else{
			throw jlm::error("Unsupported cvarg origin type type");
		}
		cpp << "    top->i_data" << ix << " = (uint64_t) &" << name << ";\n";
		cpp << "#ifdef HLS_MEM_DEBUG\n";
		cpp << "    std::cout << \"" << name << ": \" << &" << name << " << \"\\n\";\n";
		cpp << "#endif\n";
	}
	cpp <<
		"    int start = main_time;\n"
		"    for (int i = 0; i < TIMEOUT && !top->i_ready; i++) {\n"
		"        clock_cycle();\n"
		"    }\n"
		"    if (!top->i_ready) {\n"
		"        std::cout << \"i_ready not set\\n\";\n"
		"        verilator_finish();\n"
		"        exit(-1);\n"
		"    }\n"
		"    top->i_valid = 1;\n"
		"    top->o_ready = 1;\n"
		"    clock_cycle();\n"
		"    top->i_valid = 0;\n";
	for (size_t i = 0; i < ln->type().NumArguments(); ++i) {
		cpp << "    top->i_data" << i << " = 0;\n";
	}
	cpp <<
		"    for (int i = 0; i < TIMEOUT && !top->o_valid; i++) {\n"
		"        clock_cycle();\n"
		"    }\n"
		"    if (!top->o_valid) {\n"
		"        std::cout << \"o_valid not set\\n\";\n"
		"        verilator_finish();\n"
		"        exit(-1);\n"
		"    }\n"
		"    std::cout << \"finished - took \" << (main_time - start) << \"cycles\\n\";\n";
	if (ln->type().NumResults() && !dynamic_cast<const jive::statetype *>(&ln->type().ResultType(0))) {
		cpp << "    return top->o_data0;\n";
	}
	cpp << "}\n}\n";
	return cpp.str();
}

std::string
jlm::hls::VerilatorHarnessHLS::convert_to_c_type(const jive::type *type) {
	if (auto t = dynamic_cast<const jive::bittype *>(type)) {
		return "int" + jive::detail::strfmt(t->nbits()) + "_t";
	} else if (is<PointerType>(*type)) {
		return "void*";
	} else if (auto t = dynamic_cast<const jlm::arraytype *>(type)) {
		return convert_to_c_type(&t->element_type());
	} else {
		throw std::logic_error(type->debug_string() + " not implemented!");
	}
}

std::string
jlm::hls::VerilatorHarnessHLS::convert_to_c_type_postfix(const jive::type *type) {
	if (auto t = dynamic_cast<const jlm::arraytype *>(type)) {
		return jive::detail::strfmt("[", t->nelements(), "]", convert_to_c_type(&t->element_type()));
	} else {
		return "";
	}
}
