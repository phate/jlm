/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <iostream>
#include <getopt.h>
#include <set>
#include <fstream>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <jive/view.hpp>

#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/frontend/llvm/InterProceduralGraphConversion.hpp>
#include <jlm/frontend/llvm/LlvmModuleConversion.hpp>
#include <jlm/backend/hls/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/backend/hls/rhls2firrtl/dot-hls.hpp>
#include <jlm/backend/hls/rhls2firrtl/firrtl-hls.hpp>
#include <jlm/backend/hls/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/backend/llvm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/backend/llvm/jlm2llvm/jlm2llvm.hpp>
#include <unistd.h>

class cmdflags {
public:
    inline
    cmdflags() {}

    std::string file;
    jlm::StatisticsDescriptor sd;
    std::string hls_function;
    std::string outfolder;
    bool circt = false;
};

static void
print_usage(const std::string &app) {
    std::cerr << "Usage: " << app << " [OPTIONS]\n";
    std::cerr << "OPTIONS:\n";
    std::cerr << "--file name: LLVM IR file.\n";
    std::cerr << "--hls-function: name of the function to turn into an accelerator\n";
}

static void
parse_cmdflags(int argc, char **argv, cmdflags &flags) {
    static constexpr size_t file = 1;
    static constexpr size_t hls_function = 2;
    static constexpr size_t outfolder = 3;

    static struct option options[] = {
            {"file", required_argument, NULL, file},
            {"hls-function", required_argument, NULL, hls_function},
            {"outfolder", required_argument, NULL, outfolder},
            {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long_only(argc, argv, "", options, NULL)) != -1) {
        switch (opt) {
            case file: {
                flags.file = optarg;
                break;
            }
            case hls_function: {
                flags.hls_function = optarg;
                break;
            }
            case outfolder: {
                flags.outfolder = optarg;
                break;
            }

            default:
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (flags.file.empty() || flags.hls_function.empty()) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
}
void
dump_xml(std::unique_ptr<jlm::RvsdgModule> &rvsdgModule, const std::string &suffix = "") {
	auto source_file_name = rvsdgModule->SourceFileName().name();
    std::string file_name = source_file_name.substr(0, source_file_name.find_last_of('.')) + suffix + ".rvsdg";
    auto xml_file = fopen(file_name.c_str(), "w");
    jive::view_xml(rvsdgModule->Rvsdg().root(), xml_file);
    fclose(xml_file);
}

void
dump_llvm(jlm::RvsdgModule &module, std::string outfile) {
    llvm::LLVMContext ctx;
    jlm::StatisticsDescriptor sd;
    auto jm = jlm::rvsdg2jlm::rvsdg2jlm(module, sd);
    auto lm = jlm::jlm2llvm::convert(*jm, ctx);
    std::error_code EC;
    llvm::raw_fd_ostream os(outfile, EC);
    lm->print(os, nullptr);
}

int
main(int argc, char **argv) {
    cmdflags flags;
    parse_cmdflags(argc, argv, flags);

    llvm::LLVMContext ctx;
    llvm::SMDiagnostic err;
    auto llvmModule = llvm::parseIRFile(flags.file, err, ctx);
    // change folder to redirect output
    if(!flags.outfolder.empty()){
        assert(chdir(flags.outfolder.c_str())==0);
    }
    if (!llvmModule) {
        err.print(argv[0], llvm::errs());
        exit(1);
    }

    /* LLVM to JLM pass */
    auto jlmModule = jlm::ConvertLlvmModule(*llvmModule);
    auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(*jlmModule, flags.sd);

    auto hlsFunction = jlm::hls::split_hls_function(*rvsdgModule, flags.hls_function);

    dump_llvm(*rvsdgModule, "jlm_hls.rest.ll");
    dump_llvm(*hlsFunction, "jlm_hls.function.ll");

    return 0;
}
