/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/OptimizationSequence.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandPaths.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <sys/stat.h>

#include <filesystem>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <unordered_map>

namespace jlm::tooling
{

Command::~Command() = default;

PrintCommandsCommand::~PrintCommandsCommand() = default;

std::string
PrintCommandsCommand::ToString() const
{
  return "PrintCommands";
}

void
PrintCommandsCommand::Run() const
{
  for (auto & node : CommandGraph::SortNodesTopological(*CommandGraph_))
  {
    if (node != &CommandGraph_->GetEntryNode() && node != &CommandGraph_->GetExitNode())
      std::cout << node->GetCommand().ToString() << "\n";
  }
}

std::unique_ptr<CommandGraph>
PrintCommandsCommand::Create(std::unique_ptr<CommandGraph> commandGraph)
{
  std::unique_ptr<CommandGraph> newCommandGraph(new CommandGraph());
  auto & printCommandsNode =
      PrintCommandsCommand::Create(*newCommandGraph, std::move(commandGraph));
  newCommandGraph->GetEntryNode().AddEdge(printCommandsNode);
  printCommandsNode.AddEdge(newCommandGraph->GetExitNode());
  return newCommandGraph;
}

ClangCommand::~ClangCommand() = default;

std::string
ClangCommand::ToString() const
{
  std::string inputFiles;
  for (auto & inputFile : InputFiles_)
    inputFiles += inputFile.to_str() + " ";

  std::string libraryPaths;
  for (auto & libraryPath : LibraryPaths_)
    libraryPaths += "-L" + libraryPath + " ";

  std::string libraries;
  for (const auto & library : Libraries_)
    libraries += "-l" + library + " ";

  std::string includePaths;
  for (auto & includePath : IncludePaths_)
    includePaths += "-I" + includePath + " ";

  std::string macroDefinitions;
  for (auto & macroDefinition : MacroDefinitions_)
    macroDefinitions += "-D" + macroDefinition + " ";

  std::string warnings;
  for (auto & warning : Warnings_)
    warnings += "-W" + warning + " ";

  std::string flags;
  for (auto & flag : Flags_)
    flags += "-f" + flag + " ";

  std::string arguments;
  if (UsePthreads_)
    arguments += "-pthread ";

  if (Verbose_)
    arguments += "-v ";

  if (Rdynamic_)
    arguments += "-rdynamic ";

  if (Suppress_)
    arguments += "-w ";

  if (Md_)
  {
    arguments += "-MD ";
    arguments += "-MF " + DependencyFile_.to_str() + " ";
    arguments += "-MT " + Mt_ + " ";
  }

  std::string languageStandardArgument = LanguageStandard_ != LanguageStandard::Unspecified
                                           ? "-std=" + ToString(LanguageStandard_) + " "
                                           : "";

  std::string clangArguments;
  if (!ClangArguments_.empty())
  {
    for (auto & clangArgument : ClangArguments_)
      clangArguments += "-Xclang " + ToString(clangArgument) + " ";
  }

  /*
   * TODO: Remove LinkerCommand_ attribute and merge both paths into a single strfmt() call.
   */
  if (LinkerCommand_)
  {
    return util::strfmt(
        clangpath.to_str() + " ",
        "-no-pie -O0 ",
        arguments,
        inputFiles,
        "-o ",
        OutputFile_.to_str(),
        " ",
        libraryPaths,
        libraries);
  }
  else
  {
    return util::strfmt(
        clangpath.to_str() + " ",
        arguments,
        " ",
        warnings,
        " ",
        flags,
        " ",
        languageStandardArgument,
        ReplaceAll(macroDefinitions, std::string("\""), std::string("\\\"")),
        " ",
        includePaths,
        " ",
        "-S -emit-llvm ",
        clangArguments,
        "-o ",
        OutputFile_.to_str(),
        " ",
        inputFiles);
  }
}

std::string
ClangCommand::ToString(const LanguageStandard & languageStandard)
{
  static std::unordered_map<LanguageStandard, const char *> map(
      { { LanguageStandard::Unspecified, "" },
        { LanguageStandard::Gnu89, "gnu89" },
        { LanguageStandard::Gnu99, "gnu99" },
        { LanguageStandard::C89, "c89" },
        { LanguageStandard::C99, "c99" },
        { LanguageStandard::C11, "c11" },
        { LanguageStandard::Cpp98, "c++98" },
        { LanguageStandard::Cpp03, "c++03" },
        { LanguageStandard::Cpp11, "c++11" },
        { LanguageStandard::Cpp14, "c++14" } });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

std::string
ClangCommand::ToString(const ClangArgument & clangArgument)
{
  static std::unordered_map<ClangArgument, const char *> map({
      { ClangArgument::DisableO0OptNone, "-disable-O0-optnone" },
  });

  JLM_ASSERT(map.find(clangArgument) != map.end());
  return map[clangArgument];
}

std::string
ClangCommand::ReplaceAll(std::string str, const std::string & from, const std::string & to)
{
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos)
  {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}

void
ClangCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

LlcCommand::~LlcCommand() = default;

std::string
LlcCommand::ToString() const
{
  return util::strfmt(
      llcpath.to_str() + " ",
      "-",
      ToString(OptimizationLevel_),
      " ",
      "--relocation-model=",
      ToString(RelocationModel_),
      " ",
      "-filetype=obj ",
      "-o ",
      OutputFile_.to_str(),
      " ",
      InputFile_.to_str());
}

void
LlcCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

std::string
LlcCommand::ToString(const OptimizationLevel & optimizationLevel)
{
  static std::unordered_map<OptimizationLevel, const char *> map(
      { { OptimizationLevel::O0, "O0" },
        { OptimizationLevel::O1, "O1" },
        { OptimizationLevel::O2, "O2" },
        { OptimizationLevel::O3, "O3" } });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

std::string
LlcCommand::ToString(const RelocationModel & relocationModel)
{
  static std::unordered_map<RelocationModel, const char *> map({
      { RelocationModel::Static, "static" },
      { RelocationModel::Pic, "pic" },
  });

  JLM_ASSERT(map.find(relocationModel) != map.end());
  return map[relocationModel];
}

JlmOptCommand::~JlmOptCommand() = default;

std::string
JlmOptCommand::ToString() const
{
  std::string optimizationArguments;
  for (auto & optimization : CommandLineOptions_.GetOptimizationIds())
    optimizationArguments +=
        "--" + std::string(JlmOptCommandLineOptions::ToCommandLineArgument(optimization)) + " ";

  auto outputFormatArgument = "--"
                            + std::string(JlmOptCommandLineOptions::ToCommandLineArgument(
                                CommandLineOptions_.GetOutputFormat()))
                            + " ";

  auto outputFileArgument = !CommandLineOptions_.GetOutputFile().to_str().empty()
                              ? "-o " + CommandLineOptions_.GetOutputFile().to_str() + " "
                              : "";

  std::string statisticsArguments;
  auto & demandedStatistics =
      CommandLineOptions_.GetStatisticsCollectorSettings().GetDemandedStatistics();
  for (auto & statisticsId : demandedStatistics.Items())
  {
    statisticsArguments +=
        "--" + std::string(JlmOptCommandLineOptions::ToCommandLineArgument(statisticsId)) + " ";
  }

  std::string statisticsDirArgument =
      "-s " + CommandLineOptions_.GetStatisticsCollectorSettings().GetFilePath().path() + " ";

  return util::strfmt(
      ProgramName_ + " ",
      outputFormatArgument,
      optimizationArguments,
      statisticsDirArgument,
      statisticsArguments,
      outputFileArgument,
      CommandLineOptions_.GetInputFile().to_str());
}

void
JlmOptCommand::Run() const
{
  ::llvm::LLVMContext llvmContext;
  auto llvmModule = ParseLlvmIrFile(CommandLineOptions_.GetInputFile(), llvmContext);

  auto interProceduralGraphModule = llvm::ConvertLlvmModule(*llvmModule);

  /*
   * Dispose of Llvm module. It is no longer needed.
   */
  llvmModule.reset();

  jlm::util::StatisticsCollector statisticsCollector(
      CommandLineOptions_.GetStatisticsCollectorSettings());

  auto rvsdgModule =
      llvm::ConvertInterProceduralGraphModule(*interProceduralGraphModule, statisticsCollector);

  llvm::OptimizationSequence::CreateAndRun(
      *rvsdgModule,
      statisticsCollector,
      CommandLineOptions_.GetOptimizations());

  PrintRvsdgModule(
      *rvsdgModule,
      CommandLineOptions_.GetOutputFile(),
      CommandLineOptions_.GetOutputFormat(),
      statisticsCollector);

  statisticsCollector.PrintStatistics();
}

std::unique_ptr<::llvm::Module>
JlmOptCommand::ParseLlvmIrFile(const util::filepath & llvmIrFile, ::llvm::LLVMContext & llvmContext)
    const
{
  ::llvm::SMDiagnostic diagnostic;
  if (auto module = ::llvm::parseIRFile(llvmIrFile.to_str(), diagnostic, llvmContext))
  {
    return module;
  }

  std::string errors;
  ::llvm::raw_string_ostream os(errors);
  diagnostic.print(ProgramName_.c_str(), os);
  throw util::error(errors);
}

void
JlmOptCommand::PrintRvsdgModule(
    const llvm::RvsdgModule & rvsdgModule,
    const util::filepath & outputFile,
    const JlmOptCommandLineOptions::OutputFormat & outputFormat,
    util::StatisticsCollector & statisticsCollector)
{
  auto printAsXml = [](const llvm::RvsdgModule & rvsdgModule,
                       const util::filepath & outputFile,
                       util::StatisticsCollector &)
  {
    auto fd = outputFile == "" ? stdout : fopen(outputFile.to_str().c_str(), "w");

    jlm::rvsdg::view_xml(rvsdgModule.Rvsdg().root(), fd);

    if (fd != stdout)
      fclose(fd);
  };

  auto printAsLlvm = [](const llvm::RvsdgModule & rvsdgModule,
                        const util::filepath & outputFile,
                        util::StatisticsCollector & statisticsCollector)
  {
    auto jlm_module = llvm::rvsdg2jlm::rvsdg2jlm(rvsdgModule, statisticsCollector);

    ::llvm::LLVMContext ctx;
    auto llvm_module = jlm::llvm::jlm2llvm::convert(*jlm_module, ctx);

    if (outputFile == "")
    {
      ::llvm::raw_os_ostream os(std::cout);
      llvm_module->print(os, nullptr);
    }
    else
    {
      std::error_code ec;
      ::llvm::raw_fd_ostream os(outputFile.to_str(), ec);
      llvm_module->print(os, nullptr);
    }
  };

  static std::unordered_map<
      JlmOptCommandLineOptions::OutputFormat,
      std::function<
          void(const llvm::RvsdgModule &, const util::filepath &, util::StatisticsCollector &)>>
      printers({ { tooling::JlmOptCommandLineOptions::OutputFormat::Xml, printAsXml },
                 { tooling::JlmOptCommandLineOptions::OutputFormat::Llvm, printAsLlvm } });

  JLM_ASSERT(printers.find(outputFormat) != printers.end());
  printers[outputFormat](rvsdgModule, outputFile, statisticsCollector);
}

MkdirCommand::~MkdirCommand() noexcept = default;

std::string
MkdirCommand::ToString() const
{
  return util::strfmt("mkdir ", Path_.to_str());
}

void
MkdirCommand::Run() const
{
  if (mkdir(Path_.to_str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
    throw util::error("mkdir failed: " + Path_.to_str());
}

LlvmOptCommand::~LlvmOptCommand() noexcept = default;

std::string
LlvmOptCommand::ToString() const
{
  std::string optimizationArguments;
  if (!Optimizations_.empty())
  {
    optimizationArguments = "-passes=";
    bool first = true;
    for (auto & optimization : Optimizations_)
    {
      if (first)
      {
        optimizationArguments += ToString(optimization) + " ";
        first = false;
      }
      else
      {
        optimizationArguments += "," + ToString(optimization) + " ";
      }
    }
  }

  return util::strfmt(
      clangpath.path() + "opt ",
      optimizationArguments,
      WriteLlvmAssembly_ ? "-S " : "",
      "-o ",
      OutputFile().to_str(),
      " ",
      InputFile_.to_str());
}

void
LlvmOptCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

std::string
LlvmOptCommand::ToString(const Optimization & optimization)
{
  static std::unordered_map<Optimization, const char *> map({
      { Optimization::Mem2Reg, "mem2reg" },
  });

  JLM_ASSERT(map.find(optimization) != map.end());
  return map[optimization];
}

LlvmLinkCommand::~LlvmLinkCommand() noexcept = default;

std::string
LlvmLinkCommand::ToString() const
{
  std::string inputFilesArgument;
  for (auto & inputFile : InputFiles_)
    inputFilesArgument += inputFile.to_str() + " ";

  return util::strfmt(
      clangpath.path(),
      "llvm-link ",
      WriteLlvmAssembly_ ? "-S " : "",
      Verbose_ ? "-v " : "",
      "-o ",
      OutputFile_.to_str(),
      " ",
      inputFilesArgument);
}

void
LlvmLinkCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

JlmHlsCommand::~JlmHlsCommand() noexcept = default;

std::string
JlmHlsCommand::ToString() const
{
  return util::strfmt(
      "jlm-hls ",
      "-o ",
      OutputFolder_.to_str(),
      " ",
      UseCirct_ ? "--circt " : "",
      InputFile_.to_str());
}

void
JlmHlsCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

JlmHlsExtractCommand::~JlmHlsExtractCommand() noexcept = default;

std::string
JlmHlsExtractCommand::ToString() const
{
  return util::strfmt(
      "jlm-hls ",
      "--extract ",
      "--hls-function ",
      HlsFunctionName(),
      " ",
      " -o ",
      OutputFolder_.to_str(),
      " ",
      InputFile().to_str());
}

void
JlmHlsExtractCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

FirtoolCommand::~FirtoolCommand() noexcept = default;

std::string
FirtoolCommand::ToString() const
{
  return util::strfmt(
      firtoolpath.to_str(),
      " ",
      " -format=fir --verilog ",
      InputFile().to_str(),
      " > ",
      OutputFile().to_str());
}

void
FirtoolCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

VerilatorCommand::~VerilatorCommand() noexcept = default;

std::string
VerilatorCommand::ToString() const
{
  std::string objectFiles;
  for (auto & objectFile : ObjectFiles_)
    objectFiles += objectFile.to_str() + " ";

  std::string libraryPaths;
  for (auto & libraryPath : LibraryPaths_)
    libraryPaths += "-L" + libraryPath + " ";

  std::string libraries;
  for (const auto & library : Libraries_)
    libraries += "-l" + library + " ";

  std::string cflags;
  cflags = " -CFLAGS \"" + libraries + libraryPaths + " -fPIC\"";

  std::string outputFile = OutputFile_.to_str();
  if (outputFile.at(0) != '/')
  {
    outputFile = std::filesystem::current_path() / outputFile;
  }

  std::string verilator_root;
  if (!verilatorrootpath.to_str().empty())
  {
    verilator_root = util::strfmt("VERILATOR_ROOT=", verilatorrootpath.to_str(), " ");
  }

  return util::strfmt(
      verilator_root,
      verilatorpath.to_str(),
      " --cc",
      " --build",
      " --exe",
#ifndef HLS_USE_VCD
      " --trace-fst",
#else
      " --trace",
#endif
      " -Wno-WIDTH", // divisions otherwise cause errors
      " -j",
      " -Mdir ",
      TempFolder_.to_str(),
      "verilator/",
      " -MAKEFLAGS CXX=g++",
      " -CFLAGS -g", // TODO: switch for this
      " --assert",
      cflags,
      " -o ",
      outputFile,
      " ",
      VerilogFile().to_str(),
      " ",
      HarnessFile().to_str(),
      " ",
      objectFiles);
}

void
VerilatorCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

}
