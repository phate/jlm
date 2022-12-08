/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandPaths.hpp>
#include <jlm/util/strfmt.hpp>

#include <sys/stat.h>

#include <filesystem>
#include <unordered_map>

namespace jlm {

Command::~Command()
= default;

PrintCommandsCommand::~PrintCommandsCommand()
= default;

std::string
PrintCommandsCommand::ToString() const
{
  return "PrintCommands";
}

void
PrintCommandsCommand::Run() const
{
  for (auto & node : CommandGraph::SortNodesTopological(*CommandGraph_)) {
    if (node != &CommandGraph_->GetEntryNode() && node != &CommandGraph_->GetExitNode())
      std::cout << node->GetCommand().ToString() << "\n";
  }
}

std::unique_ptr<CommandGraph>
PrintCommandsCommand::Create(std::unique_ptr<CommandGraph> commandGraph)
{
  std::unique_ptr<CommandGraph> newCommandGraph(new CommandGraph());
  auto & printCommandsNode = PrintCommandsCommand::Create(*newCommandGraph, std::move(commandGraph));
  newCommandGraph->GetEntryNode().AddEdge(printCommandsNode);
  printCommandsNode.AddEdge(newCommandGraph->GetExitNode());
  return newCommandGraph;
}

ClangCommand::~ClangCommand()
= default;

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

  if (Md_) {
    arguments += "-MD ";
    arguments += "-MF " + DependencyFile_.to_str() + " ";
    arguments += "-MT " + Mt_ + " ";
  }

  std::string languageStandardArgument =
    LanguageStandard_ != LanguageStandard::Unspecified
    ? "-std="+ToString(LanguageStandard_)+" "
    : "";

  std::string clangArguments;
  if (!ClangArguments_.empty()) {
    for (auto & clangArgument : ClangArguments_)
      clangArguments += "-Xclang "+ToString(clangArgument)+" ";
  }

  /*
   * TODO: Remove LinkerCommand_ attribute and merge both paths into a single strfmt() call.
   */
  if (LinkerCommand_)
  {
    return strfmt(
      clangpath.to_str() + " ",
      "-no-pie -O0 ",
      arguments,
      inputFiles,
      "-o ", OutputFile_.to_str(), " ",
      libraryPaths,
      libraries);
  } else {
    return strfmt(
      clangpath.to_str() + " "
      , arguments, " "
      , warnings, " "
      , flags, " "
      , languageStandardArgument
      , ReplaceAll(macroDefinitions, std::string("\""), std::string("\\\"")), " "
      , includePaths, " "
      , "-S -emit-llvm "
      , clangArguments
      , "-o ", OutputFile_.to_str(), " "
      , inputFiles
    );
  }
}

std::string
ClangCommand::ToString(const LanguageStandard & languageStandard)
{
  static std::unordered_map<LanguageStandard, const char*>
    map({
          {LanguageStandard::Unspecified, ""},
          {LanguageStandard::Gnu89,       "gnu89"},
          {LanguageStandard::Gnu99,       "gnu99"},
          {LanguageStandard::C89,         "c89"},
          {LanguageStandard::C99,         "c99"},
          {LanguageStandard::C11,         "c11"},
          {LanguageStandard::Cpp98,       "c++98"},
          {LanguageStandard::Cpp03,       "c++03"},
          {LanguageStandard::Cpp11,       "c++11"},
          {LanguageStandard::Cpp14,       "c++14"}
        });

  JLM_ASSERT(map.find(languageStandard) != map.end());
  return map[languageStandard];
}

std::string
ClangCommand::ToString(const ClangArgument & clangArgument)
{
  static std::unordered_map<ClangArgument, const char*>
    map({
          {ClangArgument::DisableO0OptNone, "-disable-O0-optnone"},
        });

  JLM_ASSERT(map.find(clangArgument) != map.end());
  return map[clangArgument];
}

std::string
ClangCommand::ReplaceAll(std::string str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos) {
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

LlcCommand::~LlcCommand()
= default;

std::string
LlcCommand::ToString() const
{
  return strfmt(
    llcpath.to_str() + " "
    , "-", ToString(OptimizationLevel_), " "
    , "--relocation-model=", ToString(RelocationModel_), " "
    , "-filetype=obj "
    , "-o ", OutputFile_.to_str(), " "
    , InputFile_.to_str()
  );
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
  static std::unordered_map<OptimizationLevel, const char*>
    map({
          {OptimizationLevel::O0, "O0"},
          {OptimizationLevel::O1, "O1"},
          {OptimizationLevel::O2, "O2"},
          {OptimizationLevel::O3, "O3"}
        });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

std::string
LlcCommand::ToString(const RelocationModel & relocationModel)
{
  static std::unordered_map<RelocationModel, const char*>
    map({
          {RelocationModel::Static, "static"},
          {RelocationModel::Pic, "pic"},
        });

  JLM_ASSERT(map.find(relocationModel) != map.end());
  return map[relocationModel];
}

JlmOptCommand::~JlmOptCommand()
= default;

std::string
JlmOptCommand::ToString() const
{
  std::string optimizationArguments;
  for (auto & optimization : Optimizations_)
    optimizationArguments += ToString(optimization) + " ";

  return strfmt(
    "jlm-opt ",
    "--llvm ",
    optimizationArguments,
    "-o ", OutputFile_.to_str(), " ",
    InputFile_.to_str());
}

void
JlmOptCommand::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

std::string
JlmOptCommand::ToString(const Optimization & optimization)
{
  static std::unordered_map<Optimization, const char*>
    map({
          {Optimization::AASteensgaardAgnostic,     "--AASteensgaardAgnostic"},
          {Optimization::CommonNodeElimination,     "--cne"},
          {Optimization::DeadNodeElimination,       "--dne"},
          {Optimization::FunctionInlining,          "--iln"},
          {Optimization::InvariantValueRedirection, "--InvariantValueRedirection"},
          {Optimization::LoopUnrolling,             "--url"},
          {Optimization::NodePullIn,                "--pll"},
          {Optimization::NodePushOut,               "--psh"},
          {Optimization::NodeReduction,             "--red"},
          {Optimization::ThetaGammaInversion,       "--ivt"}
        });

  JLM_ASSERT(map.find(optimization) != map.end());
  return map[optimization];
}

MkdirCommand::~MkdirCommand() noexcept
= default;

std::string
MkdirCommand::ToString() const
{
  return strfmt(
    "mkdir ",
    Path_.to_str());
}

void
MkdirCommand::Run() const
{
  if (mkdir(Path_.to_str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
    throw error("mkdir failed: " + Path_.to_str());
}

LlvmOptCommand::~LlvmOptCommand() noexcept
= default;

std::string
LlvmOptCommand::ToString() const
{
  std::string optimizationArguments;
  for (auto & optimization : Optimizations_)
    optimizationArguments += ToString(optimization) + " ";

  return strfmt(
    clangpath.path() + "opt "
    , optimizationArguments
    , WriteLlvmAssembly_ ? "-S " : ""
    , "-o ", OutputFile().to_str(), " "
    , InputFile_.to_str()
  );
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
  static std::unordered_map<Optimization, const char*>
    map({
          {Optimization::Mem2Reg, "-mem2reg"},
        });

  JLM_ASSERT(map.find(optimization) != map.end());
  return map[optimization];
}

LlvmLinkCommand::~LlvmLinkCommand() noexcept
= default;

std::string
LlvmLinkCommand::ToString() const
{
  std::string inputFilesArgument;
  for (auto & inputFile : InputFiles_)
    inputFilesArgument += inputFile.to_str() + " ";

  return strfmt(
    clangpath.path(), "llvm-link ",
    WriteLlvmAssembly_ ? "-S " : "",
    Verbose_ ? "-v " : "",
    "-o ", OutputFile_.to_str(), " ",
    inputFilesArgument);
}

void
LlvmLinkCommand::Run() const {
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

JlmHlsCommand::~JlmHlsCommand() noexcept
= default;

std::string
JlmHlsCommand::ToString() const
{
  return strfmt(
    "jlm-hls ",
    "-o ", OutputFolder_.to_str(), " ",
    UseCirct_ ? "--circt " : "",
    InputFile_.to_str());
}

void
JlmHlsCommand::Run() const {
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

JlmHlsExtractCommand::~JlmHlsExtractCommand() noexcept
= default;

std::string
JlmHlsExtractCommand::ToString() const {
  return strfmt(
    "jlm-hls ",
    "--extract ",
    "--hls-function ", HlsFunctionName(), " ",
    " -o ", OutputFolder_.to_str(), " "
    , InputFile().to_str());
}

void
JlmHlsExtractCommand::Run() const {
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

FirtoolCommand::~FirtoolCommand() noexcept
= default;

std::string
FirtoolCommand::ToString() const {
  return strfmt(
    firtoolpath.to_str(), " ",
    " -format=fir --verilog ",
    InputFile().to_str(),
    " > ", OutputFile().to_str());
}

void
FirtoolCommand::Run() const {
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

VerilatorCommand::~VerilatorCommand() noexcept
= default;

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
  if(outputFile.at(0) != '/'){
    outputFile = std::filesystem::current_path() / outputFile;
  }

  std::string verilator_root;
  if(!verilatorrootpath.to_str().empty()){
    verilator_root = strfmt(
      "VERILATOR_ROOT="
      , verilatorrootpath.to_str()
      , " "
    );
  }

  return strfmt(
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
    " -Wno-WIDTH", //divisions otherwise cause errors
    " -j",
    " -Mdir ", TempFolder_.to_str(), "verilator/",
    " -MAKEFLAGS CXX=g++",
    " -CFLAGS -g", // TODO: switch for this
    " --assert",
    cflags,
    " -o ", outputFile,
    " ", VerilogFile().to_str(),
    " ", HarnessFile().to_str(),
    " ", objectFiles);
}

void
VerilatorCommand::Run() const {
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

}
