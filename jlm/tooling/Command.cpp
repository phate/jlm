/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/dot/DotWriter.hpp>
#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/EliminatedModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownModRefEliminator.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/IfConversion.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandPaths.hpp>

#include <llvm/IR/Module.h>

#ifdef ENABLE_MLIR
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#endif

#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <sys/stat.h>

#include <fstream>
#include <unordered_map>

namespace jlm::tooling
{

Command::~Command() = default;

void
Command::Run() const
{
  const auto command = ToString();
  const auto returnCode = system(command.c_str());
  if (returnCode != EXIT_SUCCESS)
  {
    throw util::error(
        util::strfmt("Subcommand failed with status code ", returnCode, ": ", command));
  }
}

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

JlmOptCommand::JlmOptCommand(
    std::string programName,
    const jlm::tooling::JlmOptCommandLineOptions & commandLineOptions)
    : ProgramName_(std::move(programName)),
      CommandLineOptions_(std::move(commandLineOptions))
{
  for (auto optimizationId : CommandLineOptions_.GetOptimizationIds())
  {
    if (auto it = Optimizations_.find(optimizationId); it == Optimizations_.end())
      Optimizations_[optimizationId] = CreateTransformation(optimizationId);
  }
}

std::string
JlmOptCommand::ToString() const
{
  std::string optimizationArguments;
  for (auto & optimization : CommandLineOptions_.GetOptimizationIds())
    optimizationArguments +=
        "--" + std::string(JlmOptCommandLineOptions::ToCommandLineArgument(optimization)) + " ";

  auto inputFormatArgument = "--input-format="
                           + std::string(JlmOptCommandLineOptions::ToCommandLineArgument(
                               CommandLineOptions_.GetInputFormat()))
                           + " ";

  auto outputFormatArgument = "--output-format="
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
      "-s " + CommandLineOptions_.GetStatisticsCollectorSettings().GetOutputDirectory().to_str()
      + " ";

  return util::strfmt(
      ProgramName_,
      " ",
      inputFormatArgument,
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
  jlm::util::StatisticsCollector statisticsCollector(
      CommandLineOptions_.GetStatisticsCollectorSettings());

  auto rvsdgModule = ParseInputFile(
      CommandLineOptions_.GetInputFile(),
      CommandLineOptions_.GetInputFormat(),
      statisticsCollector);

  rvsdg::TransformationSequence::CreateAndRun(
      *rvsdgModule,
      statisticsCollector,
      GetTransformations());

  PrintRvsdgModule(
      *rvsdgModule,
      CommandLineOptions_.GetOutputFile(),
      CommandLineOptions_.GetOutputFormat(),
      statisticsCollector);

  statisticsCollector.PrintStatistics();
}

std::vector<rvsdg::Transformation *>
JlmOptCommand::GetTransformations() const
{
  std::vector<rvsdg::Transformation *> optimizations;
  for (auto optimizationId : CommandLineOptions_.GetOptimizationIds())
  {
    auto it = Optimizations_.find(optimizationId);
    JLM_ASSERT(it != Optimizations_.end());
    optimizations.emplace_back(it->second.get());
  }

  return optimizations;
}

std::unique_ptr<rvsdg::Transformation>
JlmOptCommand::CreateTransformation(
    enum JlmOptCommandLineOptions::OptimizationId optimizationId) const
{
  using Andersen = llvm::aa::Andersen;
  using Steensgaard = llvm::aa::Steensgaard;
  using AgnosticMrs = llvm::aa::AgnosticModRefSummarizer;
  using RegionAwareMrs = llvm::aa::RegionAwareModRefSummarizer;
  using TopDownLifetimeMrs =
      llvm::aa::EliminatedModRefSummarizer<AgnosticMrs, llvm::aa::TopDownModRefEliminator>;

  switch (optimizationId)
  {
  case JlmOptCommandLineOptions::OptimizationId::AAAndersenAgnostic:
    return std::make_unique<llvm::aa::PointsToAnalysisStateEncoder<Andersen, AgnosticMrs>>();
  case JlmOptCommandLineOptions::OptimizationId::AAAndersenRegionAware:
    return std::make_unique<llvm::aa::PointsToAnalysisStateEncoder<Andersen, RegionAwareMrs>>();
  case JlmOptCommandLineOptions::OptimizationId::AAAndersenTopDownLifetimeAware:
    return std::make_unique<llvm::aa::PointsToAnalysisStateEncoder<Andersen, TopDownLifetimeMrs>>();
  case JlmOptCommandLineOptions::OptimizationId::AASteensgaardAgnostic:
    return std::make_unique<llvm::aa::PointsToAnalysisStateEncoder<Steensgaard, AgnosticMrs>>();
  case JlmOptCommandLineOptions::OptimizationId::AASteensgaardRegionAware:
    return std::make_unique<llvm::aa::PointsToAnalysisStateEncoder<Steensgaard, RegionAwareMrs>>();
  case JlmOptCommandLineOptions::OptimizationId::CommonNodeElimination:
    return std::make_unique<llvm::cne>();
  case JlmOptCommandLineOptions::OptimizationId::DeadNodeElimination:
    return std::make_unique<llvm::DeadNodeElimination>();
  case JlmOptCommandLineOptions::OptimizationId::FunctionInlining:
    return std::make_unique<llvm::fctinline>();
  case JlmOptCommandLineOptions::OptimizationId::IfConversion:
    return std::make_unique<llvm::IfConversion>();
  case JlmOptCommandLineOptions::OptimizationId::InvariantValueRedirection:
    return std::make_unique<llvm::InvariantValueRedirection>();
  case JlmOptCommandLineOptions::OptimizationId::LoopUnrolling:
    return std::make_unique<llvm::loopunroll>(4);
  case JlmOptCommandLineOptions::OptimizationId::NodePullIn:
    return std::make_unique<llvm::pullin>();
  case JlmOptCommandLineOptions::OptimizationId::NodePushOut:
    return std::make_unique<llvm::pushout>();
  case JlmOptCommandLineOptions::OptimizationId::NodeReduction:
    return std::make_unique<llvm::NodeReduction>();
  case JlmOptCommandLineOptions::OptimizationId::RvsdgTreePrinter:
    return std::make_unique<llvm::RvsdgTreePrinter>(
        CommandLineOptions_.GetRvsdgTreePrinterConfiguration());
  case JlmOptCommandLineOptions::OptimizationId::ThetaGammaInversion:
    return std::make_unique<llvm::LoopUnswitching>();
  default:
    JLM_UNREACHABLE("Unhandled optimization id.");
  }
}

std::unique_ptr<llvm::RvsdgModule>
JlmOptCommand::ParseLlvmIrFile(
    const util::FilePath & llvmIrFile,
    util::StatisticsCollector & statisticsCollector) const
{
  ::llvm::LLVMContext llvmContext;
  ::llvm::SMDiagnostic diagnostic;
  auto llvmModule = ::llvm::parseIRFile(llvmIrFile.to_str(), diagnostic, llvmContext);

  if (llvmModule == nullptr)
  {
    std::string errors;
    ::llvm::raw_string_ostream os(errors);
    diagnostic.print(ProgramName_.c_str(), os);
    throw util::error(errors);
  }

  auto interProceduralGraphModule = llvm::ConvertLlvmModule(*llvmModule);

  // Dispose of Llvm module. It is no longer needed.
  llvmModule.reset();

  auto rvsdgModule =
      llvm::ConvertInterProceduralGraphModule(*interProceduralGraphModule, statisticsCollector);

  return rvsdgModule;
}

std::unique_ptr<llvm::RvsdgModule>
JlmOptCommand::ParseMlirIrFile(const util::FilePath & mlirIrFile, util::StatisticsCollector &) const
{
#ifdef ENABLE_MLIR
  jlm::mlir::MlirToJlmConverter rvsdggen;
  return rvsdggen.ReadAndConvertMlir(mlirIrFile);
#else
  JLM_UNREACHABLE(
      "This version of jlm-opt has not been compiled with support for the MLIR backend\n");
#endif
}

std::unique_ptr<llvm::RvsdgModule>
JlmOptCommand::ParseInputFile(
    const util::FilePath & inputFile,
    const JlmOptCommandLineOptions::InputFormat & inputFormat,
    util::StatisticsCollector & statisticsCollector) const
{
  if (inputFormat == tooling::JlmOptCommandLineOptions::InputFormat::Llvm)
  {
    return ParseLlvmIrFile(inputFile, statisticsCollector);
  }
  else if (inputFormat == tooling::JlmOptCommandLineOptions::InputFormat::Mlir)
  {
    return ParseMlirIrFile(inputFile, statisticsCollector);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled input format.");
  }
}

void
JlmOptCommand::PrintAsAscii(
    const llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    util::StatisticsCollector &)
{
  auto ascii = view(&rvsdgModule.Rvsdg().GetRootRegion());

  if (outputFile == "")
  {
    std::cout << ascii << std::flush;
  }
  else
  {
    std::ofstream fileStream(outputFile.to_str());
    fileStream << ascii;
    fileStream.close();
  }
}

void
JlmOptCommand::PrintAsXml(
    const llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    util::StatisticsCollector &)
{
  auto fd = outputFile == "" ? stdout : fopen(outputFile.to_str().c_str(), "w");

  view_xml(&rvsdgModule.Rvsdg().GetRootRegion(), fd);

  if (fd != stdout)
    fclose(fd);
}

void
JlmOptCommand::PrintAsLlvm(
    llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    util::StatisticsCollector & statisticsCollector)
{
  auto jlm_module =
      llvm::RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);

  ::llvm::LLVMContext ctx;
  auto llvm_module = llvm::IpGraphToLlvmConverter::CreateAndConvertModule(*jlm_module, ctx);

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
}

void
JlmOptCommand::PrintAsMlir(
    const llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    util::StatisticsCollector &)
{
#ifdef ENABLE_MLIR
  jlm::mlir::JlmToMlirConverter mlirgen;
  auto omega = mlirgen.ConvertModule(rvsdgModule);
  mlirgen.Print(omega, outputFile);
#else
  throw util::error(
      "This version of jlm-opt has not been compiled with support for the MLIR backend\n");
#endif
}

void
JlmOptCommand::PrintAsRvsdgTree(
    const llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    util::StatisticsCollector &)
{
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();
  auto tree = rvsdg::Region::ToTree(rootRegion);

  if (outputFile == "")
  {
    std::cout << tree << std::flush;
  }
  else
  {
    std::ofstream fs;
    fs.open(outputFile.to_str());
    fs << tree;
    fs.close();
  }
}

void
JlmOptCommand::PrintAsDot(
    const llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    util::StatisticsCollector &)
{
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  util::GraphWriter writer;
  jlm::llvm::dot::WriteGraphs(writer, rootRegion, true);

  if (outputFile == "")
  {
    writer.OutputAllGraphs(std::cout, util::GraphOutputFormat::Dot);
  }
  else
  {
    std::ofstream fs;
    fs.open(outputFile.to_str());
    writer.OutputAllGraphs(fs, util::GraphOutputFormat::Dot);
    fs.close();
  }
}

void
JlmOptCommand::PrintRvsdgModule(
    llvm::RvsdgModule & rvsdgModule,
    const util::FilePath & outputFile,
    const JlmOptCommandLineOptions::OutputFormat & outputFormat,
    util::StatisticsCollector & statisticsCollector)
{
  if (outputFormat == tooling::JlmOptCommandLineOptions::OutputFormat::Ascii)
  {
    PrintAsAscii(rvsdgModule, outputFile, statisticsCollector);
  }
  else if (outputFormat == tooling::JlmOptCommandLineOptions::OutputFormat::Xml)
  {
    PrintAsXml(rvsdgModule, outputFile, statisticsCollector);
  }
  else if (outputFormat == tooling::JlmOptCommandLineOptions::OutputFormat::Llvm)
  {
    PrintAsLlvm(rvsdgModule, outputFile, statisticsCollector);
  }
  else if (outputFormat == tooling::JlmOptCommandLineOptions::OutputFormat::Mlir)
  {
    PrintAsMlir(rvsdgModule, outputFile, statisticsCollector);
  }
  else if (outputFormat == tooling::JlmOptCommandLineOptions::OutputFormat::Tree)
  {
    PrintAsRvsdgTree(rvsdgModule, outputFile, statisticsCollector);
  }
  else if (outputFormat == tooling::JlmOptCommandLineOptions::OutputFormat::Dot)
  {
    PrintAsDot(rvsdgModule, outputFile, statisticsCollector);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled output format.");
  }
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
      clangpath.Dirname().Join("opt").to_str(),
      " ",
      optimizationArguments,
      WriteLlvmAssembly_ ? "-S " : "",
      "-o ",
      OutputFile().to_str(),
      " ",
      InputFile_.to_str());
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
      clangpath.Dirname().Join("llvm-link").to_str(),
      " ",
      WriteLlvmAssembly_ ? "-S " : "",
      Verbose_ ? "-v " : "",
      "-o ",
      OutputFile_.to_str(),
      " ",
      inputFilesArgument);
}

JlmHlsCommand::~JlmHlsCommand() noexcept = default;

std::string
JlmHlsCommand::ToString() const
{
  std::string options;
  for (auto & o : Options)
  {
    options += o + " ";
  }
  return util::strfmt("jlm-hls ", options, "-o ", OutputFolder_.to_str(), " ", InputFile_.to_str());
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

}
