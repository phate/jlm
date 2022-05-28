/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMAND_HPP
#define JLM_TOOLING_COMMAND_HPP

#include <jlm/tooling/CommandGraph.hpp>
#include <jlm/util/file.hpp>

#include <memory>
#include <string>

namespace jlm {

/** \brief Command class
 *
 * This class represents simple commands, such as \a mkdir or \a rm, that can be executed with the Run() method.
 */
class Command {
public:
  virtual
  ~Command();

  [[nodiscard]] virtual std::string
  ToString() const = 0;

  virtual void
  Run() const = 0;
};

/**
 * The PrintCommandsCommand class prints the commands of a command graph in topological order.
 */
class PrintCommandsCommand final : public Command {
public:
  ~PrintCommandsCommand() override;

  explicit
  PrintCommandsCommand(std::unique_ptr<CommandGraph> commandGraph)
    : CommandGraph_(std::move(commandGraph))
  {}

  PrintCommandsCommand(const PrintCommandsCommand&) = delete;

  PrintCommandsCommand(PrintCommandsCommand&&) = delete;

  PrintCommandsCommand &
  operator=(const PrintCommandsCommand&) = delete;

  PrintCommandsCommand &
  operator=(PrintCommandsCommand&&)	= delete;

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  static std::unique_ptr<CommandGraph>
  Create(std::unique_ptr<CommandGraph> commandGraph);

private:
  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    std::unique_ptr<CommandGraph> printedCommandGraph)
  {
    auto command = std::make_unique<PrintCommandsCommand>(std::move(printedCommandGraph));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

  std::unique_ptr<CommandGraph> CommandGraph_;
};

/**
 * The ClangCommand class represents the clang command line tool.
 */
class ClangCommand final : public Command {
public:
  enum class LanguageStandard {
    Unspecified,
    Gnu89,
    Gnu99,
    C89,
    C99,
    C11,
    Cpp98,
    Cpp03,
    Cpp11,
    Cpp14
  };

  enum class ClangArgument {
    DisableO0OptNone
  };

  ~ClangCommand() override;

  ClangCommand(
    std::vector<jlm::filepath> inputFiles,
    filepath outputFile,
    std::vector<std::string> libraryPaths,
    std::vector<std::string> libraries,
    bool usePthreads)
    : InputFiles_(std::move(inputFiles))
    , OutputFile_(std::move(outputFile))
    , DependencyFile_("")
    , Libraries_(std::move(libraries))
    , LibraryPaths_(std::move(libraryPaths))
    , UsePthreads_(usePthreads)
    , Verbose_(false)
    , Rdynamic_(false)
    , Suppress_(false)
    , Md_(false)
    , LanguageStandard_(LanguageStandard::Unspecified)
    , LinkerCommand_(true)
  {}

  ClangCommand(
    const jlm::filepath & inputFile,
    filepath outputFile,
    filepath dependencyFile,
    std::vector<std::string> includePaths,
    std::vector<std::string> macroDefinitions,
    std::vector<std::string> warnings,
    std::vector<std::string> flags,
    bool verbose,
    bool rdynamic,
    bool suppress,
    bool usePthreads,
    bool mD,
    std::string mT,
    const LanguageStandard & languageStandard,
    std::vector<ClangArgument> clangArguments)
    : InputFiles_({inputFile})
    , OutputFile_(std::move(outputFile))
    , DependencyFile_(std::move(dependencyFile))
    , IncludePaths_(std::move(includePaths))
    , MacroDefinitions_(std::move(macroDefinitions))
    , Warnings_(std::move(warnings))
    , Flags_(std::move(flags))
    , UsePthreads_(usePthreads)
    , Verbose_(verbose)
    , Rdynamic_(rdynamic)
    , Suppress_(suppress)
    , Md_(mD)
    , Mt_(std::move(mT))
    , LanguageStandard_(languageStandard)
    , ClangArguments_(std::move(clangArguments))
    , LinkerCommand_(false)
  {}


  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] const std::vector<filepath> &
  InputFiles() const noexcept
  {
    return InputFiles_;
  }

  static CommandGraph::Node &
  CreateLinkerCommand(
    CommandGraph & commandGraph,
    const std::vector<jlm::filepath> & inputFiles,
    const filepath & outputFile,
    const std::vector<std::string> & libraryPaths,
    const std::vector<std::string> & libraries,
    bool usePthreads)
  {
    std::unique_ptr<ClangCommand> command(new ClangCommand(
      inputFiles,
      outputFile,
      libraryPaths,
      libraries,
      usePthreads));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

  static CommandGraph::Node &
  CreateParsingCommand(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile,
    const filepath & dependencyFile,
    const std::vector<std::string> & includePaths,
    const std::vector<std::string> & macroDefinitions,
    const std::vector<std::string> & warnings,
    const std::vector<std::string> & flags,
    bool verbose,
    bool rdynamic,
    bool suppress,
    bool usePthread,
    bool mD,
    const std::string & mT,
    const LanguageStandard & languageStandard,
    const std::vector<ClangArgument> & clangArguments)
  {
    std::unique_ptr<ClangCommand> command(new ClangCommand(
      inputFile,
      outputFile,
      dependencyFile,
      includePaths,
      macroDefinitions,
      warnings,
      flags,
      verbose,
      rdynamic,
      suppress,
      usePthread,
      mD,
      mT,
      languageStandard,
      clangArguments));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  static std::string
  ToString(const LanguageStandard & languageStandard);

  static std::string
  ToString(const ClangArgument & clangArgument);

  static std::string
  ReplaceAll(std::string str, const std::string& from, const std::string& to);

  std::vector<filepath> InputFiles_;
  filepath OutputFile_;
  filepath DependencyFile_;

  std::vector<std::string> IncludePaths_;
  std::vector<std::string> MacroDefinitions_;
  std::vector<std::string> Warnings_;
  std::vector<std::string> Flags_;
  std::vector<std::string> Libraries_;
  std::vector<std::string> LibraryPaths_;

  bool UsePthreads_;
  bool Verbose_;
  bool Rdynamic_;
  bool Suppress_;
  bool Md_;
  std::string Mt_;

  LanguageStandard LanguageStandard_;
  std::vector<ClangArgument> ClangArguments_;

  bool LinkerCommand_;
};

/**
 * The LlcCommand class represents the llc command line tool.
 */
class LlcCommand final : public Command {
public:
  enum class OptimizationLevel {
    O0,
    O1,
    O2,
    O3
  };

  enum class RelocationModel {
    Static,
    Pic
  };

  ~LlcCommand() override;

  LlcCommand(
    filepath inputFile,
    filepath outputFile,
    const OptimizationLevel & optimizationLevel,
    const RelocationModel & relocationModel)
    : OptimizationLevel_(optimizationLevel)
    , RelocationModel_(relocationModel)
    , InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile,
    const OptimizationLevel & optimizationLevel,
    const RelocationModel & relocationModel)
  {
    std::unique_ptr<LlcCommand> command(new LlcCommand(
      inputFile,
      outputFile,
      optimizationLevel,
      relocationModel));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  static std::string
  ToString(const OptimizationLevel & optimizationLevel);

  static std::string
  ToString(const RelocationModel & relocationModel);

  OptimizationLevel OptimizationLevel_;
  RelocationModel RelocationModel_;
  filepath InputFile_;
  filepath OutputFile_;
};

/**
 * The JlmOptCommand class represents the jlm-opt command line tool.
 */
class JlmOptCommand final : public Command {
public:
  enum class Optimization {
    AASteensgaardBasic,
    CommonNodeElimination,
    DeadNodeElimination,
    FunctionInlining,
    InvariantValueRedirection,
    LoopUnrolling,
    NodePullIn,
    NodePushOut,
    NodeReduction,
    ThetaGammaInversion
  };

  ~JlmOptCommand() override;

  JlmOptCommand(
    filepath inputFile,
    filepath outputFile,
    std::vector<Optimization> optimizations)
    : InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
    , Optimizations_(std::move(optimizations))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile,
    const std::vector<Optimization> & optimizations)
  {
    std::unique_ptr<JlmOptCommand> command(new JlmOptCommand(inputFile, outputFile, optimizations));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  static std::string
  ToString(const Optimization & optimization);

  filepath InputFile_;
  filepath OutputFile_;
  std::vector<Optimization> Optimizations_;
};

/**
 * The MkdirCommand class represents the mkdir command line tool.
 */
class MkdirCommand final : public Command {
public:
  ~MkdirCommand() noexcept override;

  explicit
  MkdirCommand(filepath path)
    : Path_(std::move(path))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & path)
  {
    std::unique_ptr<MkdirCommand> command(new MkdirCommand(path));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  filepath Path_;
};

/**
 * The LlvmOptCommand class represents the LLVM opt command line tool.
 */
class LlvmOptCommand final : public Command {
public:
  enum class Optimization {
    Mem2Reg,
  };

  ~LlvmOptCommand() noexcept override;

  LlvmOptCommand(
    filepath inputFile,
    filepath outputFile,
    bool writeLlvmAssembly,
    std::vector<Optimization> optimizations)
    : InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
    , WriteLlvmAssembly_(writeLlvmAssembly)
    , Optimizations_(std::move(optimizations))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  void
  Run() const override;

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile,
    bool writeLlvmAssembly,
    const std::vector<Optimization> & optimizations)
  {
    std::unique_ptr<LlvmOptCommand> command(new LlvmOptCommand(
      inputFile,
      outputFile,
      writeLlvmAssembly,
      optimizations));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  static std::string
  ToString(const Optimization & optimization);

  filepath InputFile_;
  filepath OutputFile_;

  bool WriteLlvmAssembly_;

  std::vector<Optimization> Optimizations_;
};

/**
 * The LlvmLinkCommand class represents the llvm-link command line tool.
 */
class LlvmLinkCommand final : public Command {
public:
  ~LlvmLinkCommand() noexcept override;

  LlvmLinkCommand(
    std::vector<filepath> inputFiles,
    filepath outputFile,
    bool writeLlvmAssembly,
    bool verbose)
    : OutputFile_(std::move(outputFile))
    , InputFiles_(std::move(inputFiles))
    , WriteLlvmAssembly_(writeLlvmAssembly)
    , Verbose_(verbose)
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] const std::vector<filepath> &
  InputFiles() const noexcept
  {
    return InputFiles_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const std::vector<filepath> & inputFiles,
    const filepath & outputFile,
    bool writeLlvmAssembly,
    bool verbose)
  {
    std::unique_ptr<LlvmLinkCommand> command(new LlvmLinkCommand(
      inputFiles,
      outputFile,
      writeLlvmAssembly,
      verbose));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  filepath OutputFile_;
  std::vector<filepath> InputFiles_;

  bool WriteLlvmAssembly_;
  bool Verbose_;
};

/**
 * The JlmHlsCommand class represents the jlm-hls command line tool.
 */
class JlmHlsCommand final : public Command {
public:
  ~JlmHlsCommand() noexcept override;

  JlmHlsCommand(
    filepath inputFile,
    filepath outputFolder,
    bool useCirct)
    : InputFile_(std::move(inputFile))
    , OutputFolder_(std::move(outputFolder))
    , UseCirct_(useCirct)
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] filepath
  FirrtlFile() const noexcept
  {
    return OutputFolder_.to_str() + "/jlm_hls.fir";
  }

  [[nodiscard]] filepath
  LlvmFile() const noexcept
  {
    return OutputFolder_.to_str() + "/jlm_hls_rest.ll";
  }

  [[nodiscard]] filepath
  HarnessFile() const noexcept
  {
    return OutputFolder_.to_str() + "/jlm_hls_harness.cpp";
  }

  [[nodiscard]] const filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFolder,
    bool useCirct)
  {
    std::unique_ptr<JlmHlsCommand> command(new JlmHlsCommand(
      inputFile,
      outputFolder,
      useCirct));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  filepath InputFile_;
  filepath OutputFolder_;
  bool UseCirct_;
};

/**
 * The JlmHlsExtractCommand class represents the jlm-hls command line tool with the --extract command line argument
 * provided.
 */
class JlmHlsExtractCommand final : public Command {
public:
  ~JlmHlsExtractCommand() noexcept override;

  JlmHlsExtractCommand(
    filepath inputFile,
    filepath outputFolder,
    std::string hlsFunctionName)
    : InputFile_(std::move(inputFile))
    , OutputFolder_(std::move(outputFolder))
    , HlsFunctionName_(std::move(hlsFunctionName))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] filepath
  HlsFunctionFile() const noexcept
  {
    return OutputFolder_.to_str() + "/jlm_hls_function.ll";
  }

  [[nodiscard]] filepath
  LlvmFile() const noexcept
  {
    return OutputFolder_.to_str() + "/jlm_hls_rest.ll";
  }

  [[nodiscard]] const filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] const std::string &
  HlsFunctionName() const noexcept
  {
    return HlsFunctionName_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const std::string & hlsFunctionName,
    const filepath & outputFolder)
  {
    std::unique_ptr<JlmHlsExtractCommand> command(new JlmHlsExtractCommand(
      inputFile,
      outputFolder,
      hlsFunctionName));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  filepath InputFile_;
  filepath OutputFolder_;

  std::string HlsFunctionName_;

};

/**
 * The FirtoolCommand class represents the firtool command line tool.
 */
class FirtoolCommand final : public Command {
public:
  ~FirtoolCommand() noexcept override;

  FirtoolCommand(
    filepath inputFile,
    filepath outputFile)
    : OutputFile_(std::move(outputFile))
    , InputFile_(std::move(inputFile))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] const filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile)
  {
    std::unique_ptr<FirtoolCommand> command(new FirtoolCommand(inputFile, outputFile));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  filepath OutputFile_;
  filepath InputFile_;
};

class verilatorcmd final : public Command {
public:
  virtual
  ~verilatorcmd(){}

  verilatorcmd(
    const jlm::filepath & vfile,
    const std::vector<jlm::filepath> & lfiles,
    const jlm::filepath & hfile,
    const jlm::filepath & ofile,
    const jlm::filepath & tmpfolder,
    const std::vector<std::string> & Lpaths,
    const std::vector<std::string> & libs)
    : ofile_(ofile)
    , vfile_(vfile)
    , hfile_(hfile)
    , tmpfolder_(tmpfolder)
    , libs_(libs)
    , lfiles_(lfiles)
    , Lpaths_(Lpaths)
  {}

  virtual std::string
  ToString() const override;

  virtual void
  Run() const override;

  inline const jlm::filepath &
  vfile() const noexcept
  {
    return vfile_;
  }

  inline const jlm::filepath &
  hfile() const noexcept
  {
    return hfile_;
  }

  inline const jlm::filepath &
  ofile() const noexcept
  {
    return ofile_;
  }

  inline const std::vector<jlm::filepath> &
  lfiles() const noexcept
  {
    return lfiles_;
  }

  static CommandGraph::Node *
  create(
    CommandGraph * pgraph,
    const jlm::filepath & vfile,
    const std::vector<jlm::filepath> & lfiles,
    const jlm::filepath & hfile,
    const jlm::filepath & ofile,
    const jlm::filepath & tmpfolder,
    const std::vector<std::string> & Lpaths,
    const std::vector<std::string> & libs)
  {
    std::unique_ptr<verilatorcmd> cmd(new verilatorcmd(vfile, lfiles, hfile, ofile, tmpfolder, Lpaths, libs));
    return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
  }

private:
  jlm::filepath ofile_;
  jlm::filepath vfile_;
  jlm::filepath hfile_;
  jlm::filepath tmpfolder_;
  std::vector<std::string> libs_;
  std::vector<jlm::filepath> lfiles_;
  std::vector<std::string> Lpaths_;
};

}

#endif