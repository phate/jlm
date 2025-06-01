/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMANDGRAPHGENERATOR_HPP
#define JLM_TOOLING_COMMANDGRAPHGENERATOR_HPP

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <memory>

namespace jlm::tooling
{

class CommandGraph;

/**
 * Interface for the generator of a command graph.
 */
template<class T>
class CommandGraphGenerator
{
  static_assert(
      std::is_base_of<CommandLineOptions, T>::value,
      "T is not derived from CommandLineOptions.");

public:
  virtual ~CommandGraphGenerator() noexcept = default;

  /**
   * Generate a command graph.
   *
   * @param commandLineOptions An instance of a CommandLineOptions class.
   * @return An instance of a CommandGraph class.
   */
  [[nodiscard]] virtual std::unique_ptr<CommandGraph>
  GenerateCommandGraph(const T & commandLineOptions) = 0;
};

/**
 * Command graph generator for the \a jlc command line tool.
 */
class JlcCommandGraphGenerator final : public CommandGraphGenerator<JlcCommandLineOptions>
{
public:
  ~JlcCommandGraphGenerator() noexcept override;

  JlcCommandGraphGenerator() = default;

  [[nodiscard]] std::unique_ptr<CommandGraph>
  GenerateCommandGraph(const JlcCommandLineOptions & commandLineOptions) override;

  [[nodiscard]] static std::unique_ptr<CommandGraph>
  Generate(const JlcCommandLineOptions & commandLineOptions)
  {
    JlcCommandGraphGenerator commandLineGenerator;
    return commandLineGenerator.GenerateCommandGraph(commandLineOptions);
  }

private:
  static util::FilePath
  CreateJlmOptCommandOutputFile(const util::FilePath & inputFile);

  static util::FilePath
  CreateParserCommandOutputFile(const util::FilePath & inputFile);

  static ClangCommand::LanguageStandard
  ConvertLanguageStandard(const JlcCommandLineOptions::LanguageStandard & languageStandard);

  static LlcCommand::OptimizationLevel
  ConvertOptimizationLevel(const JlcCommandLineOptions::OptimizationLevel & optimizationLevel);

  static CommandGraph::Node &
  CreateParserCommand(
      CommandGraph & commandGraph,
      const util::FilePath & outputFile,
      const JlcCommandLineOptions::Compilation & compilation,
      const JlcCommandLineOptions & commandLineOptions);
};

/**
 * Command graph generator for the \a jhls command line tool.
 */
class JhlsCommandGraphGenerator final : public CommandGraphGenerator<JhlsCommandLineOptions>
{
public:
  ~JhlsCommandGraphGenerator() noexcept override;

  JhlsCommandGraphGenerator() = default;

  [[nodiscard]] std::unique_ptr<CommandGraph>
  GenerateCommandGraph(const JhlsCommandLineOptions & commandLineOptions) override;

  [[nodiscard]] static std::unique_ptr<CommandGraph>
  Generate(const JhlsCommandLineOptions & commandLineOptions)
  {
    JhlsCommandGraphGenerator commandLineGenerator;
    return commandLineGenerator.GenerateCommandGraph(commandLineOptions);
  }

private:
  static util::FilePath
  CreateParserCommandOutputFile(
      const util::FilePath & tmpDirectory,
      const util::FilePath & inputFile);

  static util::FilePath
  CreateJlmOptCommandOutputFile(
      const util::FilePath & tmpDirectory,
      const util::FilePath & inputFile);

  static ClangCommand::LanguageStandard
  ConvertLanguageStandard(const JhlsCommandLineOptions::LanguageStandard & languageStandard);

  static LlcCommand::OptimizationLevel
  ConvertOptimizationLevel(const JhlsCommandLineOptions::OptimizationLevel & optimizationLevel);
};

}

#endif // JLM_TOOLING_COMMANDGRAPHGENERATOR_HPP
