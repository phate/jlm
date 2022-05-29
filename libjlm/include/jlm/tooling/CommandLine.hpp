/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMANDLINE_HPP
#define JLM_TOOLING_COMMANDLINE_HPP

namespace jlm
{

/**
 * Interface for the command line options of a Jlm command line tool.
 */
class CommandLineOptions {
public:
  virtual
  ~CommandLineOptions();

  CommandLineOptions()
  = default;

  /**
   * Resets the state of the instance.
   */
  virtual void
  Reset() noexcept = 0;
};

}

#endif //JLM_TOOLING_COMMANDLINE_HPP