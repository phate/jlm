/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMHLS_CMDLINE_HPP
#define JLM_JLMHLS_CMDLINE_HPP

#include <jlm/util/file.hpp>

namespace jlm {

class JlmHlsCommandLineOptions {
public:
  enum class OutputFormat {
    Firrtl,
    Dot
  };

  JlmHlsCommandLineOptions()
    : InputFile_("")
    , OutputFolder_("")
    , OutputFormat_(OutputFormat::Firrtl)
    , ExtractHlsFunction_(false)
    , UseCirct_(false)
  {}

  filepath InputFile_;
  filepath OutputFolder_;
  OutputFormat OutputFormat_;
  std::string HlsFunction_;
  bool ExtractHlsFunction_;
  bool UseCirct_;
};

void
parse_cmdline(int argc, char ** argv, JlmHlsCommandLineOptions & options);

}

#endif
