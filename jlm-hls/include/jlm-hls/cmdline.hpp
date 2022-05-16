/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMHLS_CMDLINE_HPP
#define JLM_JLMHLS_CMDLINE_HPP

#include <jlm/util/file.hpp>

namespace jlm {

enum class OutputFormat {firrtl, dot};

class cmdline_options {
public:
	cmdline_options()
	: inputFile("")
	, outputFolder("")
	, format(OutputFormat::firrtl)
	{}

	jlm::filepath inputFile;
	jlm::filepath outputFolder;
	OutputFormat format;
	std::string hlsFunction;
	bool extractHlsFunction = false;
	bool useCirct = false;
};

void
parse_cmdline(int argc, char ** argv, cmdline_options & options);

}

#endif
