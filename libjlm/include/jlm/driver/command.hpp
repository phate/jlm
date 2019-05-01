/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_DRIVER_COMMAND_HPP
#define JLM_DRIVER_COMMAND_HPP

#include <string>

namespace jlm {

/* command class */

class command {
public:
	virtual
	~command();

	virtual std::string
	to_str() const = 0;

	virtual void
	run() const = 0;
};

}

#endif
