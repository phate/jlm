/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_DESTRUCTION_RESTRUCTURING_HPP
#define JLM_DESTRUCTION_RESTRUCTURING_HPP

namespace jlm {

namespace frontend {
	class cfg;
}

void
restructure(jlm::frontend::cfg * cfg);

}

#endif
