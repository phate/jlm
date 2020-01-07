/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_DNE_HPP
#define JLM_OPT_DNE_HPP

namespace jlm {

class rvsdg_module;
class stats_descriptor;

void
dne(rvsdg_module & rm, const stats_descriptor & sd);

}

#endif
