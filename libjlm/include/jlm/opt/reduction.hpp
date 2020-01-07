/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_REDUCTION_HPP
#define JLM_OPT_REDUCTION_HPP

namespace jlm {

class rvsdg;
class stats_descriptor;

void
reduce(rvsdg_module & rm, const stats_descriptor & sd);

}

#endif
