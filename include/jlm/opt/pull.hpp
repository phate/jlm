/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_PULL_HPP
#define JLM_OPT_PULL_HPP

namespace jive {
	class gamma;
}

namespace jlm {

void
pullin_top(jive::gamma & gamma);

void
pullin_bottom(jive::gamma & gamma);

}

#endif
