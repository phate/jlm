/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_PUSH_HPP
#define JLM_OPT_PUSH_HPP

namespace jive {

class gamma_node;
class theta_node;

}

namespace jlm {

class rvsdg_module;

void
push_top(jive::theta_node * theta);

void
push_bottom(jive::theta_node * theta);

void
push(jive::theta_node * theta);

void
push(jive::gamma_node * gamma);

void
push(rvsdg_module & rm);

}

#endif
