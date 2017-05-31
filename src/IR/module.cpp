/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/module.hpp>

#include <jive/arch/addresstype.h>

namespace jlm {

global_variable *
module::add_global_variable(const std::string & name, const expr & e, bool exported)
{
	std::unique_ptr<jlm::variable> variable(new jlm::global_variable(
		jive::addr::type::instance(), name, exported));
	jlm::global_variable * v = static_cast<global_variable*>(variable.get());
	variables_.insert(std::move(variable));
	globals_[v] = std::move(std::unique_ptr<const expr>(new expr(e)));
	variable.release();
	return v;
}

}
