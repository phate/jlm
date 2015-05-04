/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/module.hpp>

#include <jive/arch/addresstype.h>

namespace jlm {

variable *
module::add_global_variable(const std::string & name, const expr & e)
{
	std::unique_ptr<variable> variable(new jlm::variable(jive::addr::type::instance(), name));
	jlm::variable * v = variable.get();
	variables_.insert(std::move(variable));
	globals_[v] = std::move(std::unique_ptr<const expr>(new expr(e)));
	variable.release();
	return v;
}

}
