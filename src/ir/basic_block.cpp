/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/expression.hpp>
#include <jlm/ir/tac.hpp>
#include <jlm/ir/variable.hpp>

#include <sstream>

namespace jlm {

/* basic block attribute */

basic_block::~basic_block()
{
	for (const auto & tac : tacs_)
		delete tac;
}

std::unique_ptr<attribute>
basic_block::copy() const
{
	return std::make_unique<basic_block>(*this);
}

}
