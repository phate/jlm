/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/expression.hpp>
#include <jlm/IR/tac.hpp>
#include <jlm/IR/variable.hpp>

#include <sstream>

namespace jlm {

/* basic block attribute */

basic_block::~basic_block()
{
	for (const auto & tac : tacs_)
		delete tac;
}

std::string
basic_block::debug_string() const noexcept
{
	std::stringstream sstrm;

	sstrm << this << "\\n";
	for (auto tac : tacs_)
		sstrm << tac->debug_string() << "\\n";

	return sstrm.str();
}

std::unique_ptr<attribute>
basic_block::copy() const
{
	return std::make_unique<basic_block>(*this);
}

}
