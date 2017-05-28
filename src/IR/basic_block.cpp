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

const variable *
basic_block::append(
	jlm::cfg * cfg,
	const expr & e,
	const variable * result)
{
	std::vector<const variable *> operands;
	for (size_t n = 0; n < e.noperands(); n++) {
		auto opv = cfg->create_variable(e.operand(n).type());
		operands.push_back(append(cfg, e.operand(n), opv));
	}

	auto tac = new jlm::tac(e.operation(), operands, {result});
	tacs_.push_back(tac);
	return tac->output(0);
}

const variable *
basic_block::append(jlm::cfg * cfg, const expr & e)
{
	return append(cfg, e, cfg->create_variable(e.type()));
}

}
