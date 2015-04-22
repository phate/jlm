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

basic_block::~basic_block()
{
	for (auto tac : tacs_)
		delete tac;
}

basic_block::basic_block(jlm::cfg & cfg) noexcept
	: cfg_node(cfg)
{}

std::string
basic_block::debug_string() const
{
	std::stringstream sstrm;

	sstrm << this << "\\n";
	for (auto tac : tacs_)
		sstrm << tac->debug_string() << "\\n";

	return sstrm.str();
}

const tac *
basic_block::append(
	const jive::operation & operation,
	const std::vector<const variable*> & operands)
{
	std::vector<const variable*> results;
	for (size_t n = 0; n < operation.nresults(); n++)
		results.push_back(cfg()->create_variable(operation.result_type(n)));

	jlm::tac * tac = new jlm::tac(this, operation, operands, results);
	tacs_.push_back(tac);
	return tac;
}

const tac *
basic_block::append(
	const jive::operation & operation,
	const std::vector<const variable*> & operands,
	const std::vector<const variable*> & results)
{
	jlm::tac * tac = new jlm::tac(this, operation, operands, results);
	tacs_.push_back(tac);
	return tac;
}

const variable *
basic_block::append(const expr & e)
{
	return append(e, cfg()->create_variable(e.type()));
}

const variable *
basic_block::append(
	const expr & e,
	const variable * result)
{
	std::vector<const variable *> operands;
	for (size_t n = 0; n < e.noperands(); n++) {
		const variable * opv = cfg()->create_variable(e.type());
		operands.push_back(append(e.operand(n), opv));
	}

	jlm::tac * tac = new jlm::tac(this, e.operation(), operands, {result});
	tacs_.push_back(tac);
	return tac->output(0);
}

}
