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

basic_block_attribute::~basic_block_attribute()
{
	for (const auto & tac : tacs_)
		delete tac;
}

std::string
basic_block_attribute::debug_string() const noexcept
{
	std::stringstream sstrm;

	sstrm << this << "\\n";
	for (auto tac : tacs_)
		sstrm << tac->debug_string() << "\\n";

	return sstrm.str();
}

std::unique_ptr<attribute>
basic_block_attribute::copy() const
{
	return std::make_unique<basic_block_attribute>(*this);
}

const tac *
basic_block_attribute::append(
	const jive::operation & operation,
	const std::vector<const variable*> & operands,
	const std::vector<const variable*> & results)
{
	auto tac = new jlm::tac(operation, operands, results);
	tacs_.push_back(tac);
	return tac;
}

const tac *
basic_block_attribute::append(
	jlm::cfg * cfg,
	const jive::operation & operation,
	const std::vector<const variable*> & operands)
{
	std::vector<const variable*> results;
	for (size_t n = 0; n < operation.nresults(); n++)
		results.push_back(cfg->create_variable(operation.result_type(n)));

	return append(operation, operands, results);
}

const variable *
basic_block_attribute::append(
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
basic_block_attribute::append(jlm::cfg * cfg, const expr & e)
{
	return append(cfg, e, cfg->create_variable(e.type()));
}

/* basic block */

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

	jlm::tac * tac = new jlm::tac(operation, operands, results);
	tacs_.push_back(tac);
	return tac;
}

const tac *
basic_block::append(
	const jive::operation & operation,
	const std::vector<const variable*> & operands,
	const std::vector<const variable*> & results)
{
	jlm::tac * tac = new jlm::tac(operation, operands, results);
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
		const variable * opv = cfg()->create_variable(e.operand(n).type());
		operands.push_back(append(e.operand(n), opv));
	}

	jlm::tac * tac = new jlm::tac(e.operation(), operands, {result});
	tacs_.push_back(tac);
	return tac->output(0);
}

}
