/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/tac.hpp>

#include <jive/rvsdg/type.hpp>

#include <sstream>

namespace jlm {

/* tacvariable */

tacvariable::~tacvariable()
{}

/* taclist */

taclist::~taclist()
{
	for (const auto & tac : tacs_)
		delete tac;
}

/* tac */

static void
check_operands(
	const jive::simple_op & operation,
	const std::vector<const variable*> & operands)
{
	if (operands.size() != operation.narguments())
		throw jlm::error("invalid number of operands.");

	for (size_t n = 0; n < operands.size(); n++) {
		if (operands[n]->type() != operation.argument(n).type())
			throw jlm::error("invalid type.");
	}
}

static void
check_results(
	const jive::simple_op & operation,
	const std::vector<tacvariable*> & results)
{
	if (results.size() != operation.nresults())
		throw jlm::error("invalid number of variables.");

	for (size_t n = 0; n < results.size(); n++) {
		if (results[n]->type() != operation.result(n).type())
			throw jlm::error("invalid type.");
	}
}

tac::tac(
	const jive::simple_op & operation,
	const std::vector<const variable *> & operands,
	const std::vector<tacvariable *> & results)
	: results_(results)
	, operands_(operands)
	, operation_(operation.copy())
{
	check_operands(operation, operands);
	check_results(operation, results);

	for (auto & result : results)
		result->set_tac(this);
}

void
tac::replace(
	const jive::simple_op & operation,
	const std::vector<const variable*> & operands,
	const std::vector<tacvariable*> & results)
{
	check_operands(operation, operands);
	check_results(operation, results);

	results_ = results;
	operands_ = operands;
	operation_ = operation.copy();

	for (auto & result : results)
		result->set_tac(this);
}

void
tac::replace(
	const jive::simple_op & operation,
	const std::vector<const variable*> & operands)
{
	check_operands(operation, operands);
	check_results(operation, results_);

	operands_ = operands;
	operation_ = operation.copy();
}

}
