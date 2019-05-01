/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/tac.hpp>

#include <jive/rvsdg/type.h>

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

tac::tac(
	const jive::simple_op & operation,
	const std::vector<const variable *> & operands,
	const std::vector<const variable *> & results)
	: operation_(std::move(operation.copy()))
{
	if (operands.size() != operation.narguments())
		throw jlm::error("invalid number of operands.");

	if (results.size() != operation.nresults())
		throw jlm::error("invalid number of variables.");

	for (size_t n = 0; n < operands.size(); n++) {
		if (operands[n]->type() != operation.argument(n).type())
			throw jlm::error("invalid type.");
		inputs_.push_back(operands[n]);
	}

	for (size_t n = 0; n < results.size(); n++) {
		if (results[n]->type() != operation.result(n).type())
			throw jlm::error("invalid type.");
		outputs_.push_back(results[n]);
	}
}

}
