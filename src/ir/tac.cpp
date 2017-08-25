/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/tac.hpp>

#include <jive/vsdg/basetype.h>

#include <sstream>

namespace jlm {

/* tacvariable */

tacvariable::~tacvariable()
{}

/* tac */

tac::tac(
	const jive::operation & operation,
	const std::vector<const variable *> & operands,
	const std::vector<const variable *> & results)
	: operation_(std::move(operation.copy()))
{
	/*
		FIXME: throw proper exceptions
	*/
	if (operands.size() != operation.narguments())
		throw std::logic_error("Invalid number of operands.");

	if (results.size() != operation.nresults())
		throw std::logic_error("Invalid number of variables.");

	for (size_t n = 0; n < operands.size(); n++) {
		if (operands[n]->type() != operation.argument(n).type())
			throw std::logic_error("Invalid type.");
		inputs_.push_back(operands[n]);
	}

	for (size_t n = 0; n < results.size(); n++) {
		if (results[n]->type() != operation.result(n).type())
			throw std::logic_error("Invalid type.");
		outputs_.push_back(results[n]);
	}
}

}
