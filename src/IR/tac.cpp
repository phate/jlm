/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/tac.hpp>

#include <jive/vsdg/basetype.h>

#include <sstream>

namespace jlm {

tac::tac(
	const jive::operation & operation,
	const std::vector<const variable*> & operands,
	const std::vector<const variable*> & results)
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
		if (operands[n]->type() != operation.argument_type(n))
			throw std::logic_error("Invalid type.");
		inputs_.push_back(operands[n]);
	}

	for (size_t n = 0; n < results.size(); n++) {
		if (results[n]->type() != operation.result_type(n))
			throw std::logic_error("Invalid type.");
		outputs_.push_back(results[n]);
	}
}

std::string
tac::debug_string() const
{
	std::stringstream sstrm;

	JLM_DEBUG_ASSERT(outputs_.size() != 0);
	for (size_t n = 0; n < outputs_.size()-1; n++)
		sstrm << outputs_[n]->debug_string() << ", ";
	sstrm << outputs_[outputs_.size()-1]->debug_string() << " = ";

	sstrm << operation_->debug_string();

	if (inputs_.size() != 0) {
		sstrm << " ";
		for (size_t n = 0; n < inputs_.size()-1; n++)
			sstrm << inputs_[n]->debug_string() << ", ";
		sstrm << inputs_[inputs_.size()-1]->debug_string();
	}

	return sstrm.str();
}

}
