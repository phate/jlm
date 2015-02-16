/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/tac/tac.hpp>

#include <jive/vsdg/basetype.h>

#include <sstream>

namespace jlm {
namespace frontend {

input::input(const jlm::frontend::tac * tac, size_t index, const output * origin)
	: tac_(tac)
	, index_(index)
	, origin_(origin)
{
	//FIXME: use jive_raise_type_error once it does not need jive_context any more
	if (origin->type() != type()) {
		std::string msg("Type mismatch: required '");
		msg.append(type().debug_string().c_str());
		msg.append("' got '");
		msg.append(origin->type().debug_string().c_str());
		msg.append("'");
		throw std::logic_error(msg);
	}
}

output::output(const jlm::frontend::tac * tac, size_t index,
	const jlm::frontend::variable * variable)
	: tac_(tac)
	, index_(index)
	, variable_(variable)
{
	/*
		FIXME: throw type error
	*/
	if (variable_->type() != type())
		throw std::logic_error("Invalid variable!");
}

tac::~tac() noexcept
{
	for (auto output : outputs_)
		delete output;
}

tac::tac(const cfg_node * owner,
	const jive::operation & operation,
	const std::vector<const output*> & operands)
	: owner_(owner)
	, operation_(std::move(operation.copy()))
{
	if (operands.size() != operation.narguments())
		throw std::logic_error("Invalid number of operands.");

	for (size_t n = 0; n < operation.narguments(); n++)
		inputs_.push_back(operands[n]->variable());

	for (size_t n = 0; n < operation.nresults(); n++)
		outputs_.push_back(new output(this, n,
			owner->cfg()->create_variable(operation.result_type(n))));
}

tac::tac(const cfg_node * owner,
	const jive::operation & operation,
	const std::vector<const variable*> & operands,
	const std::vector<const variable*> & results)
	: owner_(owner)
	, operation_(std::move(operation.copy()))
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

	for (size_t n = 0; n < operation.nresults(); n++)
		outputs_.push_back(new output(this, n, results[n]));
}

std::string
tac::debug_string() const
{
	std::stringstream sstrm;

	JLM_DEBUG_ASSERT(outputs_.size() != 0);
	for (size_t n = 0; n < outputs_.size()-1; n++)
		sstrm << outputs_[n]->variable()->debug_string() << ", ";
	sstrm << outputs_[outputs_.size()-1]->variable()->debug_string() << " = ";

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
}
