/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/frontend/basic_block.hpp>
#include <jlm/frontend/cfg.hpp>
#include <jlm/frontend/tac/tac.hpp>
#include <jlm/frontend/variable.hpp>

#include <sstream>

namespace jlm {
namespace frontend {

basic_block::~basic_block()
{
	for (auto tac : tacs_)
		delete tac;
}

basic_block::basic_block(jlm::frontend::cfg & cfg) noexcept
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
basic_block::append(const jive::operation & operation, const std::vector<const output*> & operands)
{
	jlm::frontend::tac * tac = new jlm::frontend::tac(this, operation, operands);
	tacs_.push_back(tac);
	return tac;
}

const tac *
basic_block::append(
	const jive::operation & operation,
	const std::vector<const output*> & operands,
	const std::vector<const jlm::frontend::variable*> & variables)
{
	jlm::frontend::tac * tac = new jlm::frontend::tac(this, operation, operands, variables);
	tacs_.push_back(tac);
	return tac;
}

}
}
