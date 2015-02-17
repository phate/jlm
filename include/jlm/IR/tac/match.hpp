/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_MATCH_HPP
#define JLM_IR_TAC_MATCH_HPP

#include <jive/vsdg/operators/match.h>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::variable *
match_tac(jlm::frontend::basic_block * basic_block, const jlm::frontend::variable * operand,
	const std::vector<size_t> & constants)
{
	if (!dynamic_cast<const jive::bits::type*>(&operand->type()))
		throw jive::type_error("bits<N>", operand->type().debug_string());

	jive::match_op op(static_cast<const jive::bits::type&>(operand->type()), constants);
	const variable * result = basic_block->cfg()->create_variable(op.result_type(0));
	const jlm::frontend::tac * tac = basic_block->append(op, {operand}, {result});

	JLM_DEBUG_ASSERT(op.narguments() == 1);
	return tac->output(0);
}

}
}

#endif
