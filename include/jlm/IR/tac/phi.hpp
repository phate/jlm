/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_PHI_H
#define JLM_IR_TAC_PHI_H

#include <jlm/IR/tac/operators.hpp>
#include <jlm/IR/tac/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::output *
phi_tac(jlm::frontend::basic_block * basic_block,
	const std::vector<const jlm::frontend::variable*> & ops)
{
	JLM_DEBUG_ASSERT(!ops.empty());

	jlm::frontend::phi_op op(ops.size(), ops[0]->type());
	const variable * result = basic_block->cfg()->create_variable(op.result_type(0));
	const jlm::frontend::tac * tac = basic_block->append(op, ops, {result});
	return tac->outputs()[0];
}

}
}

#endif

