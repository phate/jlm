/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_PHI_H
#define JLM_IR_PHI_H

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {

JIVE_EXPORTED_INLINE const jlm::variable *
phi_tac(
	jlm::basic_block * basic_block,
	const std::vector<const jlm::variable*> & ops,
	const jlm::variable * result)
{
	JLM_DEBUG_ASSERT(!ops.empty());

	jlm::phi_op op(ops.size(), ops[0]->type());
	const jlm::tac * tac = basic_block->append(op, ops, {result});
	return tac->output(0);
}

}

#endif
