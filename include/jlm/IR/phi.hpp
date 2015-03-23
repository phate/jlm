/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_PHI_H
#define JLM_IR_PHI_H

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::variable *
phi_tac(
	jlm::frontend::basic_block * basic_block,
	const std::vector<const jlm::frontend::variable*> & ops,
	const jlm::frontend::variable * result)
{
	JLM_DEBUG_ASSERT(!ops.empty());

	jlm::frontend::phi_op op(ops.size(), ops[0]->type());
	const jlm::frontend::tac * tac = basic_block->append(op, ops, {result});
	return tac->output(0);
}

}
}

#endif

