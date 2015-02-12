/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_ASSIGNMENT_H
#define JLM_IR_TAC_ASSIGNMENT_H

#include <jlm/IR/tac/operators.hpp>
#include <jlm/IR/tac/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::output *
assignment_tac(jlm::frontend::basic_block * basic_block, const jlm::frontend::variable * v,
	const jlm::frontend::output * rhs)
{
	jlm::frontend::assignment_op op(v->type());
	const jlm::frontend::tac * tac = basic_block->append(op, {rhs}, {v});
	return tac->outputs()[0];
}

}
}

#endif
