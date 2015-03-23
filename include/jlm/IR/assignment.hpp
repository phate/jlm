/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ASSIGNMENT_H
#define JLM_IR_ASSIGNMENT_H

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::variable *
assignment_tac(jlm::frontend::basic_block * basic_block, const jlm::frontend::variable * lhs,
	const jlm::frontend::variable * rhs)
{
	jlm::frontend::assignment_op op(lhs->type());
	const jlm::frontend::tac * tac = basic_block->append(op, {rhs}, {lhs});
	return tac->output(0);
}

}
}

#endif
