/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ASSIGNMENT_H
#define JLM_IR_ASSIGNMENT_H

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {

JIVE_EXPORTED_INLINE const jlm::variable *
assignment_tac(jlm::basic_block * basic_block, const jlm::variable * lhs,
	const jlm::variable * rhs)
{
	jlm::assignment_op op(lhs->type());
	const jlm::tac * tac = basic_block->append(op, {rhs}, {lhs});
	return tac->output(0);
}

}

#endif
