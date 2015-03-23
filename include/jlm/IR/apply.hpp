/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_APPLY_H
#define JLM_IR_APPLY_H

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {

JIVE_EXPORTED_INLINE std::vector<const jlm::variable*>
apply_tac(jlm::basic_block * basic_block, const clg_node * function,
	const std::vector<const jlm::variable*> & operands,
	const std::vector<const jlm::variable*> & results)
{
	jlm::apply_op op(function);
	basic_block->append(op, operands, results);
	return results;
}

}

#endif
