/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_APPLY_H
#define JLM_IR_APPLY_H

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE std::vector<const jlm::frontend::variable*>
apply_tac(jlm::frontend::basic_block * basic_block, const clg_node * function,
	const std::vector<const jlm::frontend::variable*> & operands,
	const std::vector<const jlm::frontend::variable*> & results)
{
	jlm::frontend::apply_op op(function);
	basic_block->append(op, operands, results);
	return results;
}

}
}

#endif
