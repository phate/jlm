/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_SELECT_HPP
#define JLM_IR_SELECt_HPP

#include <jlm/IR/operators.hpp>
#include <jlm/IR/tac.hpp>

namespace jlm {

static inline const jlm::variable *
select_tac(
	jlm::basic_block * bb,
	const jlm::variable * condition,
	const jlm::variable * true_value,
	const jlm::variable * false_value,
	const jlm::variable * result)
{
	jlm::select_op op(true_value->type());
	const jlm::tac * tac = bb->append(op, {condition, true_value, false_value}, {result});
	return tac->output(0);
}

}

#endif
