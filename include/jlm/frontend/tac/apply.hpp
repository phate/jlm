/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_TAC_APPLY_H
#define JLM_FRONTEND_TAC_APPLY_H

#include <jlm/frontend/tac/operators.hpp>
#include <jlm/frontend/tac/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE std::vector<const jlm::frontend::output*>
apply_tac(jlm::frontend::basic_block * basic_block, const std::string & name,
	const jive::fct::type & function_type,
	const std::vector<const jlm::frontend::output*> & operands)
{
	jlm::frontend::apply_op op(name, function_type);
	return basic_block->append(op, operands)->outputs();
}

}
}

#endif
