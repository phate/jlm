/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_TAC_PHI_H
#define JLM_FRONTEND_TAC_PHI_H

#include <jlm/frontend/tac/operators.hpp>
#include <jlm/frontend/tac/tac.hpp>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::output *
phi_tac(jlm::frontend::basic_block * basic_block,
	const std::vector<const jlm::frontend::output*> & ops)
{
	JIVE_DEBUG_ASSERT(!ops.empty());

	jlm::frontend::phi_op op(ops.size(), ops[0]->type());
	const jlm::frontend::tac * tac = basic_block->append(op, ops);
	return tac->outputs()[0];
}

}
}

#endif

