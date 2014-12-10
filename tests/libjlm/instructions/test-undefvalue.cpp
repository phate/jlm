/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/frontend/basic_block.h>
#include <jive/frontend/clg.h>
#include <jive/frontend/tac/tac.h>
#include <jive/types/bitstring/constant.h>

#include <assert.h>

static int
verify(jive::frontend::clg & clg)
{
	jive::frontend::clg_node * node = clg.lookup_function("test_undefvalue");
	assert(node != nullptr);

	jive::frontend::cfg & cfg = node->cfg();
	jive_cfg_view(cfg);

	assert(cfg.nnodes() == 3);
	assert(cfg.is_linear());

	jive::frontend::basic_block * bb = dynamic_cast<jive::frontend::basic_block*>(
		cfg.enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	std::vector<const jive::frontend::tac*> tacs = bb->tacs();
	assert(tacs.size() != 0);

	jive::bits::value_repr v(32, 'X');
	jive::bits::constant_op op(v);
	assert(tacs[0]->operation() == op);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-undefvalue", verify)
