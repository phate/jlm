/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/arch/address.h>
#include <jive/frontend/basic_block.h>
#include <jive/frontend/clg.h>
#include <jive/frontend/tac/tac.h>
#include <jive/types/bitstring/constant.h>

#include <assert.h>

static int
verify(jive::frontend::clg & clg)
{
	jive::frontend::clg_node * node = clg.lookup_function("test_getelementptr");
	assert(node != nullptr);

	jive::frontend::cfg & cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg.nnodes() == 3);
	assert(cfg.is_linear());

	jive::frontend::basic_block * bb = dynamic_cast<jive::frontend::basic_block*>(
		cfg.enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	std::vector<const jive::frontend::tac*> tacs = bb->tacs();
	assert(tacs.size() == 10);

	for (size_t n = 0; n < 5; n += 2) {
		const jive::frontend::tac * constant = tacs[n];
		const jive::frontend::tac * subscrpt = tacs[n+1];
		assert(dynamic_cast<const jive::bits::constant_op*>(&constant->operation()));
		assert(dynamic_cast<const jive::address::arraysubscript_operation*>(&subscrpt->operation()));
	}

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-getelementptr", verify);
