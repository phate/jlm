/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/arch/addresstype.h>
#include <jive/arch/memorytype.h>
#include <jive/arch/store.h>
#include <jive/frontend/basic_block.h>
#include <jive/frontend/clg.h>
#include <jive/frontend/tac/tac.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

static int
verify(jive::frontend::clg & clg)
{
	jive::frontend::clg_node * node = clg.lookup_function("test_store");
	assert(node != nullptr);

	jive::frontend::cfg & cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg.nnodes() == 3);
	assert(cfg.is_linear());

	jive::frontend::basic_block * bb = dynamic_cast<jive::frontend::basic_block*>(
		cfg.enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	std::vector<const jive::frontend::tac*> tacs = bb->tacs();
	assert(tacs.size() >= 2);

	jive::addr::type addrtype;
	jive::bits::type datatype(32);
	std::vector<std::unique_ptr<jive::state::type>> state_type;
	state_type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::store_op op(addrtype, state_type, datatype);
	assert(tacs[0]->operation() == op);
	assert(tacs[1]->inputs()[2]->origin() == tacs[0]->outputs()[0]);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-store", verify)
