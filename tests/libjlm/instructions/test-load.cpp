/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/frontend/basic_block.hpp>
#include <jlm/frontend/clg.hpp>
#include <jlm/frontend/tac/tac.hpp>

#include <jive/arch/addresstype.h>
#include <jive/arch/load.h>
#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

static int
verify(jlm::frontend::clg & clg)
{
	jlm::frontend::clg_node * node = clg.lookup_function("test_load");
	assert(node != nullptr);

	jlm::frontend::cfg * cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg->nnodes() == 3);
	assert(cfg->is_linear());

	jlm::frontend::basic_block * bb = dynamic_cast<jlm::frontend::basic_block*>(
		cfg->enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	std::vector<const jlm::frontend::tac*> tacs = bb->tacs();
	assert(tacs.size() != 0);

	jive::addr::type addrtype;
	jive::bits::type datatype(32);
	std::vector<std::unique_ptr<jive::state::type>> state_type;
	state_type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::load_op op(addrtype, state_type, datatype);
	assert(tacs[1]->operation() == op);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-load", verify)
