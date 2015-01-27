/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/frontend/basic_block.hpp>
#include <jlm/frontend/clg.hpp>
#include <jlm/frontend/tac/operators.hpp>
#include <jlm/frontend/tac/tac.hpp>

#include <jive/types/bitstring/type.h>

#include <assert.h>

static int
verify(jlm::frontend::clg & clg)
{
	jlm::frontend::clg_node * node = clg.lookup_function("test_phi");
	assert(node != nullptr);

	jlm::frontend::cfg * cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg->nnodes() == 5);
	assert(cfg->is_reducible());

	jlm::frontend::basic_block * bb = dynamic_cast<jlm::frontend::basic_block*>(
		cfg->enter()->outedges()[0]->sink()->outedges()[0]->sink());
	assert(bb != nullptr);

	std::vector<const jlm::frontend::tac*> tacs = bb->tacs();
	assert(tacs.size() != 0);

	jive::bits::type bits32(32);
	jlm::frontend::phi_op op(2, bits32);
	assert(tacs[0]->operation() == op);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-phi", verify)
