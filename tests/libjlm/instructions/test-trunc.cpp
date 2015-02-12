/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/tac/tac.hpp>

#include <jive/types/bitstring/slice.h>
#include <jive/types/bitstring/type.h>

#include <assert.h>

static int
verify(jlm::frontend::clg & clg)
{
	jlm::frontend::clg_node * node = clg.lookup_function("test_trunc");
	assert(node != nullptr);

	jlm::frontend::cfg * cfg = node->cfg();
//	jive_cfg_view(cfg);

	assert(cfg->nnodes() == 3);
	assert(cfg->is_linear());

	jlm::frontend::basic_block * bb = dynamic_cast<jlm::frontend::basic_block*>(
		cfg->enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	const std::list<const jlm::frontend::tac*> & tacs = bb->tacs();
	assert(tacs.size() != 0);

	jive::bits::type bits64(64);
	jive::bits::slice_op op(bits64, 0, 32);
	assert((*(std::next(std::next(tacs.begin()))))->operation() == op);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-trunc", verify)
