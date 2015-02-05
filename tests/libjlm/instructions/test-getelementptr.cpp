/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/frontend/basic_block.hpp>
#include <jlm/frontend/clg.hpp>
#include <jlm/frontend/tac/tac.hpp>

#include <jive/arch/address.h>
#include <jive/types/bitstring/constant.h>

#include <assert.h>

static int
verify(jlm::frontend::clg & clg)
{
	jlm::frontend::clg_node * node = clg.lookup_function("test_getelementptr");
	assert(node != nullptr);

	jlm::frontend::cfg * cfg = node->cfg();
//	jive_cfg_view(*cfg);

	assert(cfg->nnodes() == 3);
	assert(cfg->is_linear());

	jlm::frontend::basic_block * bb = dynamic_cast<jlm::frontend::basic_block*>(
		cfg->enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	const std::list<const jlm::frontend::tac*> & tacs = bb->tacs();
	assert(tacs.size() == 14);

/*
	FIXME: insert checks again
*/

/*
	for (size_t n = 1; n < 5; n += 2) {
		const jlm::frontend::tac * constant = tacs[n];
		const jlm::frontend::tac * subscrpt = tacs[n+1];
		assert(dynamic_cast<const jive::bits::constant_op*>(&constant->operation()));
		assert(dynamic_cast<const jive::address::arraysubscript_op*>(&subscrpt->operation()));
	}
*/
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-getelementptr", verify);
