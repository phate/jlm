/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/arch/memorytype.h>
#include <jive/frontend/basic_block.h>
#include <jive/frontend/clg.h>
#include <jive/frontend/tac/apply.h>
#include <jive/frontend/tac/tac.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>

#include <assert.h>

static int
verify(jive::frontend::clg & clg)
{
//	jive_clg_view(clg);

	jive::frontend::clg_node * caller = clg.lookup_function("caller");
	assert(caller != nullptr);

	assert(caller->calls.size() == 2);

	jive::frontend::cfg & cfg = caller->cfg();
//	jive_cfg_view(cfg);

	assert(cfg.nnodes() == 3);
	assert(cfg.is_linear());

	jive::frontend::basic_block * bb = dynamic_cast<jive::frontend::basic_block*>(
		cfg.enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	std::vector<const jive::frontend::tac*> tacs = bb->tacs();
	assert(tacs.size() != 0);

	std::vector<std::unique_ptr<jive::base::type>> argument_types;
	argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));
	argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::bits::type(32)));
	std::vector<std::unique_ptr<jive::base::type>> result_types;
	result_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));
	result_types.push_back(std::unique_ptr<jive::base::type>(new jive::bits::type(32)));
	jive::fct::type fcttype(argument_types, result_types);
	jive::frontend::apply_op op("callee1", fcttype);
	assert(tacs[0]->operation() == op);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-call", verify)
