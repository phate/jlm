/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/frontend/basic_block.hpp>
#include <jlm/frontend/clg.hpp>
#include <jlm/frontend/tac/apply.hpp>
#include <jlm/frontend/tac/tac.hpp>

#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>

#include <assert.h>

static int
verify(jlm::frontend::clg & clg)
{
//	jive_clg_view(clg);

	jlm::frontend::clg_node * caller = clg.lookup_function("caller");
	assert(caller != nullptr);

	assert(caller->calls().size() == 2);

	jlm::frontend::cfg * cfg = caller->cfg();
//	jive_cfg_view(cfg);

	assert(cfg->nnodes() == 3);
	assert(cfg->is_linear());

	jlm::frontend::basic_block * bb = dynamic_cast<jlm::frontend::basic_block*>(
		cfg->enter()->outedges()[0]->sink());
	assert(bb != nullptr);

	const std::list<const jlm::frontend::tac*> & tacs = bb->tacs();
	assert(tacs.size() != 0);

	std::vector<std::unique_ptr<jive::base::type>> argument_types;
	argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::bits::type(32)));
	argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));
	std::vector<std::unique_ptr<jive::base::type>> result_types;
	result_types.push_back(std::unique_ptr<jive::base::type>(new jive::bits::type(32)));
	result_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));
	jive::fct::type fcttype(argument_types, result_types);
	jlm::frontend::apply_op op("callee1", fcttype);
	assert((*std::next(std::next(tacs.begin())))->operation() == op);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/instructions/test-call", verify)
