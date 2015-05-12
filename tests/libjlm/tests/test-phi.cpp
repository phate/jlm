/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <assert.h>

/*
	The phi instruction's operands are the result of an instruction that is not residing
	in the same basic block as the phi instruction.
*/

JLM_UNIT_TEST_REGISTER("libjlm/tests/test-phi", nullptr, nullptr);
