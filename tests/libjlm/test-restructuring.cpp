/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>

#include <assert.h>

/*

unsigned int
test(
  unsigned int p1,
  unsigned int p2,
  unsigned int p3)
{
  unsigned int r = 0;
  for (unsigned int i = 0; i < p1; i++) {
    for (unsigned int f = 0; f < p2; f++) {
      r++;
    }
  }

  for (unsigned int k = 0; k < p3; k++)
    r++;

  return r;
}

*/

static int
verify(const jive::graph * graph)
{
	/* FIXME: insert checks */

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-restructuring", nullptr, verify);
