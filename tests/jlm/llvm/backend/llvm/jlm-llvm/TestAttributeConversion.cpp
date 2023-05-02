/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>

#include <llvm/IR/Attributes.h>

static void
TestAttributeKindConversion()
{
  typedef jlm::attribute::kind ak;

  int begin = static_cast<int>(ak::None);
  int end = static_cast<int>(ak::EndAttrKinds);
  for (int attributeKind = begin; attributeKind != end; attributeKind++) {
    jlm::jlm2llvm::convert_attribute_kind(static_cast<ak>(attributeKind));
  }
}

static int
test()
{
  TestAttributeKindConversion();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/TestAttributeConversion", test)
