/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>

#include <llvm/IR/Attributes.h>

static void
TestAttributeKindConversion()
{
  typedef llvm::Attribute::AttrKind ak;

  for (int attributeKind = ak::None; attributeKind != ak::EndAttrKinds; attributeKind++) {
    jlm::ConvertAttributeKind(static_cast<ak>(attributeKind));
  }
}

static int
test()
{
  TestAttributeKindConversion();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/TestAttributeConversion", test)