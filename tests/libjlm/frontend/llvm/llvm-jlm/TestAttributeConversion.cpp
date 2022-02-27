/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/frontend/llvm/llvm2jlm/LlvmModuleConversion.hpp>

#include <llvm/IR/Attributes.h>

static void
TestAttributeKindConversion()
{
  typedef llvm::Attribute::AttrKind ak;

  for (int attributeKind = ak::None; attributeKind != ak::EndAttrKinds; attributeKind++) {
    jlm::convert_attribute_kind(static_cast<ak>(attributeKind));
  }
}

static int
test()
{
  TestAttributeKindConversion();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/frontend/llvm/llvm-jlm/TestAttributeConversion", test)