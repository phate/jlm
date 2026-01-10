/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>

#include <llvm/IR/Attributes.h>

TEST(AttributeConversionTests, TestAttributeKindConversion)
{
  typedef llvm::Attribute::AttrKind ak;

  for (int attributeKind = ak::None; attributeKind != ak::EndAttrKinds; attributeKind++)
  {
    jlm::llvm::ConvertAttributeKind(static_cast<ak>(attributeKind));
  }
}
