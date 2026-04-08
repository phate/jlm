/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/TypeConverter.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/LLVMContext.h>

TEST(AttributeConversionTests, TestAttributeKindConversion)
{
  typedef llvm::Attribute::AttrKind ak;

  for (int attributeKind = ak::None; attributeKind != ak::EndAttrKinds; attributeKind++)
  {
    jlm::llvm::ConvertAttributeKind(static_cast<ak>(attributeKind));
  }
}

TEST(AttributeConversionTests, TestAttributeListConversion)
{
  // Arrange
  llvm::LLVMContext context;
  llvm::AttrBuilder functionAttributeBuilder(context);
  functionAttributeBuilder.addAttribute(llvm::Attribute::NoUnwind);

  llvm::AttrBuilder returnAttributeBuilder(context);
  returnAttributeBuilder.addAttribute(llvm::Attribute::NoReturn);

  llvm::AttrBuilder parameter0AttributeBuilder(context);
  parameter0AttributeBuilder.addAttribute(llvm::Attribute::ZExt);

  llvm::AttrBuilder parameter1AttributeBuilder(context);
  parameter1AttributeBuilder.addAttribute(llvm::Attribute::NoUndef);

  auto llvmFunctionAttributes = llvm::AttributeSet::get(context, functionAttributeBuilder);
  auto llvmReturnAttributes = llvm::AttributeSet::get(context, returnAttributeBuilder);
  auto llvmParameter0Attributes = llvm::AttributeSet::get(context, parameter0AttributeBuilder);
  auto llvmParameter1Attributes = llvm::AttributeSet::get(context, parameter1AttributeBuilder);

  auto llvmAttributeList = llvm::AttributeList::get(
      context,
      llvmFunctionAttributes,
      llvmReturnAttributes,
      { llvmParameter0Attributes, llvmParameter1Attributes });

  // Act
  jlm::llvm::TypeConverter typeConverter;
  auto jlmAttributeList = jlm::llvm::convertAttributeList(llvmAttributeList, 2, typeConverter);

  // Assert
  auto jlmFunctionAttributes = jlmAttributeList.getFunctionAttributes();
  EXPECT_EQ(jlmFunctionAttributes.numAttributes(), 1u);
  EXPECT_EQ(
      *jlmFunctionAttributes.EnumAttributes().begin(),
      jlm::llvm::EnumAttribute(jlm::llvm::Attribute::kind::NoUnwind));

  auto jlmReturnAttributes = jlmAttributeList.getReturnAttributes();
  EXPECT_EQ(jlmReturnAttributes.numAttributes(), 1u);
  EXPECT_EQ(
      *jlmReturnAttributes.EnumAttributes().begin(),
      jlm::llvm::EnumAttribute(jlm::llvm::Attribute::kind::NoReturn));

  EXPECT_EQ(jlmAttributeList.getParameterAttributes().size(), 2u);
  auto jlmParameter0Attributes = jlmAttributeList.getParameterAttributes()[0];
  auto jlmParameter1Attributes = jlmAttributeList.getParameterAttributes()[1];

  EXPECT_EQ(jlmParameter0Attributes.numAttributes(), 1u);
  EXPECT_EQ(
      *jlmParameter0Attributes.EnumAttributes().begin(),
      jlm::llvm::EnumAttribute(jlm::llvm::Attribute::kind::ZExt));

  EXPECT_EQ(jlmParameter1Attributes.numAttributes(), 1u);
  EXPECT_EQ(
      *jlmParameter1Attributes.EnumAttributes().begin(),
      jlm::llvm::EnumAttribute(jlm::llvm::Attribute::kind::NoUndef));
}
