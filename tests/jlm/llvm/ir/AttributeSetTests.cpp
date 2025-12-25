/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/rvsdg/TestType.hpp>

#include <cassert>

static void
TestEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();

  EnumAttribute enumAttribute1(Attribute::kind::AllocAlign);
  EnumAttribute enumAttribute2(Attribute::kind::AlwaysInline);

  IntAttribute intAttribute1(Attribute::kind::Alignment, 4);
  IntAttribute intAttribute2(Attribute::kind::AllocSize, 8);

  StringAttribute stringAttribute1("myKind1", "myValue");
  StringAttribute stringAttribute2("myKind2", "myValue");

  TypeAttribute typeAttribute1(Attribute::kind::ByRef, valueType);
  TypeAttribute typeAttribute2(Attribute::kind::ByVal, valueType);

  AttributeSet set1;
  set1.InsertEnumAttribute(enumAttribute1);
  set1.InsertIntAttribute(intAttribute1);
  set1.InsertStringAttribute(stringAttribute1);
  set1.InsertTypeAttribute(typeAttribute1);

  AttributeSet set2;
  set2.InsertEnumAttribute(enumAttribute2);
  set2.InsertIntAttribute(intAttribute2);
  set2.InsertStringAttribute(stringAttribute2);
  set2.InsertTypeAttribute(typeAttribute2);

  AttributeSet set3;
  set3.InsertEnumAttribute(enumAttribute1);
  set3.InsertIntAttribute(intAttribute1);
  set3.InsertStringAttribute(stringAttribute1);
  set3.InsertTypeAttribute(typeAttribute1);

  // Act & Assert
  assert(set1 == set1);
  assert(set1 != set2);
  assert(set1 == set3);

  assert(set2 == set2);
  assert(set2 != set3);

  assert(set3 == set3);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/AttributeSetTests-TestEquality", TestEquality);
