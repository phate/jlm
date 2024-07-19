/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/attribute.hpp>

#include <cassert>

static int
TestEquality()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  enum_attribute enumAttribute1(attribute::kind::AllocAlign);
  enum_attribute enumAttribute2(attribute::kind::AlwaysInline);

  int_attribute intAttribute1(attribute::kind::Alignment, 4);
  int_attribute intAttribute2(attribute::kind::AllocSize, 8);

  string_attribute stringAttribute1("myKind1", "myValue");
  string_attribute stringAttribute2("myKind2", "myValue");

  type_attribute typeAttribute1(attribute::kind::ByRef, valueType);
  type_attribute typeAttribute2(attribute::kind::ByVal, valueType);

  attributeset set1;
  set1.InsertEnumAttribute(enumAttribute1);
  set1.InsertIntAttribute(intAttribute1);
  set1.InsertStringAttribute(stringAttribute1);
  set1.InsertTypeAttribute(typeAttribute1);

  attributeset set2;
  set2.InsertEnumAttribute(enumAttribute2);
  set2.InsertIntAttribute(intAttribute2);
  set2.InsertStringAttribute(stringAttribute2);
  set2.InsertTypeAttribute(typeAttribute2);

  attributeset set3;
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

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/AttributeSetTests-TestEquality", TestEquality);
