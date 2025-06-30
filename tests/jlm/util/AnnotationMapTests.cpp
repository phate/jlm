/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/util/AnnotationMap.hpp>

static void
AnnotationKeyValueRetrieval()
{
  using namespace jlm::util;

  // Arrange
  Annotation stringAnnotation("string", "value");
  Annotation intAnnotation("int", (int64_t)-1);
  Annotation uintAnnotation("uint", (uint64_t)1);
  Annotation doubleAnnotation("double", 1.0);

  // Act & Assert
  assert(stringAnnotation.Label() == "string");
  assert(stringAnnotation.Value<std::string>() == "value");
  assert(stringAnnotation.HasValueType<std::string>());
  assert(!stringAnnotation.HasValueType<uint64_t>());

  assert(intAnnotation.Label() == "int");
  assert(intAnnotation.Value<int64_t>() == -1);
  assert(intAnnotation.HasValueType<int64_t>());
  assert(!intAnnotation.HasValueType<uint64_t>());

  assert(uintAnnotation.Label() == "uint");
  assert(uintAnnotation.Value<uint64_t>() == 1);
  assert(uintAnnotation.HasValueType<uint64_t>());

  assert(doubleAnnotation.Label() == "double");
  assert(doubleAnnotation.Value<double>() == 1.0);
  assert(!doubleAnnotation.HasValueType<uint64_t>());

  try
  {
    (void)doubleAnnotation.Value<uint64_t>();
    assert(false); // the line above should have thrown an exception
  }
  catch (...)
  {}
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/AnnotationMapTests-AnnotationKeyValueRetrieval",
    AnnotationKeyValueRetrieval)

static void
AnnotationEquality()
{
  using namespace jlm::util;

  // Arrange
  Annotation stringAnnotation("string", "value");
  Annotation intAnnotation("int", (int64_t)-1);
  Annotation uintAnnotation("uint", (uint64_t)1);
  Annotation doubleAnnotation("double", 1.0);

  // Act & Assert
  assert(stringAnnotation != doubleAnnotation);
  assert(stringAnnotation != intAnnotation);
  assert(stringAnnotation != uintAnnotation);

  Annotation otherStringAnnotation("string", "value");
  assert(stringAnnotation == otherStringAnnotation);

  Annotation otherIntAnnotation("uint", (int64_t)1);
  assert(uintAnnotation != otherIntAnnotation);
}

JLM_UNIT_TEST_REGISTER("jlm/util/AnnotationMapTests-AnnotationEquality", AnnotationEquality)

static void
AnnotationMap()
{
  using namespace jlm::util;

  // Arrange
  Annotation annotation("foo", "bar");

  jlm::util::AnnotationMap map;
  map.AddAnnotation((const void *)&AnnotationEquality, annotation);

  // Act & Assert
  assert(map.HasAnnotations((const void *)&AnnotationEquality));
  assert(!map.HasAnnotations((const void *)&AnnotationKeyValueRetrieval));

  auto annotations = map.GetAnnotations((const void *)&AnnotationEquality);
  assert(annotations.size() == 1);
  assert(annotations[0] == annotation);

  for (auto & iteratedAnnotations : map.Annotations())
  {
    for (auto & iteratedAnnotation : iteratedAnnotations)
    {
      assert(iteratedAnnotation == annotation);
    }
  }
}

JLM_UNIT_TEST_REGISTER("jlm/util/AnnotationMapTests-AnnotationMap", AnnotationMap)
