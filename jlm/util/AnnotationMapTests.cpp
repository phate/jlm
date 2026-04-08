/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/AnnotationMap.hpp>

TEST(AnnotationMapTests, AnnotationKeyValueRetrieval)
{
  using namespace jlm::util;

  // Arrange
  Annotation stringAnnotation("string", "value");
  Annotation intAnnotation("int", (int64_t)-1);
  Annotation uintAnnotation("uint", (uint64_t)1);
  Annotation doubleAnnotation("double", 1.0);

  // Act & Assert
  EXPECT_EQ(stringAnnotation.Label(), "string");
  EXPECT_EQ(stringAnnotation.Value<std::string>(), "value");
  EXPECT_TRUE(stringAnnotation.HasValueType<std::string>());
  EXPECT_FALSE(stringAnnotation.HasValueType<uint64_t>());

  EXPECT_EQ(intAnnotation.Label(), "int");
  EXPECT_EQ(intAnnotation.Value<int64_t>(), -1);
  EXPECT_TRUE(intAnnotation.HasValueType<int64_t>());
  EXPECT_FALSE(intAnnotation.HasValueType<uint64_t>());

  EXPECT_EQ(uintAnnotation.Label(), "uint");
  EXPECT_EQ(uintAnnotation.Value<uint64_t>(), 1u);
  EXPECT_TRUE(uintAnnotation.HasValueType<uint64_t>());

  EXPECT_EQ(doubleAnnotation.Label(), "double");
  EXPECT_EQ(doubleAnnotation.Value<double>(), 1.0);
  EXPECT_FALSE(doubleAnnotation.HasValueType<uint64_t>());

  EXPECT_THROW((void)doubleAnnotation.Value<uint64_t>(), std::bad_variant_access);
}

TEST(AnnotationMapTests, AnnotationEquality)
{
  using namespace jlm::util;

  // Arrange
  Annotation stringAnnotation("string", "value");
  Annotation intAnnotation("int", (int64_t)-1);
  Annotation uintAnnotation("uint", (uint64_t)1);
  Annotation doubleAnnotation("double", 1.0);

  // Act & Assert
  EXPECT_NE(stringAnnotation, doubleAnnotation);
  EXPECT_NE(stringAnnotation, intAnnotation);
  EXPECT_NE(stringAnnotation, uintAnnotation);

  Annotation otherStringAnnotation("string", "value");
  EXPECT_EQ(stringAnnotation, otherStringAnnotation);

  Annotation otherIntAnnotation("uint", (int64_t)1);
  EXPECT_NE(uintAnnotation, otherIntAnnotation);
}

TEST(AnnotationMapTests, AnnotationMap)
{
  using namespace jlm::util;

  // Arrange
  constexpr size_t key1 = 0;
  constexpr size_t key2 = 1;
  Annotation annotation("foo", "bar");

  jlm::util::AnnotationMap map;
  map.AddAnnotation(&key1, annotation);

  // Act & Assert
  EXPECT_TRUE(map.HasAnnotations(&key1));
  EXPECT_FALSE(map.HasAnnotations(&key2));

  auto annotations = map.GetAnnotations(&key1);
  EXPECT_EQ(annotations.size(), 1u);
  EXPECT_EQ(annotations[0], annotation);

  for (auto & iteratedAnnotations : map.Annotations())
  {
    for (auto & iteratedAnnotation : iteratedAnnotations)
    {
      EXPECT_EQ(iteratedAnnotation, annotation);
    }
  }
}
