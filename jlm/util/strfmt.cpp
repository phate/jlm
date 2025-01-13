/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/strfmt.hpp>

#include <random>

namespace jlm::util {

std::string
CreateRandomAlphanumericString(std::size_t length)
{
  const std::string characterSet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<> distribution(0, characterSet.size() - 1);

  std::string result;
  for (std::size_t i = 0; i < length; ++i)
  {
    result += characterSet[distribution(generator)];
  }

  return result;
}

}