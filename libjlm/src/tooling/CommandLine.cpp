/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandLine.hpp>

#include <unordered_map>

namespace jlm
{

CommandLineOptions::~CommandLineOptions()
= default;

std::string
to_str(const optlvl & ol)
{
  static std::unordered_map<optlvl, const char*>
    map({
      {optlvl::O0, "O0"},
      {optlvl::O1, "O1"},
      {optlvl::O2, "O2"},
      {optlvl::O3, "O3"}
    });

  JLM_ASSERT(map.find(ol) != map.end());
  return map[ol];
}

std::string
to_str(const standard & std)
{
  static std::unordered_map<standard, const char*>
    map({
      {standard::none, ""},
      {standard::gnu89, "gnu89"},
      {standard::gnu99, "gnu99"},
      {standard::c89, "c89"},
      {standard::c99, "c99"},
      {standard::c11, "c11"},
      {standard::cpp98, "c++98"},
      {standard::cpp03, "c++03"},
      {standard::cpp11, "c++11"},
      {standard::cpp14, "c++14"}
    });

  JLM_ASSERT(map.find(std) != map.end());
  return map[std];
}

}