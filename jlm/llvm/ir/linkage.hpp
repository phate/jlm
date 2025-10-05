/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_LINKAGE_HPP
#define JLM_LLVM_IR_LINKAGE_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/strfmt.hpp>
#include <string>
#include <unordered_map>

namespace jlm::llvm
{

/**
 * Types of linkage for global variables, constants and functions.
 * Based on LLVM's "GlobalValue::LinkageTypes"
 */
enum class linkage
{
  external_linkage,
  available_externally_linkage,
  link_once_any_linkage,
  link_once_odr_linkage,
  weak_any_linkage,
  weak_odr_linkage,
  appending_linkage,
  internal_linkage,
  private_linkage,
  external_weak_linkage,
  common_linkage
};

/**
 * Determines if a function / global variable with the given linkage should be exported.
 * @param lnk the linkage
 * @return true if the function / global should be exported, false otherwise
 */
inline bool
is_externally_visible(const linkage & lnk)
{
  // TODO: LLVM has a function isDiscardableIfUnused, which might be closer to what we mean
  // It is true for:
  //  - link once any
  //  - link once odr
  //  - internal
  //  - private
  //  - available_externally_linkage

  switch (lnk)
  {
  case linkage::external_linkage:
  case linkage::available_externally_linkage:
  case linkage::link_once_any_linkage:
  case linkage::link_once_odr_linkage:
  case linkage::weak_any_linkage:
  case linkage::weak_odr_linkage:
  case linkage::appending_linkage:
  case linkage::external_weak_linkage:
  case linkage::common_linkage:
    return true;
  case linkage::internal_linkage:
  case linkage::private_linkage:
    return false;
  default:
    JLM_UNREACHABLE("Unknown linkage type");
  }
}

inline std::string
ToString(const linkage & lnk)
{
  static std::unordered_map<linkage, std::string> strings = {
    { linkage::external_linkage, "external_linkage" },
    { linkage::available_externally_linkage, "available_externally_linkage" },
    { linkage::link_once_any_linkage, "link_once_any_linkage" },
    { linkage::link_once_odr_linkage, "link_once_odr_linkage" },
    { linkage::weak_any_linkage, "weak_any_linkage" },
    { linkage::weak_odr_linkage, "weak_odr_linkage" },
    { linkage::appending_linkage, "appending_linkage" },
    { linkage::internal_linkage, "internal_linkage" },
    { linkage::private_linkage, "private_linkage" },
    { linkage::external_weak_linkage, "external_weak_linkage" },
    { linkage::common_linkage, "common_linkage" }
  };

  JLM_ASSERT(strings.find(lnk) != strings.end());
  return strings[lnk];
}

inline linkage
FromString(const std::string_view stringValue)
{
  static std::unordered_map<std::string_view, linkage> linkages = {
    { "external_linkage", linkage::external_linkage },
    { "available_externally_linkage", linkage::available_externally_linkage },
    { "link_once_any_linkage", linkage::link_once_any_linkage },
    { "link_once_odr_linkage", linkage::link_once_odr_linkage },
    { "weak_any_linkage", linkage::weak_any_linkage },
    { "weak_odr_linkage", linkage::weak_odr_linkage },
    { "appending_linkage", linkage::appending_linkage },
    { "internal_linkage", linkage::internal_linkage },
    { "private_linkage", linkage::private_linkage },
    { "external_weak_linkage", linkage::external_weak_linkage },
    { "common_linkage", linkage::common_linkage }
  };

  if (linkages.find(stringValue) == linkages.end())
  {
    auto message = util::strfmt("Unsupported linkage: ", stringValue, "\n");
    JLM_UNREACHABLE(message.c_str());
  }

  return linkages[stringValue];
}
}

#endif
