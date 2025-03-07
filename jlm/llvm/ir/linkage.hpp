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

static inline bool
is_externally_visible(const linkage & lnk)
{
  /* FIXME: Refine this again. */
  return lnk != linkage::internal_linkage;
}

static inline std::string
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

static inline linkage
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
