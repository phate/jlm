/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/Linkage.hpp>

#include <string_view>
#include <unordered_map>

namespace jlm::llvm
{
[[nodiscard]] std::string_view
linkageToString(const Linkage lnk)
{
  static std::unordered_map<Linkage, std::string> strings = {
    { Linkage::external_linkage, "external_linkage" },
    { Linkage::available_externally_linkage, "available_externally_linkage" },
    { Linkage::link_once_any_linkage, "link_once_any_linkage" },
    { Linkage::link_once_odr_linkage, "link_once_odr_linkage" },
    { Linkage::weak_any_linkage, "weak_any_linkage" },
    { Linkage::weak_odr_linkage, "weak_odr_linkage" },
    { Linkage::appending_linkage, "appending_linkage" },
    { Linkage::internal_linkage, "internal_linkage" },
    { Linkage::private_linkage, "private_linkage" },
    { Linkage::external_weak_linkage, "external_weak_linkage" },
    { Linkage::common_linkage, "common_linkage" }
  };

  JLM_ASSERT(strings.find(lnk) != strings.end());
  return strings[lnk];
}

Linkage
linkageFromString(const std::string_view stringValue)
{
  static std::unordered_map<std::string_view, Linkage> linkages = {
    { "external_linkage", Linkage::external_linkage },
    { "available_externally_linkage", Linkage::available_externally_linkage },
    { "link_once_any_linkage", Linkage::link_once_any_linkage },
    { "link_once_odr_linkage", Linkage::link_once_odr_linkage },
    { "weak_any_linkage", Linkage::weak_any_linkage },
    { "weak_odr_linkage", Linkage::weak_odr_linkage },
    { "appending_linkage", Linkage::appending_linkage },
    { "internal_linkage", Linkage::internal_linkage },
    { "private_linkage", Linkage::private_linkage },
    { "external_weak_linkage", Linkage::external_weak_linkage },
    { "common_linkage", Linkage::common_linkage }
  };

  if (const auto it = linkages.find(stringValue); it != linkages.end())
    return it->second;

  throw std::logic_error("Unknown linkage type: " + std::string(stringValue));
}

}
