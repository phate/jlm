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
linkageToString(const Linkage linkage)
{
  static std::unordered_map<Linkage, std::string> strings = {
    { Linkage::externalLinkage, "external_linkage" },
    { Linkage::availableExternallyLinkage, "available_externally_linkage" },
    { Linkage::linkOnceAnyLinkage, "link_once_any_linkage" },
    { Linkage::linkOnceOdrLinkage, "link_once_odr_linkage" },
    { Linkage::weakAnyLinkage, "weak_any_linkage" },
    { Linkage::weakOdrLinkage, "weak_odr_linkage" },
    { Linkage::appendingLinkage, "appending_linkage" },
    { Linkage::internalLinkage, "internal_linkage" },
    { Linkage::privateLinkage, "private_linkage" },
    { Linkage::externalWeakLinkage, "external_weak_linkage" },
    { Linkage::commonLinkage, "common_linkage" }
  };

  if (const auto it = strings.find(linkage); it != strings.end())
    return it->second;

  throw std::logic_error(
      "Unknown linkage type: "
      + std::to_string(static_cast<std::underlying_type_t<Linkage>>(linkage)));
}

Linkage
linkageFromString(const std::string_view stringValue)
{
  static std::unordered_map<std::string_view, Linkage> linkages = {
    { "external_linkage", Linkage::externalLinkage },
    { "available_externally_linkage", Linkage::availableExternallyLinkage },
    { "link_once_any_linkage", Linkage::linkOnceAnyLinkage },
    { "link_once_odr_linkage", Linkage::linkOnceOdrLinkage },
    { "weak_any_linkage", Linkage::weakAnyLinkage },
    { "weak_odr_linkage", Linkage::weakOdrLinkage },
    { "appending_linkage", Linkage::appendingLinkage },
    { "internal_linkage", Linkage::internalLinkage },
    { "private_linkage", Linkage::privateLinkage },
    { "external_weak_linkage", Linkage::externalWeakLinkage },
    { "common_linkage", Linkage::commonLinkage }
  };

  if (const auto it = linkages.find(stringValue); it != linkages.end())
    return it->second;

  throw std::logic_error("Unknown linkage type: " + std::string(stringValue));
}

}
