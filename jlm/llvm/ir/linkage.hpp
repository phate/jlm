/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_LINKAGE_HPP
#define JLM_LLVM_IR_LINKAGE_HPP

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
linkage_to_string(const linkage & lnk)
{
  std::unordered_map<linkage, std::string> strings = {
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

  return strings[lnk];
}

}

#endif
