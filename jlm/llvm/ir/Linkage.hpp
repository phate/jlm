/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_LINKAGE_HPP
#define JLM_LLVM_IR_LINKAGE_HPP

#include <jlm/util/common.hpp>

namespace jlm::llvm
{

/**
 * Types of linkage for global variables, constants and functions.
 * Based on LLVM's "::llvm::GlobalValue::LinkageTypes"
 */
enum class Linkage
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
 * Determines if a function / global variable with the given linkage can be removed,
 * if it is not used within the current module.
 * @param linkage the linkage type
 * @return true if the global value can be discarded.
 *
 * @note Based on LLVM's "::llvm::GlobalValue::isDiscardableIfUnused"
 */
[[nodiscard]] inline bool
isDiscardableIfUnused(const Linkage linkage)
{
  switch (linkage)
  {
  case Linkage::available_externally_linkage:
  case Linkage::link_once_any_linkage:
  case Linkage::link_once_odr_linkage:
  case Linkage::internal_linkage:
  case Linkage::private_linkage:
    return true;
  default:
    return false;
  }
}

/**
 * Checks if the given linkage is private or internal.
 * Internal symbols are only included in the module's own local symbol table,
 * while private linkage is excluded from all symbol tables.
 * @param linkage the linkage
 * @return true if the linkage type is private or internal
 */
[[nodiscard]] inline bool
isPrivateOrInternal(const Linkage linkage)
{
  return linkage == Linkage::private_linkage || linkage == Linkage::internal_linkage;
}

[[nodiscard]] std::string_view
linkageToString(Linkage lnk);

[[nodiscard]] Linkage
linkageFromString(std::string_view stringValue);
}

#endif
