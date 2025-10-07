/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_LINKAGE_HPP
#define JLM_LLVM_IR_LINKAGE_HPP

#include <string_view>

namespace jlm::llvm
{

/**
 * Types of linkage for global variables, constants and functions.
 * Based on LLVM's "::llvm::GlobalValue::LinkageTypes"
 */
enum class Linkage
{
  externalLinkage,
  availableExternallyLinkage,
  linkOnceAnyLinkage,
  linkOnceOdrLinkage,
  weakAnyLinkage,
  weakOdrLinkage,
  appendingLinkage,
  // internal symbols are only included in the module's own local symbol table.
  internalLinkage,
  // private linkage is excluded from all symbol tables.
  privateLinkage,
  externalWeakLinkage,
  commonLinkage
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
  case Linkage::availableExternallyLinkage:
  case Linkage::linkOnceAnyLinkage:
  case Linkage::linkOnceOdrLinkage:
  case Linkage::internalLinkage:
  case Linkage::privateLinkage:
    return true;
  default:
    return false;
  }
}

/**
 * Checks if the given linkage is private or internal.
 * @param linkage the linkage
 * @return true if the linkage type is private or internal
 */
[[nodiscard]] inline bool
isPrivateOrInternal(const Linkage linkage)
{
  return linkage == Linkage::privateLinkage || linkage == Linkage::internalLinkage;
}

[[nodiscard]] std::string_view
linkageToString(Linkage linkage);

[[nodiscard]] Linkage
linkageFromString(std::string_view stringValue);
}

#endif
