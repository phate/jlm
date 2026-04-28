/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_CALLINGCONVENTION_HPP
#define JLM_LLVM_IR_CALLINGCONVENTION_HPP

#include <llvm/IR/CallingConv.h>

namespace jlm::llvm
{

/**
 * Types of calling conventions.
 * Based on LLVM's "::llvm::CallingConv" namespace.
 * LLVM allows arbitrary numbers to be used as calling convention identifiers,
 * but jlm limits the set of possible values to this enum.
 */
enum class CallingConvention
{
  // The default llvm calling convention, compatible with C.
  // Only convention to support varargs. Supports some prototype mismatch.
  C,

  // Default is defined as an alias for the C calling convention
  Default = C,

  // The following are generic LLVM calling conventions. None of these support varargs calls,
  // and all assume that the caller and callee prototype exactly match.

  // Attempts to make calls as fast as possible (e.g. by passing things in
  // registers).
  Fast,

  // Attempts to make code in the caller as efficient as possible under the
  // assumption that the call is not commonly executed. As such, these calls
  // often preserve all registers so that the call does not break any live
  // ranges in the caller side.
  Cold,

  // Attemps to make calls as fast as possible while guaranteeing that tail
  // call optimization can always be performed.
  Tail,
};

/**
 * Converts the given calling convention from LLVM to jlm.
 * @param cc the calling convention
 * @return the same calling convention as a jlm enum value
 * @throws jlm::util::Error if the calling convention is unknown
 */
[[nodiscard]] jlm::llvm::CallingConvention
convertCallingConventionToJlm(::llvm::CallingConv::ID cc);

/**
 * Converts the given calling convention from jlm to LLVM.
 * @param cc the calling convention
 * @return the same calling convention as an LLVM enum value
 * @throws jlm::util::Error if the calling convention is unknown
 */
[[nodiscard]] ::llvm::CallingConv::ID
convertCallingConventionToLlvm(jlm::llvm::CallingConvention cc);

}

#endif // JLM_LLVM_IR_CALLINGCONVENTION_HPP
