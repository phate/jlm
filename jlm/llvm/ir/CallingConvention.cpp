/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/CallingConvention.hpp>
#include <jlm/util/BijectiveMap.hpp>
#include <jlm/util/common.hpp>

#include <llvm/IR/CallingConv.h>

namespace jlm::llvm
{

static const util::BijectiveMap<::llvm::CallingConv::ID, CallingConvention> &
getCallingConventionMap()
{
  static util::BijectiveMap<::llvm::CallingConv::ID, CallingConvention> map = {
    { ::llvm::CallingConv::C, CallingConvention::C },
    { ::llvm::CallingConv::Fast, CallingConvention::Fast },
    { ::llvm::CallingConv::Cold, CallingConvention::Cold },
    { ::llvm::CallingConv::Tail, CallingConvention::Tail },
  };
  return map;
}

jlm::llvm::CallingConvention
convertCallingConventionToJlm(::llvm::CallingConv::ID cc)
{
  const auto & map = getCallingConventionMap();
  // LookupKey throws if the calling convention could not be found
  return map.LookupKey(cc);
}

::llvm::CallingConv::ID
convertCallingConventionToLlvm(jlm::llvm::CallingConvention cc)
{
  const auto & map = getCallingConventionMap();
  // LookupValue throws if the calling convention could not be found
  return map.LookupValue(cc);
}

}
