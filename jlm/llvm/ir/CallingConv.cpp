/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/util/common.hpp"
#include <jlm/llvm/ir/CallingConv.hpp>
#include <jlm/util/BijectiveMap.hpp>
#include <llvm/IR/CallingConv.h>

namespace jlm::llvm
{

static const util::BijectiveMap<::llvm::CallingConv::ID, CallingConv> &
getCallingConventionMap()
{
  static util::BijectiveMap<::llvm::CallingConv::ID, CallingConv> map = {
    { ::llvm::CallingConv::C, CallingConv::C },
    { ::llvm::CallingConv::Fast, CallingConv::Fast },
    { ::llvm::CallingConv::Cold, CallingConv::Cold },
    { ::llvm::CallingConv::Tail, CallingConv::Tail },
  };
  return map;
}

jlm::llvm::CallingConv
convertCallingConvToJlm(::llvm::CallingConv::ID cc)
{
  const auto & map = getCallingConventionMap();
  JLM_ASSERT(map.HasKey(cc) && "Unknown calling convention");
  return map.LookupKey(cc);
}

::llvm::CallingConv::ID
convertCallingConvToLlvm(jlm::llvm::CallingConv cc)
{
  const auto & map = getCallingConventionMap();
  JLM_ASSERT(map.HasValue(cc) && "Unknown calling convention");
  return map.LookupValue(cc);
}

}
