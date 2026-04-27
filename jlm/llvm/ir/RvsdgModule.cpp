/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

LlvmGraphImport &
LlvmGraphImport::Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) const
{
  // The copy interface is more general that what we support
  JLM_ASSERT(region.IsRootRegion());
  JLM_ASSERT(input == nullptr);

  return create(
      *region.graph(),
      ValueType(),
      ImportedType(),
      Name(),
      linkage(),
      callingConvention(),
      isConstant(),
      getAlignment());
}

std::unique_ptr<rvsdg::RvsdgModule>
LlvmRvsdgModule::copy() const
{
  return std::make_unique<LlvmRvsdgModule>(
      SourceFileName(),
      TargetTriple(),
      DataLayout(),
      Rvsdg().Copy());
}

}
