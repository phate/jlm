/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::llvm
{

GraphImport &
GraphImport::Copy(rvsdg::Region & region, rvsdg::StructuralInput *)
{
  return GraphImport::Create(*region.graph(), ValueType(), ImportedType(), Name(), Linkage());
}

GraphExport &
GraphExport::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
{
  JLM_ASSERT(output == nullptr);
  return GraphExport::Create(origin, Name());
}

}
