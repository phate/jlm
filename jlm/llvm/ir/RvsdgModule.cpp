/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::llvm
{

GraphImport &
GraphImport::Copy(rvsdg::Region & region, rvsdg::StructuralInput *) const
{
  return Create(*region.graph(), ValueType(), ImportedType(), Name(), linkage(), isConstant());
}

}
