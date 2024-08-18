/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::llvm
{

GraphImport &
GraphImport::Copy(rvsdg::region & region, rvsdg::structural_input * input)
{
  return GraphImport::Create(*region.graph(), ValueType(), Name(), Linkage());
}

}
