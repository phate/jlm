/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>

namespace jlm::aa
{

MemoryNodeProvisioning::~MemoryNodeProvisioning() noexcept
= default;

MemoryNodeProvider::~MemoryNodeProvider() noexcept
= default;

}