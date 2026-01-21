/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/RvsdgModule.hpp>
#include <memory>

namespace jlm::rvsdg
{

RvsdgModule::~RvsdgModule() noexcept = default;

std::unique_ptr<RvsdgModule>
RvsdgModule::copy() const
{
  return std::make_unique<RvsdgModule>(*SourceFilePath_, rvsdg_->Copy());
}

}
