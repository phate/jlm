/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/static/rvsdg2rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/static/ThetaConversion.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::static_hls
{

using namespace jlm::static_hls;

void
rvsdg2rhls(llvm::RvsdgModule & rvsdgModule)
{
    ConvertThetaNodes(rvsdgModule);
}

} // namespace jlm::static_hls