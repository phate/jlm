/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_GAMMACONVERSION_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_GAMMACONVERSION_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm::hls
{

/**
 * Converts every rvsdg::GammaNode in \p rvsdgModule to its respective HLS equivalent.
 *
 * @param rvsdgModule The RVSDG module the transformation is performed on.
 */
void
ConvertGammaNodes(llvm::RvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_GAMMACONVERSION_HPP
