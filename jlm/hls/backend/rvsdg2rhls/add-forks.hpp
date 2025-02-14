/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

/**
 * Adds a fork for every output that has multiple consumers (node inputs). The original output is
 * connected to the fork's input and each consumer is connected to one of the fork's outputs.
 *
 * /param region The region for which to insert forks.
 */
void
add_forks(rvsdg::Region * region);

/**
 * Adds a fork for every output that has multiple consumers (node inputs). The original output is
 * connected to the fork's input and each consumer is connected to one of the fork's outputs.
 *
 * /param rvsdgModule The RVSDG module for which to insert forks.
 */
void
add_forks(llvm::RvsdgModule & rvsdgModule);

}
#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP
