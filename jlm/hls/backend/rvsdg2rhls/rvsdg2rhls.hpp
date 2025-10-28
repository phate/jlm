/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_RVSDG2RHLS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_RVSDG2RHLS_HPP

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::hls
{

static inline bool
is_constant(const rvsdg::Node * node)
{
  return jlm::rvsdg::is<llvm::IntegerConstantOperation>(node)
      || jlm::rvsdg::is<llvm::UndefValueOperation>(node) || jlm::rvsdg::is<llvm::ConstantFP>(node)
      || jlm::rvsdg::is<rvsdg::ControlConstantOperation>(node);
}

std::unique_ptr<rvsdg::TransformationSequence>
createTransformationSequence(rvsdg::DotWriter & dotWriter, bool dumpRvsdgDotGraphs);

void
rvsdg2ref(llvm::RvsdgModule & rm, const util::FilePath & function_name);

void
dump_ref(llvm::RvsdgModule & rhls, const util::FilePath & function_name);

const jlm::rvsdg::Output *
trace_call(jlm::rvsdg::Input * input);

std::unique_ptr<llvm::RvsdgModule>
split_hls_function(llvm::RvsdgModule & rm, const std::string & function_name);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_RVSDG2RHLS_HPP
