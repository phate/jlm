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
  auto simple_node = dynamic_cast<const rvsdg::SimpleNode *>(node);
  if (!simple_node)
    return false;

  const auto & op = simple_node->GetOperation();
  return jlm::rvsdg::is<llvm::IntegerConstantOperation>(op)
      || jlm::rvsdg::is<llvm::UndefValueOperation>(op) || jlm::rvsdg::is<llvm::ConstantFP>(op)
      || jlm::rvsdg::is<rvsdg::ControlConstantOperation>(op);
}

std::unique_ptr<rvsdg::TransformationSequence>
createTransformationSequence(rvsdg::DotWriter & dotWriter, bool dumpRvsdgDotGraphs);

void
rvsdg2ref(llvm::LlvmRvsdgModule & rm, const util::FilePath & function_name);

void
dump_ref(llvm::LlvmRvsdgModule & rhls, const util::FilePath & function_name);

const jlm::rvsdg::Output *
trace_call(jlm::rvsdg::Input * input);

std::unique_ptr<llvm::LlvmRvsdgModule>
split_hls_function(llvm::LlvmRvsdgModule & rm, const std::string & function_name);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_RVSDG2RHLS_HPP
