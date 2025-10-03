/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_HLSDOTWRITER_HPP
#define JLM_HLS_HLSDOTWRITER_HPP

#include <jlm/llvm/DotWriter.hpp>

namespace jlm::hls
{

class HlsDotWriter final : public llvm::LlvmDotWriter
{
public:
  ~HlsDotWriter() noexcept override;

protected:
  void
  AnnotateTypeGraphNode(const rvsdg::Type & type, util::graph::Node & node) override;
};

}

#endif
