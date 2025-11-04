/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP

#include <memory>
#include <vector>

namespace jlm::rvsdg
{
class DeltaNode;
class GammaNode;
class LambdaNode;
class Output;
class PhiNode;
class Region;
class RvsdgModule;
class SimpleNode;
class StructuralNode;
class ThetaNode;
}

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

namespace aa
{

class ModRefSummary;

class MemoryStateEncoder final
{
public:
  class Context;

  ~MemoryStateEncoder() noexcept;

  MemoryStateEncoder();

  MemoryStateEncoder(const MemoryStateEncoder &) = delete;

  MemoryStateEncoder &
  operator=(const MemoryStateEncoder &) = delete;

  void
  Encode(
      rvsdg::RvsdgModule & rvsdgModule,
      const ModRefSummary & modRefSummary,
      util::StatisticsCollector & statisticsCollector);

private:
  void
  EncodeRegion(rvsdg::Region & region);

  void
  EncodeStructuralNode(rvsdg::StructuralNode & structuralNode);

  void
  EncodeSimpleNode(const rvsdg::SimpleNode & simpleNode);

  void
  EncodeAlloca(const rvsdg::SimpleNode & allocaNode);

  void
  EncodeMalloc(const rvsdg::SimpleNode & mallocNode);

  void
  EncodeLoad(const rvsdg::SimpleNode & node);

  void
  EncodeStore(const rvsdg::SimpleNode & node);

  void
  EncodeFree(const rvsdg::SimpleNode & freeNode);

  void
  EncodeCall(const rvsdg::SimpleNode & callNode);

  void
  EncodeMemcpy(const rvsdg::SimpleNode & memcpyNode);

  void
  EncodeLambda(const rvsdg::LambdaNode & lambda);

  void
  EncodeLambdaEntry(const rvsdg::LambdaNode & lambdaNode);

  void
  EncodeLambdaExit(const rvsdg::LambdaNode & lambdaNode);

  void
  EncodePhi(const rvsdg::PhiNode & phiNode);

  void
  EncodeDelta(const rvsdg::DeltaNode & deltaNode);

  void
  EncodeGamma(rvsdg::GammaNode & gammaNode);

  void
  EncodeGammaEntry(rvsdg::GammaNode & gammaNode);

  void
  EncodeGammaExit(rvsdg::GammaNode & gammaNode);

  void
  EncodeTheta(rvsdg::ThetaNode & thetaNode);

  std::vector<rvsdg::Output *>
  EncodeThetaEntry(rvsdg::ThetaNode & thetaNode);

  void
  EncodeThetaExit(
      rvsdg::ThetaNode & thetaNode,
      const std::vector<rvsdg::Output *> & thetaStateOutputs);

  std::unique_ptr<Context> Context_;
};

}
}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
