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
class RegionIntervalOutputMapping;

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
  EncodeInterProceduralRegion(rvsdg::Region & region);

  void
  EncodePhi(rvsdg::PhiNode & phiNode);

  void
  EncodeLambda(rvsdg::LambdaNode & lambda);

  void
  EncodeLambdaEntry(
      rvsdg::LambdaNode & lambdaNode,
      RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeLambdaExit(
      rvsdg::LambdaNode & lambdaNode,
      RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeIntraProceduralRegion(rvsdg::Region & region, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeSimpleNode(
      rvsdg::SimpleNode & simpleNode,
      RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeAlloca(rvsdg::SimpleNode & allocaNode, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeMalloc(rvsdg::SimpleNode & mallocNode, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeLoad(rvsdg::SimpleNode & node, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeStore(rvsdg::SimpleNode & node, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeFree(rvsdg::SimpleNode & freeNode, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeCall(rvsdg::SimpleNode & callNode, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeMemcpy(rvsdg::SimpleNode & memcpyNode, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeGamma(rvsdg::GammaNode & gammaNode, RegionIntervalOutputMapping & liveIntervals);

  void
  EncodeTheta(rvsdg::ThetaNode & thetaNode, RegionIntervalOutputMapping & liveIntervals);

  [[nodiscard]] bool
  ShouldHandle(const rvsdg::SimpleNode & simpleNode) const noexcept;

  std::unique_ptr<Context> Context_;
};

}
}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
