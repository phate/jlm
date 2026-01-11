/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP
#define JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP

#include <jlm/rvsdg/theta.hpp>

#include <memory>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::rvsdg
{
class DeltaNode;
class Graph;
class GammaNode;
class Input;
class LambdaNode;
class Node;
class PhiNode;
class Region;
}

namespace jlm::llvm
{

class ControlFlowGraph;
class DataNodeInit;
class InterProceduralGraphModule;
class LlvmRvsdgModule;
class Variable;

class RvsdgToIpGraphConverter final
{
  class Context;
  class Statistics;

public:
  ~RvsdgToIpGraphConverter();

  RvsdgToIpGraphConverter();

  RvsdgToIpGraphConverter(const RvsdgToIpGraphConverter &) = delete;

  RvsdgToIpGraphConverter(RvsdgToIpGraphConverter &&) = delete;

  RvsdgToIpGraphConverter &
  operator=(const RvsdgToIpGraphConverter &) = delete;

  RvsdgToIpGraphConverter &
  operator=(RvsdgToIpGraphConverter &&) = delete;

  std::unique_ptr<InterProceduralGraphModule>
  ConvertModule(LlvmRvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector);

  static std::unique_ptr<InterProceduralGraphModule>
  CreateAndConvertModule(
      LlvmRvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector);

private:
  void
  ConvertImports(const rvsdg::Graph & graph);

  void
  ConvertNodes(const rvsdg::Graph & graph);

  void
  ConvertNode(const rvsdg::Node & node);

  void
  ConvertDeltaNode(const rvsdg::DeltaNode & deltaNode);

  void
  ConvertPhiNode(const rvsdg::PhiNode & phiNode);

  void
  ConvertLambdaNode(const rvsdg::LambdaNode & lambdaNode);

  void
  ConvertThetaNode(const rvsdg::ThetaNode & thetaNode);

  void
  ConvertGammaNode(const rvsdg::GammaNode & gammaNode);

  void
  ConvertSimpleNode(const rvsdg::SimpleNode & simpleNode);

  std::unique_ptr<ControlFlowGraph>
  CreateControlFlowGraph(const rvsdg::LambdaNode & lambda);

  void
  ConvertRegion(rvsdg::Region & region);

  std::unique_ptr<DataNodeInit>
  CreateInitialization(const rvsdg::DeltaNode & deltaNode);

  static bool
  RequiresSsaPhiOperation(const rvsdg::ThetaNode::LoopVar & loopVar, const Variable & v);

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP
