/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP
#define JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP

#include <jlm/llvm/backend/RegionSequentializer.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <memory>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::rvsdg
{
class Graph;
class GammaNode;
class input;
class LambdaNode;
class Node;
class Region;
}

namespace jlm::llvm
{

namespace phi
{
class node;
}

namespace delta
{
class node;
}

class cfg;
class data_node_init;
class ipgraph_module;
class RvsdgModule;
class variable;

class RvsdgToIpGraphConverter final
{
  class Context;
  class Statistics;

public:
  ~RvsdgToIpGraphConverter();

  explicit RvsdgToIpGraphConverter(
      const std::shared_ptr<RegionTreeSequentializer> & regionSequentializer);

  RvsdgToIpGraphConverter(const RvsdgToIpGraphConverter &) = delete;

  RvsdgToIpGraphConverter(RvsdgToIpGraphConverter &&) = delete;

  RvsdgToIpGraphConverter &
  operator=(const RvsdgToIpGraphConverter &) = delete;

  RvsdgToIpGraphConverter &
  operator=(RvsdgToIpGraphConverter &&) = delete;

  std::unique_ptr<ipgraph_module>
  ConvertModule(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector);

  static std::unique_ptr<ipgraph_module>
  CreateAndConvertModule(
      RvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector,
      const std::shared_ptr<RegionTreeSequentializer> & regionSequentializer);

private:
  void
  ConvertImports(const rvsdg::Graph & graph);

  void
  ConvertInterProceduralNodes(const rvsdg::Graph & graph);

  void
  ConvertInterProceduralNode(const rvsdg::Node & node);

  void
  ConvertIntraProceduralNode(const rvsdg::Node & node);

  void
  ConvertDeltaNode(const delta::node & deltaNode);

  void
  ConvertPhiNode(const phi::node & phiNode);

  void
  ConvertLambdaNode(const rvsdg::LambdaNode & lambdaNode);

  void
  ConvertThetaNode(const rvsdg::ThetaNode & thetaNode);

  void
  ConvertGammaNode(const rvsdg::GammaNode & gammaNode);

  void
  ConvertEmptyGammaNode(const rvsdg::GammaNode & gammaNode);

  void
  ConvertSimpleNode(const rvsdg::SimpleNode & simpleNode);

  std::unique_ptr<llvm::cfg>
  CreateControlFlowGraph(const rvsdg::LambdaNode & lambda);

  void
  ConvertRegion(rvsdg::Region & region);

  std::unique_ptr<data_node_init>
  CreateInitialization(const delta::node & deltaNode);

  static bool
  RequiresSsaPhiOperation(const rvsdg::ThetaNode::LoopVar & loopVar, const variable & v);

  std::unique_ptr<Context> Context_;

  std::shared_ptr<RegionTreeSequentializer> RegionTreeSequentializer_;
};

}

#endif // JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP
