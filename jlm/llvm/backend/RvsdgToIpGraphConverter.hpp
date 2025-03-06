/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP
#define JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP

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

  RvsdgToIpGraphConverter();

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
      util::StatisticsCollector & statisticsCollector);

private:
  void
  ConvertImports(const rvsdg::Graph & graph);

  void
  convert_nodes(const rvsdg::Graph & graph);

  void
  convert_node(const rvsdg::Node & node);

  void
  convert_delta_node(const rvsdg::Node & node);

  void
  convert_phi_node(const rvsdg::Node & node);

  void
  convert_lambda_node(const rvsdg::Node & node);

  void
  convert_theta_node(const rvsdg::Node & node);

  bool
  phi_needed(const rvsdg::input * i, const llvm::variable * v);

  void
  convert_gamma_node(const rvsdg::Node & node);

  void
  convert_empty_gamma_node(const rvsdg::GammaNode * gamma);

  void
  convert_simple_node(const rvsdg::Node & node);

  std::unique_ptr<llvm::cfg>
  create_cfg(const rvsdg::LambdaNode & lambda);

  void
  convert_region(rvsdg::Region & region);

  std::unique_ptr<data_node_init>
  create_initialization(const delta::node * delta);

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_BACKEND_RVSDGTOIPGRAPHCONVERTER_HPP
