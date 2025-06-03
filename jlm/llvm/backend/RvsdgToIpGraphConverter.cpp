/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <vector>

namespace jlm::llvm
{

class RvsdgToIpGraphConverter::Context final
{
public:
  explicit Context(ipgraph_module & ipGraphModule)
      : ControlFlowGraph_(nullptr),
        IPGraphModule_(ipGraphModule),
        LastProcessedBasicBlock(nullptr)
  {}

  Context(const Context &) = delete;

  Context(Context &&) = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) = delete;

  ipgraph_module &
  GetIpGraphModule() const noexcept
  {
    return IPGraphModule_;
  }

  void
  InsertVariable(const rvsdg::Output * output, const llvm::variable * variable)
  {
    JLM_ASSERT(VariableMap_.find(output) == VariableMap_.end());
    JLM_ASSERT(*output->Type() == *variable->Type());
    VariableMap_[output] = variable;
  }

  const llvm::variable *
  GetVariable(const rvsdg::Output * output)
  {
    const auto it = VariableMap_.find(output);
    JLM_ASSERT(it != VariableMap_.end());
    return it->second;
  }

  basic_block *
  GetLastProcessedBasicBlock() const noexcept
  {
    return LastProcessedBasicBlock;
  }

  void
  SetLastProcessedBasicBlock(basic_block * lastProcessedBasicBlock) noexcept
  {
    LastProcessedBasicBlock = lastProcessedBasicBlock;
  }

  llvm::cfg *
  GetControlFlowGraph() const noexcept
  {
    return ControlFlowGraph_;
  }

  void
  SetControlFlowGraph(llvm::cfg * cfg) noexcept
  {
    ControlFlowGraph_ = cfg;
  }

  static std::unique_ptr<Context>
  Create(ipgraph_module & ipGraphModule)
  {
    return std::make_unique<Context>(ipGraphModule);
  }

private:
  llvm::cfg * ControlFlowGraph_;
  ipgraph_module & IPGraphModule_;
  basic_block * LastProcessedBasicBlock;
  std::unordered_map<const rvsdg::Output *, const llvm::variable *> VariableMap_;
};

class RvsdgToIpGraphConverter::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & filename)
      : util::Statistics(Id::RvsdgDestruction, filename)
  {}

  void
  Start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  End(const ipgraph_module & im)
  {
    AddMeasurement(Label::NumThreeAddressCodes, llvm::ntacs(im));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

RvsdgToIpGraphConverter::~RvsdgToIpGraphConverter() = default;

RvsdgToIpGraphConverter::RvsdgToIpGraphConverter() = default;

std::unique_ptr<data_node_init>
RvsdgToIpGraphConverter::CreateInitialization(const delta::node & deltaNode)
{
  const auto subregion = deltaNode.subregion();

  // add delta dependencies to context
  for (size_t n = 0; n < deltaNode.ninputs(); n++)
  {
    const auto variable = Context_->GetVariable(deltaNode.input(n)->origin());
    Context_->InsertVariable(deltaNode.input(n)->arguments.first(), variable);
  }

  if (subregion->nnodes() == 0)
  {
    auto value = Context_->GetVariable(subregion->result(0)->origin());
    return std::make_unique<data_node_init>(value);
  }

  tacsvector_t tacs;
  for (const auto & node : rvsdg::TopDownTraverser(deltaNode.subregion()))
  {
    JLM_ASSERT(node->noutputs() == 1);
    const auto output = node->output(0);

    // collect operand variables
    std::vector<const variable *> operands;
    for (size_t n = 0; n < node->ninputs(); n++)
      operands.push_back(Context_->GetVariable(node->input(n)->origin()));

    // convert node to tac
    auto & op = *static_cast<const rvsdg::SimpleOperation *>(&node->GetOperation());
    tacs.push_back(tac::create(op, operands));
    Context_->InsertVariable(output, tacs.back()->result(0));
  }

  return std::make_unique<data_node_init>(std::move(tacs));
}

void
RvsdgToIpGraphConverter::ConvertRegion(rvsdg::Region & region)
{
  const auto entryBlock = basic_block::create(*Context_->GetControlFlowGraph());
  Context_->GetLastProcessedBasicBlock()->add_outedge(entryBlock);
  Context_->SetLastProcessedBasicBlock(entryBlock);

  for (const auto & node : rvsdg::TopDownTraverser(&region))
    ConvertNode(*node);

  const auto exitBlock = basic_block::create(*Context_->GetControlFlowGraph());
  Context_->GetLastProcessedBasicBlock()->add_outedge(exitBlock);
  Context_->SetLastProcessedBasicBlock(exitBlock);
}

std::unique_ptr<llvm::cfg>
RvsdgToIpGraphConverter::CreateControlFlowGraph(const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(Context_->GetLastProcessedBasicBlock() == nullptr);
  const auto & lambdaOperation = *util::AssertedCast<LlvmLambdaOperation>(&lambda.GetOperation());

  auto controlFlowGraph = cfg::create(Context_->GetIpGraphModule());
  const auto entryBlock = basic_block::create(*controlFlowGraph);
  controlFlowGraph->exit()->divert_inedges(entryBlock);
  Context_->SetLastProcessedBasicBlock(entryBlock);
  Context_->SetControlFlowGraph(controlFlowGraph.get());

  // add function arguments
  for (const auto functionArgument : lambda.GetFunctionArguments())
  {
    auto name = util::strfmt("_a", functionArgument->index(), "_");
    auto argument = argument::create(
        name,
        functionArgument->Type(),
        lambdaOperation.GetArgumentAttributes(functionArgument->index()));
    const auto variable = controlFlowGraph->entry()->append_argument(std::move(argument));
    Context_->InsertVariable(functionArgument, variable);
  }

  // add context variables
  for (const auto & [input, inner] : lambda.GetContextVars())
  {
    const auto variable = Context_->GetVariable(input->origin());
    Context_->InsertVariable(inner, variable);
  }

  ConvertRegion(*lambda.subregion());

  // add results
  for (const auto result : lambda.GetFunctionResults())
    controlFlowGraph->exit()->append_result(Context_->GetVariable(result->origin()));

  Context_->GetLastProcessedBasicBlock()->add_outedge(controlFlowGraph->exit());
  Context_->SetLastProcessedBasicBlock(nullptr);
  Context_->SetControlFlowGraph(nullptr);

  straighten(*controlFlowGraph);
  JLM_ASSERT(is_closed(*controlFlowGraph));
  return controlFlowGraph;
}

void
RvsdgToIpGraphConverter::ConvertSimpleNode(const rvsdg::SimpleNode & simpleNode)
{
  std::vector<const variable *> operands;
  for (size_t n = 0; n < simpleNode.ninputs(); n++)
    operands.push_back(Context_->GetVariable(simpleNode.input(n)->origin()));

  Context_->GetLastProcessedBasicBlock()->append_last(
      tac::create(simpleNode.GetOperation(), operands));

  for (size_t n = 0; n < simpleNode.noutputs(); n++)
    Context_->InsertVariable(
        simpleNode.output(n),
        Context_->GetLastProcessedBasicBlock()->last()->result(n));
}

void
RvsdgToIpGraphConverter::ConvertGammaNode(const rvsdg::GammaNode & gammaNode)
{
  const auto numSubregions = gammaNode.nsubregions();
  const auto predicate = gammaNode.predicate()->origin();
  const auto controlFlowGraph = Context_->GetControlFlowGraph();

  const auto entryBlock = basic_block::create(*controlFlowGraph);
  const auto exitBlock = basic_block::create(*controlFlowGraph);
  Context_->GetLastProcessedBasicBlock()->add_outedge(entryBlock);

  // convert gamma regions
  std::vector<cfg_node *> phi_nodes;
  entryBlock->append_last(BranchOperation::create(numSubregions, Context_->GetVariable(predicate)));
  auto entryvars = gammaNode.GetEntryVars();
  for (size_t n = 0; n < gammaNode.nsubregions(); n++)
  {
    const auto subregion = gammaNode.subregion(n);

    // add arguments to context
    for (size_t i = 0; i < entryvars.size(); i++)
    {
      auto & entryvar = entryvars[i];
      Context_->InsertVariable(
          entryvar.branchArgument[n],
          Context_->GetVariable(entryvar.input->origin()));
    }

    // convert subregion
    const auto regionEntryBlock = basic_block::create(*controlFlowGraph);
    entryBlock->add_outedge(regionEntryBlock);
    Context_->SetLastProcessedBasicBlock(regionEntryBlock);
    ConvertRegion(*subregion);

    phi_nodes.push_back(Context_->GetLastProcessedBasicBlock());
    Context_->GetLastProcessedBasicBlock()->add_outedge(exitBlock);
  }

  // add phi instructions
  for (size_t n = 0; n < gammaNode.noutputs(); n++)
  {
    const auto output = gammaNode.output(n);

    bool invariant = true;
    std::vector<std::pair<const variable *, cfg_node *>> arguments;
    for (size_t r = 0; r < gammaNode.nsubregions(); r++)
    {
      const auto origin = gammaNode.subregion(r)->result(n)->origin();

      auto v = Context_->GetVariable(origin);
      arguments.push_back(std::make_pair(v, phi_nodes[r]));
      invariant &= (v == Context_->GetVariable(gammaNode.subregion(0)->result(n)->origin()));
    }

    if (invariant)
    {
      // all operands are the same
      Context_->InsertVariable(output, arguments[0].first);
      continue;
    }

    // create phi instruction
    exitBlock->append_last(SsaPhiOperation::create(arguments, output->Type()));
    Context_->InsertVariable(output, exitBlock->last()->result(0));
  }

  Context_->SetLastProcessedBasicBlock(exitBlock);
}

bool
RvsdgToIpGraphConverter::RequiresSsaPhiOperation(
    const rvsdg::ThetaNode::LoopVar & loopVar,
    const variable & v)
{
  // FIXME: solely decide on the input instead of using the variable
  if (is<gblvariable>(&v))
    return false;

  if (ThetaLoopVarIsInvariant(loopVar))
    return false;

  if (loopVar.pre->nusers() == 0)
    return false;

  return true;
}

void
RvsdgToIpGraphConverter::ConvertThetaNode(const rvsdg::ThetaNode & thetaNode)
{
  const auto subregion = thetaNode.subregion();
  const auto predicate = subregion->result(0)->origin();

  auto preEntryBlock = Context_->GetLastProcessedBasicBlock();
  const auto entryBlock = basic_block::create(*Context_->GetControlFlowGraph());
  preEntryBlock->add_outedge(entryBlock);
  Context_->SetLastProcessedBasicBlock(entryBlock);

  // create SSA phi nodes in entry block and add arguments to context
  std::vector<llvm::tac *> phis;
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    auto variable = Context_->GetVariable(loopVar.input->origin());
    if (RequiresSsaPhiOperation(loopVar, *variable))
    {
      auto phi = entryBlock->append_last(SsaPhiOperation::create({}, loopVar.pre->Type()));
      phis.push_back(phi);
      variable = phi->result(0);
    }
    Context_->InsertVariable(loopVar.pre, variable);
  }

  ConvertRegion(*subregion);

  // add phi operands and results to context
  size_t phiIndex = 0;
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    auto entryVariable = Context_->GetVariable(loopVar.input->origin());
    if (RequiresSsaPhiOperation(loopVar, *entryVariable))
    {
      auto resultVariable = Context_->GetVariable(loopVar.post->origin());
      const auto phi = phis[phiIndex++];
      phi->replace(
          SsaPhiOperation(
              { preEntryBlock, Context_->GetLastProcessedBasicBlock() },
              resultVariable->Type()),
          { entryVariable, resultVariable });
      Context_->InsertVariable(loopVar.output, resultVariable);
    }
    else
    {
      Context_->InsertVariable(loopVar.output, Context_->GetVariable(loopVar.post->origin()));
    }
  }
  JLM_ASSERT(phiIndex == phis.size());

  Context_->GetLastProcessedBasicBlock()->append_last(
      BranchOperation::create(2, Context_->GetVariable(predicate)));
  const auto exitBlock = basic_block::create(*Context_->GetControlFlowGraph());
  Context_->GetLastProcessedBasicBlock()->add_outedge(exitBlock);
  Context_->GetLastProcessedBasicBlock()->add_outedge(entryBlock);
  Context_->SetLastProcessedBasicBlock(exitBlock);
}

void
RvsdgToIpGraphConverter::ConvertLambdaNode(const rvsdg::LambdaNode & lambdaNode)
{
  auto & ipGraphModule = Context_->GetIpGraphModule();
  auto & ipGraph = ipGraphModule.ipgraph();

  const auto & operation = *util::AssertedCast<LlvmLambdaOperation>(&lambdaNode.GetOperation());
  const auto functionNode = function_node::create(
      ipGraph,
      operation.name(),
      operation.Type(),
      operation.linkage(),
      operation.attributes());
  const auto variable = ipGraphModule.create_variable(functionNode);

  functionNode->add_cfg(CreateControlFlowGraph(lambdaNode));
  Context_->InsertVariable(lambdaNode.output(), variable);
}

void
RvsdgToIpGraphConverter::ConvertPhiNode(const rvsdg::PhiNode & phiNode)
{
  const auto subregion = phiNode.subregion();
  auto & ipGraphModule = Context_->GetIpGraphModule();
  auto & ipGraph = ipGraphModule.ipgraph();

  // add dependencies to context
  for (size_t n = 0; n < phiNode.ninputs(); n++)
  {
    const auto variable = Context_->GetVariable(phiNode.input(n)->origin());
    Context_->InsertVariable(phiNode.input(n)->arguments.first(), variable);
  }

  // forward declare all functions and global variables
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    const auto & origin = *subregion->result(n)->origin();

    if (const auto lambdaNode = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(origin))
    {
      const auto & lambdaOperation =
          dynamic_cast<LlvmLambdaOperation &>(lambdaNode->GetOperation());
      const auto functionNode = function_node::create(
          ipGraph,
          lambdaOperation.name(),
          lambdaOperation.Type(),
          lambdaOperation.linkage(),
          lambdaOperation.attributes());
      Context_->InsertVariable(subregion->argument(n), ipGraphModule.create_variable(functionNode));
    }
    else if (const auto deltaNode = rvsdg::TryGetOwnerNode<delta::node>(origin))
    {
      const auto dataNode = data_node::Create(
          ipGraph,
          deltaNode->name(),
          deltaNode->Type(),
          deltaNode->linkage(),
          deltaNode->Section(),
          deltaNode->constant());
      Context_->InsertVariable(subregion->argument(n), ipGraphModule.create_global_value(dataNode));
    }
    else
    {
      JLM_UNREACHABLE(util::strfmt(
                          "Unhandled node type: ",
                          rvsdg::AssertGetOwnerNode<rvsdg::Node>(origin).DebugString())
                          .c_str());
    }
  }

  // convert function bodies and global variable initializations
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    const auto result = subregion->result(n);
    const auto & origin = *result->origin();

    if (const auto lambdaNode = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(origin))
    {
      const auto variable =
          util::AssertedCast<const fctvariable>(Context_->GetVariable(subregion->argument(n)));
      variable->function()->add_cfg(CreateControlFlowGraph(*lambdaNode));
      Context_->InsertVariable(lambdaNode->output(), variable);
    }
    else if (const auto deltaNode = rvsdg::TryGetOwnerNode<delta::node>(origin))
    {
      const auto variable =
          util::AssertedCast<const gblvalue>(Context_->GetVariable(subregion->argument(n)));
      variable->node()->set_initialization(CreateInitialization(*deltaNode));
      Context_->InsertVariable(deltaNode->output(), variable);
    }
    else
    {
      JLM_UNREACHABLE(util::strfmt(
                          "Unhandled node type: ",
                          rvsdg::AssertGetOwnerNode<rvsdg::Node>(origin).DebugString())
                          .c_str());
    }
  }

  // add functions and globals to context
  JLM_ASSERT(phiNode.noutputs() == subregion->nresults());
  for (size_t n = 0; n < phiNode.noutputs(); n++)
    Context_->InsertVariable(
        phiNode.output(n),
        Context_->GetVariable(subregion->result(n)->origin()));
}

void
RvsdgToIpGraphConverter::ConvertDeltaNode(const delta::node & deltaNode)
{
  auto & ipGraphModule = Context_->GetIpGraphModule();

  const auto dataNode = data_node::Create(
      ipGraphModule.ipgraph(),
      deltaNode.name(),
      deltaNode.Type(),
      deltaNode.linkage(),
      deltaNode.Section(),
      deltaNode.constant());
  dataNode->set_initialization(CreateInitialization(deltaNode));
  const auto variable = ipGraphModule.create_global_value(dataNode);
  Context_->InsertVariable(deltaNode.output(), variable);
}

void
RvsdgToIpGraphConverter::ConvertNode(const rvsdg::Node & node)
{
  if (const auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(&node))
  {
    ConvertLambdaNode(*lambdaNode);
  }
  else if (const auto gammaNode = dynamic_cast<const rvsdg::GammaNode *>(&node))
  {
    ConvertGammaNode(*gammaNode);
  }
  else if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(&node))
  {
    ConvertThetaNode(*thetaNode);
  }
  else if (const auto phiNode = dynamic_cast<const rvsdg::PhiNode *>(&node))
  {
    ConvertPhiNode(*phiNode);
  }
  else if (const auto deltaNode = dynamic_cast<const delta::node *>(&node))
  {
    ConvertDeltaNode(*deltaNode);
  }
  else if (const auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
  {
    ConvertSimpleNode(*simpleNode);
  }
  else
  {
    JLM_UNREACHABLE(util::strfmt("Unhandled node type: ", node.DebugString()).c_str());
  }
}

void
RvsdgToIpGraphConverter::ConvertNodes(const rvsdg::Graph & graph)
{
  for (const auto & node : rvsdg::TopDownTraverser(&graph.GetRootRegion()))
  {
    ConvertNode(*node);
  }
}

void
RvsdgToIpGraphConverter::ConvertImports(const rvsdg::Graph & graph)
{
  auto & ipGraphModule = Context_->GetIpGraphModule();
  auto & ipGraph = ipGraphModule.ipgraph();

  for (size_t n = 0; n < graph.GetRootRegion().narguments(); n++)
  {
    const auto graphImport = util::AssertedCast<GraphImport>(graph.GetRootRegion().argument(n));
    if (const auto functionType =
            std::dynamic_pointer_cast<const rvsdg::FunctionType>(graphImport->ValueType()))
    {
      const auto functionNode =
          function_node::create(ipGraph, graphImport->Name(), functionType, graphImport->Linkage());
      const auto variable = ipGraphModule.create_variable(functionNode);
      Context_->InsertVariable(graphImport, variable);
    }
    else
    {
      const auto dataNode = data_node::Create(
          ipGraph,
          graphImport->Name(),
          graphImport->ValueType(),
          graphImport->Linkage(),
          "",
          false);
      const auto variable = ipGraphModule.create_global_value(dataNode);
      Context_->InsertVariable(graphImport, variable);
    }
  }
}

std::unique_ptr<ipgraph_module>
RvsdgToIpGraphConverter::ConvertModule(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFileName());
  statistics->Start(rvsdgModule.Rvsdg());

  auto ipGraphModule = ipgraph_module::Create(
      rvsdgModule.SourceFileName(),
      rvsdgModule.TargetTriple(),
      rvsdgModule.DataLayout(),
      std::move(rvsdgModule.ReleaseStructTypeDeclarations()));

  Context_ = Context::Create(*ipGraphModule);
  ConvertImports(rvsdgModule.Rvsdg());
  ConvertNodes(rvsdgModule.Rvsdg());

  statistics->End(*ipGraphModule);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return ipGraphModule;
}

std::unique_ptr<ipgraph_module>
RvsdgToIpGraphConverter::CreateAndConvertModule(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  RvsdgToIpGraphConverter converter;
  return converter.ConvertModule(rvsdgModule, statisticsCollector);
}

}
