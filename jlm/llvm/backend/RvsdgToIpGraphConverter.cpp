/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
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
      : cfg_(nullptr),
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
  insert(const rvsdg::output * output, const llvm::variable * v)
  {
    JLM_ASSERT(ports_.find(output) == ports_.end());
    JLM_ASSERT(*output->Type() == *v->Type());
    ports_[output] = v;
  }

  const llvm::variable *
  variable(const rvsdg::output * port)
  {
    auto it = ports_.find(port);
    JLM_ASSERT(it != ports_.end());
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
  cfg() const noexcept
  {
    return cfg_;
  }

  void
  set_cfg(llvm::cfg * cfg) noexcept
  {
    cfg_ = cfg;
  }

  static std::unique_ptr<Context>
  Create(ipgraph_module & ipGraphModule)
  {
    return std::make_unique<Context>(ipGraphModule);
  }

private:
  llvm::cfg * cfg_;
  ipgraph_module & IPGraphModule_;
  basic_block * LastProcessedBasicBlock;
  std::unordered_map<const rvsdg::output *, const llvm::variable *> ports_;
};

class RvsdgToIpGraphConverter::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::filepath & filename)
      : util::Statistics(Id::RvsdgDestruction, filename)
  {}

  void
  start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const ipgraph_module & im)
  {
    AddMeasurement(Label::NumThreeAddressCodes, llvm::ntacs(im));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

RvsdgToIpGraphConverter::~RvsdgToIpGraphConverter() = default;

RvsdgToIpGraphConverter::RvsdgToIpGraphConverter() = default;

std::unique_ptr<data_node_init>
RvsdgToIpGraphConverter::CreateInitialization(const delta::node & deltaNode)
{
  auto subregion = deltaNode.subregion();

  // add delta dependencies to context
  for (size_t n = 0; n < deltaNode.ninputs(); n++)
  {
    auto v = Context_->variable(deltaNode.input(n)->origin());
    Context_->insert(deltaNode.input(n)->arguments.first(), v);
  }

  if (subregion->nnodes() == 0)
  {
    auto value = Context_->variable(subregion->result(0)->origin());
    return std::make_unique<data_node_init>(value);
  }

  tacsvector_t tacs;
  for (const auto & node : rvsdg::TopDownTraverser(deltaNode.subregion()))
  {
    JLM_ASSERT(node->noutputs() == 1);
    auto output = node->output(0);

    // collect operand variables
    std::vector<const variable *> operands;
    for (size_t n = 0; n < node->ninputs(); n++)
      operands.push_back(Context_->variable(node->input(n)->origin()));

    // convert node to tac
    auto & op = *static_cast<const rvsdg::SimpleOperation *>(&node->GetOperation());
    tacs.push_back(tac::create(op, operands));
    Context_->insert(output, tacs.back()->result(0));
  }

  return std::make_unique<data_node_init>(std::move(tacs));
}

void
RvsdgToIpGraphConverter::convert_region(rvsdg::Region & region)
{
  auto entry = basic_block::create(*Context_->cfg());
  Context_->GetLastProcessedBasicBlock()->add_outedge(entry);
  Context_->SetLastProcessedBasicBlock(entry);

  for (const auto & node : rvsdg::TopDownTraverser(&region))
    ConvertNode(*node);

  auto exit = basic_block::create(*Context_->cfg());
  Context_->GetLastProcessedBasicBlock()->add_outedge(exit);
  Context_->SetLastProcessedBasicBlock(exit);
}

std::unique_ptr<llvm::cfg>
RvsdgToIpGraphConverter::create_cfg(const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(Context_->GetLastProcessedBasicBlock() == nullptr);
  std::unique_ptr<llvm::cfg> cfg(new llvm::cfg(Context_->GetIpGraphModule()));
  auto entry = basic_block::create(*cfg);
  cfg->exit()->divert_inedges(entry);
  Context_->SetLastProcessedBasicBlock(entry);
  Context_->set_cfg(cfg.get());

  /* add arguments */
  for (auto fctarg : lambda.GetFunctionArguments())
  {
    auto name = util::strfmt("_a", fctarg->index(), "_");
    auto argument = llvm::argument::create(
        name,
        fctarg->Type(),
        dynamic_cast<llvm::LlvmLambdaOperation &>(lambda.GetOperation())
            .GetArgumentAttributes(fctarg->index()));
    auto v = cfg->entry()->append_argument(std::move(argument));
    Context_->insert(fctarg, v);
  }

  /* add context variables */
  for (const auto & cv : lambda.GetContextVars())
  {
    auto v = Context_->variable(cv.input->origin());
    Context_->insert(cv.inner, v);
  }

  convert_region(*lambda.subregion());

  /* add results */
  for (auto result : lambda.GetFunctionResults())
    cfg->exit()->append_result(Context_->variable(result->origin()));

  Context_->GetLastProcessedBasicBlock()->add_outedge(cfg->exit());
  Context_->SetLastProcessedBasicBlock(nullptr);
  Context_->set_cfg(nullptr);

  straighten(*cfg);
  JLM_ASSERT(is_closed(*cfg));
  return cfg;
}

void
RvsdgToIpGraphConverter::ConvertSimpleNode(const rvsdg::SimpleNode & simpleNode)
{
  std::vector<const variable *> operands;
  for (size_t n = 0; n < simpleNode.ninputs(); n++)
    operands.push_back(Context_->variable(simpleNode.input(n)->origin()));

  Context_->GetLastProcessedBasicBlock()->append_last(
      tac::create(simpleNode.GetOperation(), operands));

  for (size_t n = 0; n < simpleNode.noutputs(); n++)
    Context_->insert(
        simpleNode.output(n),
        Context_->GetLastProcessedBasicBlock()->last()->result(n));
}

void
RvsdgToIpGraphConverter::ConvertEmptyGammaNode(const rvsdg::GammaNode & gammaNode)
{
  JLM_ASSERT(gammaNode.nsubregions() == 2);
  JLM_ASSERT(gammaNode.subregion(0)->nnodes() == 0 && gammaNode.subregion(1)->nnodes() == 0);

  // both regions are empty, create only select instructions
  const auto predicate = gammaNode.predicate()->origin();
  const auto controlFlowGraph = Context_->cfg();

  const auto basicBlock = basic_block::create(*controlFlowGraph);
  Context_->GetLastProcessedBasicBlock()->add_outedge(basicBlock);

  for (size_t n = 0; n < gammaNode.noutputs(); n++)
  {
    const auto output = gammaNode.output(n);

    const auto argument0 = util::AssertedCast<const rvsdg::RegionArgument>(
        gammaNode.subregion(0)->result(n)->origin());
    const auto argument1 = util::AssertedCast<const rvsdg::RegionArgument>(
        gammaNode.subregion(1)->result(n)->origin());
    const auto output0 = argument0->input()->origin();
    const auto output1 = argument1->input()->origin();

    // both operands are the same, no select is necessary
    if (output0 == output1)
    {
      Context_->insert(output, Context_->variable(output0));
      continue;
    }

    const auto node = rvsdg::output::GetNode(*predicate);
    if (is<rvsdg::match_op>(node))
    {
      const auto matchOperation = util::AssertedCast<const rvsdg::match_op>(&node->GetOperation());
      const auto defaultAlternative = matchOperation->default_alternative();
      const auto condition = Context_->variable(node->input(0)->origin());
      const auto trueAlternative =
          defaultAlternative == 0 ? Context_->variable(output1) : Context_->variable(output0);
      const auto falseAlternative =
          defaultAlternative == 0 ? Context_->variable(output0) : Context_->variable(output1);
      basicBlock->append_last(
          SelectOperation::create(condition, trueAlternative, falseAlternative));
    }
    else
    {
      const auto vo0 = Context_->variable(output0);
      const auto vo1 = Context_->variable(output1);
      basicBlock->append_last(
          ctl2bits_op::create(Context_->variable(predicate), rvsdg::bittype::Create(1)));
      basicBlock->append_last(SelectOperation::create(basicBlock->last()->result(0), vo0, vo1));
    }

    Context_->insert(output, basicBlock->last()->result(0));
  }

  Context_->SetLastProcessedBasicBlock(basicBlock);
}

void
RvsdgToIpGraphConverter::ConvertGammaNode(const rvsdg::GammaNode & gammaNode)
{
  const auto numSubregions = gammaNode.nsubregions();
  const auto predicate = gammaNode.predicate()->origin();
  const auto controlFlowGraph = Context_->cfg();

  if (gammaNode.nsubregions() == 2 && gammaNode.subregion(0)->nnodes() == 0
      && gammaNode.subregion(1)->nnodes() == 0)
    return ConvertEmptyGammaNode(gammaNode);

  const auto entryBlock = basic_block::create(*controlFlowGraph);
  const auto exitBlock = basic_block::create(*controlFlowGraph);
  Context_->GetLastProcessedBasicBlock()->add_outedge(entryBlock);

  // convert gamma regions
  std::vector<cfg_node *> phi_nodes;
  entryBlock->append_last(branch_op::create(numSubregions, Context_->variable(predicate)));
  for (size_t n = 0; n < gammaNode.nsubregions(); n++)
  {
    const auto subregion = gammaNode.subregion(n);

    // add arguments to context
    for (size_t i = 0; i < subregion->narguments(); i++)
    {
      const auto argument = subregion->argument(i);
      Context_->insert(argument, Context_->variable(argument->input()->origin()));
    }

    if (subregion->nnodes() == 0 && numSubregions == 2)
    {
      // subregion is empty
      phi_nodes.push_back(entryBlock);
      entryBlock->add_outedge(exitBlock);
    }
    else
    {
      // convert subregion
      const auto regionEntryBlock = basic_block::create(*controlFlowGraph);
      entryBlock->add_outedge(regionEntryBlock);
      Context_->SetLastProcessedBasicBlock(regionEntryBlock);
      convert_region(*subregion);

      phi_nodes.push_back(Context_->GetLastProcessedBasicBlock());
      Context_->GetLastProcessedBasicBlock()->add_outedge(exitBlock);
    }
  }

  // add phi instructions
  for (size_t n = 0; n < gammaNode.noutputs(); n++)
  {
    const auto output = gammaNode.output(n);

    bool invariant = true;
    const auto matchNode = rvsdg::output::GetNode(*predicate);
    bool select = gammaNode.nsubregions() == 2 && is<rvsdg::match_op>(matchNode);
    std::vector<std::pair<const variable *, cfg_node *>> arguments;
    for (size_t r = 0; r < gammaNode.nsubregions(); r++)
    {
      const auto origin = gammaNode.subregion(r)->result(n)->origin();

      auto v = Context_->variable(origin);
      arguments.push_back(std::make_pair(v, phi_nodes[r]));
      invariant &= (v == Context_->variable(gammaNode.subregion(0)->result(n)->origin()));
      const auto tmpNode = rvsdg::output::GetNode(*origin);
      select &= tmpNode == nullptr && origin->region()->node() == &gammaNode;
    }

    if (invariant)
    {
      // all operands are the same
      Context_->insert(output, arguments[0].first);
      continue;
    }

    if (select)
    {
      // use select instead of phi
      const auto matchOperation =
          util::AssertedCast<const rvsdg::match_op>(&matchNode->GetOperation());
      const auto defaultAlternative = matchOperation->default_alternative();
      const auto condition = Context_->variable(matchNode->input(0)->origin());
      const auto trueAlternative =
          defaultAlternative == 0 ? arguments[1].first : arguments[0].first;
      const auto falseAlternative =
          defaultAlternative == 0 ? arguments[0].first : arguments[1].first;
      entryBlock->append_first(
          SelectOperation::create(condition, trueAlternative, falseAlternative));
      Context_->insert(output, entryBlock->first()->result(0));
      continue;
    }

    // create phi instruction
    exitBlock->append_last(SsaPhiOperation::create(arguments, output->Type()));
    Context_->insert(output, exitBlock->last()->result(0));
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
  const auto entryBlock = basic_block::create(*Context_->cfg());
  preEntryBlock->add_outedge(entryBlock);
  Context_->SetLastProcessedBasicBlock(entryBlock);

  // create SSA phi nodes in entry block and add arguments to context
  std::vector<llvm::tac *> phis;
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    auto variable = Context_->variable(loopVar.input->origin());
    if (RequiresSsaPhiOperation(loopVar, *variable))
    {
      auto phi = entryBlock->append_last(SsaPhiOperation::create({}, loopVar.pre->Type()));
      phis.push_back(phi);
      variable = phi->result(0);
    }
    Context_->insert(loopVar.pre, variable);
  }

  convert_region(*subregion);

  // add phi operands and results to context
  size_t phiIndex = 0;
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    auto entryVariable = Context_->variable(loopVar.input->origin());
    if (RequiresSsaPhiOperation(loopVar, *entryVariable))
    {
      auto resultVariable = Context_->variable(loopVar.post->origin());
      const auto phi = phis[phiIndex++];
      phi->replace(
          SsaPhiOperation(
              { preEntryBlock, Context_->GetLastProcessedBasicBlock() },
              resultVariable->Type()),
          { entryVariable, resultVariable });
      Context_->insert(loopVar.output, resultVariable);
    }
    else
    {
      Context_->insert(loopVar.output, Context_->variable(loopVar.post->origin()));
    }
  }
  JLM_ASSERT(phiIndex == phis.size());

  Context_->GetLastProcessedBasicBlock()->append_last(
      branch_op::create(2, Context_->variable(predicate)));
  const auto exitBlock = basic_block::create(*Context_->cfg());
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

  functionNode->add_cfg(create_cfg(lambdaNode));
  Context_->insert(lambdaNode.output(), variable);
}

void
RvsdgToIpGraphConverter::ConvertPhiNode(const phi::node & phiNode)
{
  const auto subregion = phiNode.subregion();
  auto & ipGraphModule = Context_->GetIpGraphModule();
  auto & ipGraph = ipGraphModule.ipgraph();

  // add dependencies to context
  for (size_t n = 0; n < phiNode.ninputs(); n++)
  {
    const auto variable = Context_->variable(phiNode.input(n)->origin());
    Context_->insert(phiNode.input(n)->arguments.first(), variable);
  }

  // forward declare all functions and global variables
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    const auto node = rvsdg::output::GetNode(*subregion->result(n)->origin());

    if (const auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(node))
    {
      const auto & lambdaOperation =
          dynamic_cast<LlvmLambdaOperation &>(lambdaNode->GetOperation());
      const auto functionNode = function_node::create(
          ipGraph,
          lambdaOperation.name(),
          lambdaOperation.Type(),
          lambdaOperation.linkage(),
          lambdaOperation.attributes());
      Context_->insert(subregion->argument(n), ipGraphModule.create_variable(functionNode));
    }
    else if (const auto deltaNode = dynamic_cast<const delta::node *>(node))
    {
      const auto dataNode = data_node::Create(
          ipGraph,
          deltaNode->name(),
          deltaNode->Type(),
          deltaNode->linkage(),
          deltaNode->Section(),
          deltaNode->constant());
      Context_->insert(subregion->argument(n), ipGraphModule.create_global_value(dataNode));
    }
    else
    {
      JLM_UNREACHABLE(
          util::strfmt("Unhandled node type: ", node->GetOperation().debug_string()).c_str());
    }
  }

  // convert function bodies and global variable initializations
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    const auto result = subregion->result(n);
    const auto node = rvsdg::output::GetNode(*result->origin());

    if (const auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(node))
    {
      const auto variable =
          util::AssertedCast<const fctvariable>(Context_->variable(subregion->argument(n)));
      variable->function()->add_cfg(create_cfg(*lambdaNode));
      Context_->insert(node->output(0), variable);
    }
    else if (const auto deltaNode = dynamic_cast<const delta::node *>(node))
    {
      const auto variable =
          util::AssertedCast<const gblvalue>(Context_->variable(subregion->argument(n)));
      variable->node()->set_initialization(CreateInitialization(*deltaNode));
      Context_->insert(node->output(0), variable);
    }
    else
    {
      JLM_UNREACHABLE(
          util::strfmt("Unhandled node type: ", node->GetOperation().debug_string()).c_str());
    }
  }

  // add functions and globals to context
  JLM_ASSERT(phiNode.noutputs() == subregion->nresults());
  for (size_t n = 0; n < phiNode.noutputs(); n++)
    Context_->insert(phiNode.output(n), Context_->variable(subregion->result(n)->origin()));
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
  Context_->insert(deltaNode.output(), variable);
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
  else if (const auto phiNode = dynamic_cast<const phi::node *>(&node))
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
    JLM_UNREACHABLE(
        util::strfmt("Unhandled node type: ", node.GetOperation().debug_string()).c_str());
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
      Context_->insert(graphImport, variable);
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
      Context_->insert(graphImport, variable);
    }
  }
}

std::unique_ptr<ipgraph_module>
RvsdgToIpGraphConverter::ConvertModule(
    RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFileName());
  statistics->start(rvsdgModule.Rvsdg());

  auto ipGraphModule = ipgraph_module::Create(
      rvsdgModule.SourceFileName(),
      rvsdgModule.TargetTriple(),
      rvsdgModule.DataLayout(),
      std::move(rvsdgModule.ReleaseStructTypeDeclarations()));

  Context_ = Context::Create(*ipGraphModule);
  ConvertImports(rvsdgModule.Rvsdg());
  ConvertNodes(rvsdgModule.Rvsdg());

  statistics->end(*ipGraphModule);
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
