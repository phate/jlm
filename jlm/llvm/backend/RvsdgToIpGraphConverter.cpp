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

#include <deque>

namespace jlm::llvm
{

class RvsdgToIpGraphConverter::Context final
{
public:
  explicit Context(ipgraph_module & ipGraphModule)
      : cfg_(nullptr),
        IPGraphModule_(ipGraphModule),
        lpbb_(nullptr)
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
  lpbb() const noexcept
  {
    return lpbb_;
  }

  void
  set_lpbb(basic_block * lpbb) noexcept
  {
    lpbb_ = lpbb;
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
  basic_block * lpbb_;
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
RvsdgToIpGraphConverter::create_initialization(const delta::node * delta)
{
  auto subregion = delta->subregion();

  // add delta dependencies to context
  for (size_t n = 0; n < delta->ninputs(); n++)
  {
    auto v = Context_->variable(delta->input(n)->origin());
    Context_->insert(delta->input(n)->arguments.first(), v);
  }

  if (subregion->nnodes() == 0)
  {
    auto value = Context_->variable(subregion->result(0)->origin());
    return std::make_unique<data_node_init>(value);
  }

  tacsvector_t tacs;
  for (const auto & node : rvsdg::TopDownTraverser(delta->subregion()))
  {
    JLM_ASSERT(node->noutputs() == 1);
    auto output = node->output(0);

    /* collect operand variables */
    std::vector<const variable *> operands;
    for (size_t n = 0; n < node->ninputs(); n++)
      operands.push_back(Context_->variable(node->input(n)->origin()));

    /* convert node to tac */
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
  Context_->lpbb()->add_outedge(entry);
  Context_->set_lpbb(entry);

  for (const auto & node : rvsdg::TopDownTraverser(&region))
    convert_node(*node);

  auto exit = basic_block::create(*Context_->cfg());
  Context_->lpbb()->add_outedge(exit);
  Context_->set_lpbb(exit);
}

std::unique_ptr<llvm::cfg>
RvsdgToIpGraphConverter::create_cfg(const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(Context_->lpbb() == nullptr);
  std::unique_ptr<llvm::cfg> cfg(new llvm::cfg(Context_->GetIpGraphModule()));
  auto entry = basic_block::create(*cfg);
  cfg->exit()->divert_inedges(entry);
  Context_->set_lpbb(entry);
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

  Context_->lpbb()->add_outedge(cfg->exit());
  Context_->set_lpbb(nullptr);
  Context_->set_cfg(nullptr);

  straighten(*cfg);
  JLM_ASSERT(is_closed(*cfg));
  return cfg;
}

void
RvsdgToIpGraphConverter::convert_simple_node(const rvsdg::Node & node)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::SimpleOperation *>(&node.GetOperation()));

  std::vector<const variable *> operands;
  for (size_t n = 0; n < node.ninputs(); n++)
    operands.push_back(Context_->variable(node.input(n)->origin()));

  auto & op = *static_cast<const rvsdg::SimpleOperation *>(&node.GetOperation());
  Context_->lpbb()->append_last(tac::create(op, operands));

  for (size_t n = 0; n < node.noutputs(); n++)
    Context_->insert(node.output(n), Context_->lpbb()->last()->result(n));
}

void
RvsdgToIpGraphConverter::convert_empty_gamma_node(const rvsdg::GammaNode * gamma)
{
  JLM_ASSERT(gamma->nsubregions() == 2);
  JLM_ASSERT(gamma->subregion(0)->nnodes() == 0 && gamma->subregion(1)->nnodes() == 0);

  /* both regions are empty, create only select instructions */

  auto predicate = gamma->predicate()->origin();
  auto cfg = Context_->cfg();

  auto bb = basic_block::create(*cfg);
  Context_->lpbb()->add_outedge(bb);

  for (size_t n = 0; n < gamma->noutputs(); n++)
  {
    auto output = gamma->output(n);

    auto a0 = static_cast<const rvsdg::RegionArgument *>(gamma->subregion(0)->result(n)->origin());
    auto a1 = static_cast<const rvsdg::RegionArgument *>(gamma->subregion(1)->result(n)->origin());
    auto o0 = a0->input()->origin();
    auto o1 = a1->input()->origin();

    /* both operands are the same, no select is necessary */
    if (o0 == o1)
    {
      Context_->insert(output, Context_->variable(o0));
      continue;
    }

    auto matchnode = rvsdg::output::GetNode(*predicate);
    if (is<rvsdg::match_op>(matchnode))
    {
      auto matchop = static_cast<const rvsdg::match_op *>(&matchnode->GetOperation());
      auto d = matchop->default_alternative();
      auto c = Context_->variable(matchnode->input(0)->origin());
      auto t = d == 0 ? Context_->variable(o1) : Context_->variable(o0);
      auto f = d == 0 ? Context_->variable(o0) : Context_->variable(o1);
      bb->append_last(SelectOperation::create(c, t, f));
    }
    else
    {
      auto vo0 = Context_->variable(o0);
      auto vo1 = Context_->variable(o1);
      bb->append_last(
          ctl2bits_op::create(Context_->variable(predicate), rvsdg::bittype::Create(1)));
      bb->append_last(SelectOperation::create(bb->last()->result(0), vo0, vo1));
    }

    Context_->insert(output, bb->last()->result(0));
  }

  Context_->set_lpbb(bb);
}

void
RvsdgToIpGraphConverter::convert_gamma_node(const rvsdg::Node & node)
{
  JLM_ASSERT(is<rvsdg::GammaOperation>(&node));
  auto gamma = static_cast<const rvsdg::GammaNode *>(&node);
  auto nalternatives = gamma->nsubregions();
  auto predicate = gamma->predicate()->origin();
  auto cfg = Context_->cfg();

  if (gamma->nsubregions() == 2 && gamma->subregion(0)->nnodes() == 0
      && gamma->subregion(1)->nnodes() == 0)
    return convert_empty_gamma_node(gamma);

  auto entry = basic_block::create(*cfg);
  auto exit = basic_block::create(*cfg);
  Context_->lpbb()->add_outedge(entry);

  /* convert gamma regions */
  std::vector<cfg_node *> phi_nodes;
  entry->append_last(branch_op::create(nalternatives, Context_->variable(predicate)));
  for (size_t n = 0; n < gamma->nsubregions(); n++)
  {
    auto subregion = gamma->subregion(n);

    /* add arguments to context */
    for (size_t i = 0; i < subregion->narguments(); i++)
    {
      auto argument = subregion->argument(i);
      Context_->insert(argument, Context_->variable(argument->input()->origin()));
    }

    if (subregion->nnodes() == 0 && nalternatives == 2)
    {
      /* subregin is empty */
      phi_nodes.push_back(entry);
      entry->add_outedge(exit);
    }
    else
    {
      /* convert subregion */
      auto region_entry = basic_block::create(*cfg);
      entry->add_outedge(region_entry);
      Context_->set_lpbb(region_entry);
      convert_region(*subregion);

      phi_nodes.push_back(Context_->lpbb());
      Context_->lpbb()->add_outedge(exit);
    }
  }

  /* add phi instructions */
  for (size_t n = 0; n < gamma->noutputs(); n++)
  {
    auto output = gamma->output(n);

    bool invariant = true;
    auto matchnode = rvsdg::output::GetNode(*predicate);
    bool select = (gamma->nsubregions() == 2) && is<rvsdg::match_op>(matchnode);
    std::vector<std::pair<const variable *, cfg_node *>> arguments;
    for (size_t r = 0; r < gamma->nsubregions(); r++)
    {
      auto origin = gamma->subregion(r)->result(n)->origin();

      auto v = Context_->variable(origin);
      arguments.push_back(std::make_pair(v, phi_nodes[r]));
      invariant &= (v == Context_->variable(gamma->subregion(0)->result(n)->origin()));
      auto tmp = rvsdg::output::GetNode(*origin);
      select &= (tmp == nullptr && origin->region()->node() == &node);
    }

    if (invariant)
    {
      /* all operands are the same */
      Context_->insert(output, arguments[0].first);
      continue;
    }

    if (select)
    {
      /* use select instead of phi */
      auto matchnode = rvsdg::output::GetNode(*predicate);
      auto matchop = static_cast<const rvsdg::match_op *>(&matchnode->GetOperation());
      auto d = matchop->default_alternative();
      auto c = Context_->variable(matchnode->input(0)->origin());
      auto t = d == 0 ? arguments[1].first : arguments[0].first;
      auto f = d == 0 ? arguments[0].first : arguments[1].first;
      entry->append_first(SelectOperation::create(c, t, f));
      Context_->insert(output, entry->first()->result(0));
      continue;
    }

    /* create phi instruction */
    exit->append_last(SsaPhiOperation::create(arguments, output->Type()));
    Context_->insert(output, exit->last()->result(0));
  }

  Context_->set_lpbb(exit);
}

bool
RvsdgToIpGraphConverter::phi_needed(const rvsdg::input * i, const llvm::variable * v)
{
  auto node = rvsdg::input::GetNode(*i);
  JLM_ASSERT(is<rvsdg::ThetaOperation>(node));
  auto theta = static_cast<const rvsdg::StructuralNode *>(node);
  auto input = static_cast<const rvsdg::StructuralInput *>(i);
  auto output = theta->output(input->index());

  // FIXME: solely decide on the input instead of using the variable
  if (is<gblvariable>(v))
    return false;

  if (output->results.first()->origin() == input->arguments.first())
    return false;

  if (input->arguments.first()->nusers() == 0)
    return false;

  return true;
}

void
RvsdgToIpGraphConverter::convert_theta_node(const rvsdg::Node & node)
{
  JLM_ASSERT(is<rvsdg::ThetaOperation>(&node));
  auto subregion = static_cast<const rvsdg::StructuralNode *>(&node)->subregion(0);
  auto predicate = subregion->result(0)->origin();

  auto pre_entry = Context_->lpbb();
  auto entry = basic_block::create(*Context_->cfg());
  pre_entry->add_outedge(entry);
  Context_->set_lpbb(entry);

  // create phi nodes and add arguments to context
  std::deque<llvm::tac *> phis;
  for (size_t n = 0; n < subregion->narguments(); n++)
  {
    auto argument = subregion->argument(n);
    auto v = Context_->variable(argument->input()->origin());
    if (phi_needed(argument->input(), v))
    {
      auto phi = entry->append_last(SsaPhiOperation::create({}, argument->Type()));
      phis.push_back(phi);
      v = phi->result(0);
    }
    Context_->insert(argument, v);
  }

  convert_region(*subregion);

  // add phi operands and results to context
  for (size_t n = 1; n < subregion->nresults(); n++)
  {
    auto result = subregion->result(n);
    auto ve = Context_->variable(node.input(n - 1)->origin());
    if (!phi_needed(node.input(n - 1), ve))
    {
      Context_->insert(result->output(), Context_->variable(result->origin()));
      continue;
    }

    auto vr = Context_->variable(result->origin());
    auto phi = phis.front();
    phis.pop_front();
    phi->replace(SsaPhiOperation({ pre_entry, Context_->lpbb() }, vr->Type()), { ve, vr });
    Context_->insert(result->output(), vr);
  }
  JLM_ASSERT(phis.empty());

  Context_->lpbb()->append_last(branch_op::create(2, Context_->variable(predicate)));
  auto exit = basic_block::create(*Context_->cfg());
  Context_->lpbb()->add_outedge(exit);
  Context_->lpbb()->add_outedge(entry);
  Context_->set_lpbb(exit);
}

void
RvsdgToIpGraphConverter::convert_lambda_node(const rvsdg::Node & node)
{
  JLM_ASSERT(is<llvm::LlvmLambdaOperation>(&node));
  auto lambda = static_cast<const rvsdg::LambdaNode *>(&node);
  auto & module = Context_->GetIpGraphModule();
  auto & clg = module.ipgraph();

  const auto & op = dynamic_cast<llvm::LlvmLambdaOperation &>(lambda->GetOperation());
  auto f = function_node::create(clg, op.name(), op.Type(), op.linkage(), op.attributes());
  auto v = module.create_variable(f);

  f->add_cfg(create_cfg(*lambda));
  Context_->insert(node.output(0), v);
}

void
RvsdgToIpGraphConverter::convert_phi_node(const rvsdg::Node & node)
{
  JLM_ASSERT(rvsdg::is<phi::operation>(&node));
  auto phi = static_cast<const rvsdg::StructuralNode *>(&node);
  auto subregion = phi->subregion(0);
  auto & module = Context_->GetIpGraphModule();
  auto & ipg = module.ipgraph();

  /* add dependencies to context */
  for (size_t n = 0; n < phi->ninputs(); n++)
  {
    auto v = Context_->variable(phi->input(n)->origin());
    Context_->insert(phi->input(n)->arguments.first(), v);
  }

  /* forward declare all functions and globals */
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    auto node = rvsdg::output::GetNode(*subregion->result(n)->origin());

    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(node))
    {
      const auto & op = dynamic_cast<llvm::LlvmLambdaOperation &>(lambda->GetOperation());
      auto f = function_node::create(ipg, op.name(), op.Type(), op.linkage(), op.attributes());
      Context_->insert(subregion->argument(n), module.create_variable(f));
    }
    else
    {
      JLM_ASSERT(is<delta::operation>(node));
      auto d = static_cast<const delta::node *>(node);
      auto data =
          data_node::Create(ipg, d->name(), d->Type(), d->linkage(), d->Section(), d->constant());
      Context_->insert(subregion->argument(n), module.create_global_value(data));
    }
  }

  /* convert function bodies and global initializations */
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    auto result = subregion->result(n);
    auto node = rvsdg::output::GetNode(*result->origin());

    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(node))
    {
      auto v = static_cast<const fctvariable *>(Context_->variable(subregion->argument(n)));
      v->function()->add_cfg(create_cfg(*lambda));
      Context_->insert(node->output(0), v);
    }
    else
    {
      JLM_ASSERT(is<delta::operation>(node));
      auto delta = static_cast<const delta::node *>(node);
      auto v = static_cast<const gblvalue *>(Context_->variable(subregion->argument(n)));

      v->node()->set_initialization(create_initialization(delta));
      Context_->insert(node->output(0), v);
    }
  }

  /* add functions and globals to context */
  JLM_ASSERT(node.noutputs() == subregion->nresults());
  for (size_t n = 0; n < node.noutputs(); n++)
    Context_->insert(node.output(n), Context_->variable(subregion->result(n)->origin()));
}

void
RvsdgToIpGraphConverter::convert_delta_node(const rvsdg::Node & node)
{
  JLM_ASSERT(is<delta::operation>(&node));
  auto delta = static_cast<const delta::node *>(&node);
  auto & m = Context_->GetIpGraphModule();

  auto dnode = data_node::Create(
      m.ipgraph(),
      delta->name(),
      delta->Type(),
      delta->linkage(),
      delta->Section(),
      delta->constant());
  dnode->set_initialization(create_initialization(delta));
  auto v = m.create_global_value(dnode);
  Context_->insert(delta->output(), v);
}

void
RvsdgToIpGraphConverter::convert_node(const rvsdg::Node & node)
{
  if (const auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(&node))
  {
    convert_lambda_node(*lambdaNode);
  }
  else if (const auto gammaNode = dynamic_cast<const rvsdg::GammaNode *>(&node))
  {
    convert_gamma_node(*gammaNode);
  }
  else if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(&node))
  {
    convert_theta_node(*thetaNode);
  }
  else if (const auto phiNode = dynamic_cast<const phi::node *>(&node))
  {
    convert_phi_node(*phiNode);
  }
  else if (const auto deltaNode = dynamic_cast<const delta::node *>(&node))
  {
    convert_delta_node(*deltaNode);
  }
  else if (dynamic_cast<const rvsdg::SimpleNode *>(&node))
  {
    convert_simple_node(node);
  }
  else
  {
    JLM_UNREACHABLE(
        util::strfmt("Unhandled node type: ", node.GetOperation().debug_string()).c_str());
  }
}

void
RvsdgToIpGraphConverter::convert_nodes(const rvsdg::Graph & graph)
{
  for (const auto & node : rvsdg::TopDownTraverser(&graph.GetRootRegion()))
    convert_node(*node);
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
  convert_nodes(rvsdgModule.Rvsdg());

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
