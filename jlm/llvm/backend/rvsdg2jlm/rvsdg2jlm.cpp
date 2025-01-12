/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/rvsdg2jlm/context.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
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
#include <typeindex>

namespace jlm::llvm
{

class rvsdg_destruction_stat final : public util::Statistics
{
public:
  ~rvsdg_destruction_stat() override = default;

  explicit rvsdg_destruction_stat(const util::filepath & filename)
      : Statistics(Statistics::Id::RvsdgDestruction, filename)
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

  static std::unique_ptr<rvsdg_destruction_stat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<rvsdg_destruction_stat>(sourceFile);
  }
};

namespace rvsdg2jlm
{

static std::shared_ptr<const rvsdg::FunctionType>
is_function_import(const llvm::GraphImport * graphImport)
{
  return std::dynamic_pointer_cast<const rvsdg::FunctionType>(graphImport->ValueType());
}

static std::unique_ptr<data_node_init>
create_initialization(const delta::node * delta, context & ctx)
{
  auto subregion = delta->subregion();

  /* add delta dependencies to context */
  for (size_t n = 0; n < delta->ninputs(); n++)
  {
    auto v = ctx.variable(delta->input(n)->origin());
    ctx.insert(delta->input(n)->arguments.first(), v);
  }

  if (subregion->nnodes() == 0)
  {
    auto value = ctx.variable(subregion->result(0)->origin());
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
      operands.push_back(ctx.variable(node->input(n)->origin()));

    /* convert node to tac */
    auto & op = *static_cast<const rvsdg::SimpleOperation *>(&node->GetOperation());
    tacs.push_back(tac::create(op, operands));
    ctx.insert(output, tacs.back()->result(0));
  }

  return std::make_unique<data_node_init>(std::move(tacs));
}

static void
convert_node(const rvsdg::Node & node, context & ctx);

static inline void
convert_region(rvsdg::Region & region, context & ctx)
{
  auto entry = basic_block::create(*ctx.cfg());
  ctx.lpbb()->add_outedge(entry);
  ctx.set_lpbb(entry);

  for (const auto & node : rvsdg::TopDownTraverser(&region))
    convert_node(*node, ctx);

  auto exit = basic_block::create(*ctx.cfg());
  ctx.lpbb()->add_outedge(exit);
  ctx.set_lpbb(exit);
}

static inline std::unique_ptr<llvm::cfg>
create_cfg(const lambda::node & lambda, context & ctx)
{
  JLM_ASSERT(ctx.lpbb() == nullptr);
  std::unique_ptr<llvm::cfg> cfg(new llvm::cfg(ctx.module()));
  auto entry = basic_block::create(*cfg);
  cfg->exit()->divert_inedges(entry);
  ctx.set_lpbb(entry);
  ctx.set_cfg(cfg.get());

  /* add arguments */
  for (auto fctarg : lambda.GetFunctionArguments())
  {
    auto name = util::strfmt("_a", fctarg->index(), "_");
    auto argument = llvm::argument::create(
        name,
        fctarg->Type(),
        lambda.GetOperation().GetArgumentAttributes(fctarg->index()));
    auto v = cfg->entry()->append_argument(std::move(argument));
    ctx.insert(fctarg, v);
  }

  /* add context variables */
  for (const auto & cv : lambda.GetContextVars())
  {
    auto v = ctx.variable(cv.input->origin());
    ctx.insert(cv.inner, v);
  }

  convert_region(*lambda.subregion(), ctx);

  /* add results */
  for (auto result : lambda.GetFunctionResults())
    cfg->exit()->append_result(ctx.variable(result->origin()));

  ctx.lpbb()->add_outedge(cfg->exit());
  ctx.set_lpbb(nullptr);
  ctx.set_cfg(nullptr);

  straighten(*cfg);
  JLM_ASSERT(is_closed(*cfg));
  return cfg;
}

static inline void
convert_simple_node(const rvsdg::Node & node, context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::SimpleOperation *>(&node.GetOperation()));

  std::vector<const variable *> operands;
  for (size_t n = 0; n < node.ninputs(); n++)
    operands.push_back(ctx.variable(node.input(n)->origin()));

  auto & op = *static_cast<const rvsdg::SimpleOperation *>(&node.GetOperation());
  ctx.lpbb()->append_last(tac::create(op, operands));

  for (size_t n = 0; n < node.noutputs(); n++)
    ctx.insert(node.output(n), ctx.lpbb()->last()->result(n));
}

static void
convert_empty_gamma_node(const rvsdg::GammaNode * gamma, context & ctx)
{
  JLM_ASSERT(gamma->nsubregions() == 2);
  JLM_ASSERT(gamma->subregion(0)->nnodes() == 0 && gamma->subregion(1)->nnodes() == 0);

  /* both regions are empty, create only select instructions */

  auto predicate = gamma->predicate()->origin();
  auto cfg = ctx.cfg();

  auto bb = basic_block::create(*cfg);
  ctx.lpbb()->add_outedge(bb);

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
      ctx.insert(output, ctx.variable(o0));
      continue;
    }

    auto matchnode = rvsdg::output::GetNode(*predicate);
    if (is<rvsdg::match_op>(matchnode))
    {
      auto matchop = static_cast<const rvsdg::match_op *>(&matchnode->GetOperation());
      auto d = matchop->default_alternative();
      auto c = ctx.variable(matchnode->input(0)->origin());
      auto t = d == 0 ? ctx.variable(o1) : ctx.variable(o0);
      auto f = d == 0 ? ctx.variable(o0) : ctx.variable(o1);
      bb->append_last(select_op::create(c, t, f));
    }
    else
    {
      auto vo0 = ctx.variable(o0);
      auto vo1 = ctx.variable(o1);
      bb->append_last(ctl2bits_op::create(ctx.variable(predicate), rvsdg::bittype::Create(1)));
      bb->append_last(select_op::create(bb->last()->result(0), vo0, vo1));
    }

    ctx.insert(output, bb->last()->result(0));
  }

  ctx.set_lpbb(bb);
}

static inline void
convert_gamma_node(const rvsdg::Node & node, context & ctx)
{
  JLM_ASSERT(is<rvsdg::GammaOperation>(&node));
  auto gamma = static_cast<const rvsdg::GammaNode *>(&node);
  auto nalternatives = gamma->nsubregions();
  auto predicate = gamma->predicate()->origin();
  auto cfg = ctx.cfg();

  if (gamma->nsubregions() == 2 && gamma->subregion(0)->nnodes() == 0
      && gamma->subregion(1)->nnodes() == 0)
    return convert_empty_gamma_node(gamma, ctx);

  auto entry = basic_block::create(*cfg);
  auto exit = basic_block::create(*cfg);
  ctx.lpbb()->add_outedge(entry);

  /* convert gamma regions */
  std::vector<cfg_node *> phi_nodes;
  entry->append_last(branch_op::create(nalternatives, ctx.variable(predicate)));
  for (size_t n = 0; n < gamma->nsubregions(); n++)
  {
    auto subregion = gamma->subregion(n);

    /* add arguments to context */
    for (size_t i = 0; i < subregion->narguments(); i++)
    {
      auto argument = subregion->argument(i);
      ctx.insert(argument, ctx.variable(argument->input()->origin()));
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
      ctx.set_lpbb(region_entry);
      convert_region(*subregion, ctx);

      phi_nodes.push_back(ctx.lpbb());
      ctx.lpbb()->add_outedge(exit);
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

      auto v = ctx.variable(origin);
      arguments.push_back(std::make_pair(v, phi_nodes[r]));
      invariant &= (v == ctx.variable(gamma->subregion(0)->result(n)->origin()));
      auto tmp = rvsdg::output::GetNode(*origin);
      select &= (tmp == nullptr && origin->region()->node() == &node);
    }

    if (invariant)
    {
      /* all operands are the same */
      ctx.insert(output, arguments[0].first);
      continue;
    }

    if (select)
    {
      /* use select instead of phi */
      auto matchnode = rvsdg::output::GetNode(*predicate);
      auto matchop = static_cast<const rvsdg::match_op *>(&matchnode->GetOperation());
      auto d = matchop->default_alternative();
      auto c = ctx.variable(matchnode->input(0)->origin());
      auto t = d == 0 ? arguments[1].first : arguments[0].first;
      auto f = d == 0 ? arguments[0].first : arguments[1].first;
      entry->append_first(select_op::create(c, t, f));
      ctx.insert(output, entry->first()->result(0));
      continue;
    }

    /* create phi instruction */
    exit->append_last(phi_op::create(arguments, output->Type()));
    ctx.insert(output, exit->last()->result(0));
  }

  ctx.set_lpbb(exit);
}

static inline bool
phi_needed(const rvsdg::input * i, const llvm::variable * v)
{
  auto node = rvsdg::input::GetNode(*i);
  JLM_ASSERT(is<rvsdg::ThetaOperation>(node));
  auto theta = static_cast<const rvsdg::StructuralNode *>(node);
  auto input = static_cast<const rvsdg::StructuralInput *>(i);
  auto output = theta->output(input->index());

  /* FIXME: solely decide on the input instead of using the variable */
  if (is<gblvariable>(v))
    return false;

  if (output->results.first()->origin() == input->arguments.first())
    return false;

  if (input->arguments.first()->nusers() == 0)
    return false;

  return true;
}

static inline void
convert_theta_node(const rvsdg::Node & node, context & ctx)
{
  JLM_ASSERT(is<rvsdg::ThetaOperation>(&node));
  auto subregion = static_cast<const rvsdg::StructuralNode *>(&node)->subregion(0);
  auto predicate = subregion->result(0)->origin();

  auto pre_entry = ctx.lpbb();
  auto entry = basic_block::create(*ctx.cfg());
  pre_entry->add_outedge(entry);
  ctx.set_lpbb(entry);

  /* create phi nodes and add arguments to context */
  std::deque<llvm::tac *> phis;
  for (size_t n = 0; n < subregion->narguments(); n++)
  {
    auto argument = subregion->argument(n);
    auto v = ctx.variable(argument->input()->origin());
    if (phi_needed(argument->input(), v))
    {
      auto phi = entry->append_last(phi_op::create({}, argument->Type()));
      phis.push_back(phi);
      v = phi->result(0);
    }
    ctx.insert(argument, v);
  }

  convert_region(*subregion, ctx);

  /* add phi operands and results to context */
  for (size_t n = 1; n < subregion->nresults(); n++)
  {
    auto result = subregion->result(n);
    auto ve = ctx.variable(node.input(n - 1)->origin());
    if (!phi_needed(node.input(n - 1), ve))
    {
      ctx.insert(result->output(), ctx.variable(result->origin()));
      continue;
    }

    auto vr = ctx.variable(result->origin());
    auto phi = phis.front();
    phis.pop_front();
    phi->replace(phi_op({ pre_entry, ctx.lpbb() }, vr->Type()), { ve, vr });
    ctx.insert(result->output(), vr);
  }
  JLM_ASSERT(phis.empty());

  ctx.lpbb()->append_last(branch_op::create(2, ctx.variable(predicate)));
  auto exit = basic_block::create(*ctx.cfg());
  ctx.lpbb()->add_outedge(exit);
  ctx.lpbb()->add_outedge(entry);
  ctx.set_lpbb(exit);
}

static inline void
convert_lambda_node(const rvsdg::Node & node, context & ctx)
{
  JLM_ASSERT(is<lambda::operation>(&node));
  auto lambda = static_cast<const lambda::node *>(&node);
  auto & module = ctx.module();
  auto & clg = module.ipgraph();

  const auto & op = lambda->GetOperation();
  auto f = function_node::create(clg, op.name(), op.Type(), op.linkage(), op.attributes());
  auto v = module.create_variable(f);

  f->add_cfg(create_cfg(*lambda, ctx));
  ctx.insert(node.output(0), v);
}

static inline void
convert_phi_node(const rvsdg::Node & node, context & ctx)
{
  JLM_ASSERT(rvsdg::is<phi::operation>(&node));
  auto phi = static_cast<const rvsdg::StructuralNode *>(&node);
  auto subregion = phi->subregion(0);
  auto & module = ctx.module();
  auto & ipg = module.ipgraph();

  /* add dependencies to context */
  for (size_t n = 0; n < phi->ninputs(); n++)
  {
    auto v = ctx.variable(phi->input(n)->origin());
    ctx.insert(phi->input(n)->arguments.first(), v);
  }

  /* forward declare all functions and globals */
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    auto node = rvsdg::output::GetNode(*subregion->result(n)->origin());

    if (auto lambda = dynamic_cast<const lambda::node *>(node))
    {
      const auto & op = lambda->GetOperation();
      auto f = function_node::create(ipg, op.name(), op.Type(), op.linkage(), op.attributes());
      ctx.insert(subregion->argument(n), module.create_variable(f));
    }
    else
    {
      JLM_ASSERT(is<delta::operation>(node));
      auto d = static_cast<const delta::node *>(node);
      auto data =
          data_node::Create(ipg, d->name(), d->Type(), d->linkage(), d->Section(), d->constant());
      ctx.insert(subregion->argument(n), module.create_global_value(data));
    }
  }

  /* convert function bodies and global initializations */
  for (size_t n = 0; n < subregion->nresults(); n++)
  {
    JLM_ASSERT(subregion->argument(n)->input() == nullptr);
    auto result = subregion->result(n);
    auto node = rvsdg::output::GetNode(*result->origin());

    if (auto lambda = dynamic_cast<const lambda::node *>(node))
    {
      auto v = static_cast<const fctvariable *>(ctx.variable(subregion->argument(n)));
      v->function()->add_cfg(create_cfg(*lambda, ctx));
      ctx.insert(node->output(0), v);
    }
    else
    {
      JLM_ASSERT(is<delta::operation>(node));
      auto delta = static_cast<const delta::node *>(node);
      auto v = static_cast<const gblvalue *>(ctx.variable(subregion->argument(n)));

      v->node()->set_initialization(create_initialization(delta, ctx));
      ctx.insert(node->output(0), v);
    }
  }

  /* add functions and globals to context */
  JLM_ASSERT(node.noutputs() == subregion->nresults());
  for (size_t n = 0; n < node.noutputs(); n++)
    ctx.insert(node.output(n), ctx.variable(subregion->result(n)->origin()));
}

static inline void
convert_delta_node(const rvsdg::Node & node, context & ctx)
{
  JLM_ASSERT(is<delta::operation>(&node));
  auto delta = static_cast<const delta::node *>(&node);
  auto & m = ctx.module();

  auto dnode = data_node::Create(
      m.ipgraph(),
      delta->name(),
      delta->Type(),
      delta->linkage(),
      delta->Section(),
      delta->constant());
  dnode->set_initialization(create_initialization(delta, ctx));
  auto v = m.create_global_value(dnode);
  ctx.insert(delta->output(), v);
}

static inline void
convert_node(const rvsdg::Node & node, context & ctx)
{
  static std::
      unordered_map<std::type_index, std::function<void(const rvsdg::Node & node, context & ctx)>>
          map({ { typeid(lambda::operation), convert_lambda_node },
                { std::type_index(typeid(rvsdg::GammaOperation)), convert_gamma_node },
                { std::type_index(typeid(rvsdg::ThetaOperation)), convert_theta_node },
                { typeid(phi::operation), convert_phi_node },
                { typeid(delta::operation), convert_delta_node } });

  if (dynamic_cast<const rvsdg::SimpleOperation *>(&node.GetOperation()))
  {
    convert_simple_node(node, ctx);
    return;
  }

  auto & op = node.GetOperation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](node, ctx);
}

static void
convert_nodes(const rvsdg::Graph & graph, context & ctx)
{
  for (const auto & node : rvsdg::TopDownTraverser(&graph.GetRootRegion()))
    convert_node(*node, ctx);
}

static void
convert_imports(const rvsdg::Graph & graph, ipgraph_module & im, context & ctx)
{
  auto & ipg = im.ipgraph();

  for (size_t n = 0; n < graph.GetRootRegion().narguments(); n++)
  {
    auto graphImport = util::AssertedCast<GraphImport>(graph.GetRootRegion().argument(n));
    if (auto ftype = is_function_import(graphImport))
    {
      auto f = function_node::create(ipg, graphImport->Name(), ftype, graphImport->Linkage());
      auto v = im.create_variable(f);
      ctx.insert(graphImport, v);
    }
    else
    {
      auto dnode = data_node::Create(
          ipg,
          graphImport->Name(),
          graphImport->ValueType(),
          graphImport->Linkage(),
          "",
          false);
      auto v = im.create_global_value(dnode);
      ctx.insert(graphImport, v);
    }
  }
}

static std::unique_ptr<ipgraph_module>
convert_rvsdg(RvsdgModule & rm)
{
  auto im = ipgraph_module::Create(
      rm.SourceFileName(),
      rm.TargetTriple(),
      rm.DataLayout(),
      std::move(rm.ReleaseStructTypeDeclarations()));

  context ctx(*im);
  convert_imports(rm.Rvsdg(), *im, ctx);
  convert_nodes(rm.Rvsdg(), ctx);

  return im;
}

std::unique_ptr<ipgraph_module>
rvsdg2jlm(RvsdgModule & rm, jlm::util::StatisticsCollector & statisticsCollector)
{
  auto statistics = rvsdg_destruction_stat::Create(rm.SourceFileName());

  statistics->start(rm.Rvsdg());
  auto im = convert_rvsdg(rm);
  statistics->end(*im);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return im;
}

}
}
