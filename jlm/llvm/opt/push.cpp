/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <deque>

namespace jlm::llvm
{

class pushstat final : public util::Statistics
{
public:
  ~pushstat() override = default;

  explicit pushstat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::PushNodes, sourceFile)
  {}

  void
  start(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsBefore, jlm::rvsdg::ninputs(graph.root()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsAfter, jlm::rvsdg::ninputs(graph.root()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<pushstat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<pushstat>(sourceFile);
  }
};

class worklist
{
public:
  inline void
  push_back(jlm::rvsdg::node * node) noexcept
  {
    if (set_.find(node) != set_.end())
      return;

    queue_.push_back(node);
    set_.insert(node);
  }

  inline jlm::rvsdg::node *
  pop_front() noexcept
  {
    JLM_ASSERT(!empty());
    auto node = queue_.front();
    queue_.pop_front();
    set_.erase(node);
    return node;
  }

  inline bool
  empty() const noexcept
  {
    JLM_ASSERT(queue_.size() == set_.size());
    return queue_.empty();
  }

private:
  std::deque<jlm::rvsdg::node *> queue_;
  std::unordered_set<jlm::rvsdg::node *> set_;
};

static bool
has_side_effects(const jlm::rvsdg::node * node)
{
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    if (dynamic_cast<const jlm::rvsdg::statetype *>(&node->output(n)->type()))
      return true;
  }

  return false;
}

static std::vector<jlm::rvsdg::argument *>
copy_from_gamma(jlm::rvsdg::node * node, size_t r)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::gamma_op>(node->region()->node()));
  JLM_ASSERT(node->depth() == 0);

  auto target = node->region()->node()->region();
  auto gamma = static_cast<jlm::rvsdg::gamma_node *>(node->region()->node());

  std::vector<jlm::rvsdg::output *> operands;
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::argument *>(node->input(n)->origin()));
    auto argument = static_cast<const jlm::rvsdg::argument *>(node->input(n)->origin());
    operands.push_back(argument->input()->origin());
  }

  std::vector<jlm::rvsdg::argument *> arguments;
  auto copy = node->copy(target, operands);
  for (size_t n = 0; n < copy->noutputs(); n++)
  {
    auto ev = gamma->add_entryvar(copy->output(n));
    node->output(n)->divert_users(ev->argument(r));
    arguments.push_back(ev->argument(r));
  }

  return arguments;
}

static std::vector<jlm::rvsdg::argument *>
copy_from_theta(jlm::rvsdg::node * node)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(node->region()->node()));
  JLM_ASSERT(node->depth() == 0);

  auto target = node->region()->node()->region();
  auto theta = static_cast<jlm::rvsdg::theta_node *>(node->region()->node());

  std::vector<jlm::rvsdg::output *> operands;
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::argument *>(node->input(n)->origin()));
    auto argument = static_cast<const jlm::rvsdg::argument *>(node->input(n)->origin());
    operands.push_back(argument->input()->origin());
  }

  std::vector<jlm::rvsdg::argument *> arguments;
  auto copy = node->copy(target, operands);
  for (size_t n = 0; n < copy->noutputs(); n++)
  {
    auto lv = theta->add_loopvar(copy->output(n));
    node->output(n)->divert_users(lv->argument());
    arguments.push_back(lv->argument());
  }

  return arguments;
}

static bool
is_gamma_top_pushable(const jlm::rvsdg::node * node)
{
  /*
    FIXME: This is techically not fully correct. It is
    only possible to push a load out of a gamma node, if
    it is guaranteed to load from a valid address.
  */
  if (is<LoadNonVolatileOperation>(node))
    return true;

  return !has_side_effects(node);
}

void
push(jlm::rvsdg::gamma_node * gamma)
{
  for (size_t r = 0; r < gamma->nsubregions(); r++)
  {
    auto region = gamma->subregion(r);

    /* push out all nullary nodes */
    for (auto & node : region->top_nodes)
    {
      if (!has_side_effects(&node))
        copy_from_gamma(&node, r);
    }

    /* initialize worklist */
    worklist wl;
    for (size_t n = 0; n < region->narguments(); n++)
    {
      auto argument = region->argument(n);
      for (const auto & user : *argument)
      {
        auto tmp = input_node(user);
        if (tmp && tmp->depth() == 0)
          wl.push_back(tmp);
      }
    }

    /* process worklist */
    while (!wl.empty())
    {
      auto node = wl.pop_front();

      if (!is_gamma_top_pushable(node))
        continue;

      auto arguments = copy_from_gamma(node, r);

      /* add consumers to worklist */
      for (const auto & argument : arguments)
      {
        for (const auto & user : *argument)
        {
          auto tmp = input_node(user);
          if (tmp && tmp->depth() == 0)
            wl.push_back(tmp);
        }
      }
    }
  }
}

static bool
is_theta_invariant(
    const jlm::rvsdg::node * node,
    const std::unordered_set<jlm::rvsdg::argument *> & invariants)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(node->region()->node()));
  JLM_ASSERT(node->depth() == 0);

  for (size_t n = 0; n < node->ninputs(); n++)
  {
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::argument *>(node->input(n)->origin()));
    auto argument = static_cast<jlm::rvsdg::argument *>(node->input(n)->origin());
    if (invariants.find(argument) == invariants.end())
      return false;
  }

  return true;
}

void
push_top(jlm::rvsdg::theta_node * theta)
{
  auto subregion = theta->subregion();

  /* push out all nullary nodes */
  for (auto & node : subregion->top_nodes)
  {
    if (!has_side_effects(&node))
      copy_from_theta(&node);
  }

  /* collect loop invariant arguments */
  std::unordered_set<jlm::rvsdg::argument *> invariants;
  for (const auto & lv : *theta)
  {
    if (lv->result()->origin() == lv->argument())
      invariants.insert(lv->argument());
  }

  /* initialize worklist */
  worklist wl;
  for (const auto & lv : *theta)
  {
    auto argument = lv->argument();
    for (const auto & user : *argument)
    {
      auto tmp = input_node(user);
      if (tmp && tmp->depth() == 0 && is_theta_invariant(tmp, invariants))
        wl.push_back(tmp);
    }
  }

  /* process worklist */
  while (!wl.empty())
  {
    auto node = wl.pop_front();

    /* we cannot push out nodes with side-effects */
    if (has_side_effects(node))
      continue;

    auto arguments = copy_from_theta(node);
    invariants.insert(arguments.begin(), arguments.end());

    /* add consumers to worklist */
    for (const auto & argument : arguments)
    {
      for (const auto & user : *argument)
      {
        auto tmp = input_node(user);
        if (tmp && tmp->depth() == 0 && is_theta_invariant(tmp, invariants))
          wl.push_back(tmp);
      }
    }
  }
}

static bool
is_invariant(const jlm::rvsdg::argument * argument)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(argument->region()->node()));
  return argument->region()->result(argument->index() + 1)->origin() == argument;
}

static bool
is_movable_store(jlm::rvsdg::node * node)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(node->region()->node()));
  JLM_ASSERT(jlm::rvsdg::is<StoreOperation>(node));

  auto address = dynamic_cast<jlm::rvsdg::argument *>(node->input(0)->origin());
  if (!address || !is_invariant(address) || address->nusers() != 2)
    return false;

  for (size_t n = 2; n < node->ninputs(); n++)
  {
    auto argument = dynamic_cast<jlm::rvsdg::argument *>(node->input(n)->origin());
    if (!argument || argument->nusers() > 1)
      return false;
  }

  for (size_t n = 0; n < node->noutputs(); n++)
  {
    auto output = node->output(n);
    if (output->nusers() != 1)
      return false;

    if (!dynamic_cast<jlm::rvsdg::result *>(*output->begin()))
      return false;
  }

  return true;
}

static void
pushout_store(jlm::rvsdg::node * storenode)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(storenode->region()->node()));
  JLM_ASSERT(jlm::rvsdg::is<StoreOperation>(storenode) && is_movable_store(storenode));
  auto theta = static_cast<jlm::rvsdg::theta_node *>(storenode->region()->node());
  auto storeop = static_cast<const StoreOperation *>(&storenode->operation());
  auto oaddress = static_cast<jlm::rvsdg::argument *>(storenode->input(0)->origin());
  auto ovalue = storenode->input(1)->origin();

  /* insert new value for store */
  auto nvalue = theta->add_loopvar(UndefValueOperation::Create(*theta->region(), ovalue->type()));
  nvalue->result()->divert_to(ovalue);

  /* collect store operands */
  std::vector<jlm::rvsdg::output *> states;
  auto address = oaddress->input()->origin();
  for (size_t n = 0; n < storenode->noutputs(); n++)
  {
    JLM_ASSERT(storenode->output(n)->nusers() == 1);
    auto result = static_cast<jlm::rvsdg::result *>(*storenode->output(n)->begin());
    result->divert_to(storenode->input(n + 2)->origin());
    states.push_back(result->output());
  }

  /* create new store and redirect theta output users */
  auto nstates = StoreNode::Create(address, nvalue, states, storeop->GetAlignment());
  for (size_t n = 0; n < states.size(); n++)
  {
    std::unordered_set<jlm::rvsdg::input *> users;
    for (const auto & user : *states[n])
    {
      if (input_node(user) != jlm::rvsdg::node_output::node(nstates[0]))
        users.insert(user);
    }

    for (const auto & user : users)
      user->divert_to(nstates[n]);
  }

  remove(storenode);
}

void
push_bottom(jlm::rvsdg::theta_node * theta)
{
  for (const auto & lv : *theta)
  {
    auto storenode = jlm::rvsdg::node_output::node(lv->result()->origin());
    if (jlm::rvsdg::is<StoreOperation>(storenode) && is_movable_store(storenode))
    {
      pushout_store(storenode);
      break;
    }
  }
}

void
push(jlm::rvsdg::theta_node * theta)
{
  bool done = false;
  while (!done)
  {
    auto nnodes = theta->subregion()->nnodes();
    push_top(theta);
    push_bottom(theta);
    if (nnodes == theta->subregion()->nnodes())
      done = true;
  }
}

static void
push(jlm::rvsdg::region * region)
{
  for (auto node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto strnode = dynamic_cast<const jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < strnode->nsubregions(); n++)
        push(strnode->subregion(n));
    }

    if (auto gamma = dynamic_cast<jlm::rvsdg::gamma_node *>(node))
      push(gamma);

    if (auto theta = dynamic_cast<jlm::rvsdg::theta_node *>(node))
      push(theta);
  }
}

static void
push(RvsdgModule & rm, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = pushstat::Create(rm.SourceFileName());

  statistics->start(rm.Rvsdg());
  push(rm.Rvsdg().root());
  statistics->end(rm.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* pushout class */

pushout::~pushout()
{}

void
pushout::run(RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  push(module, statisticsCollector);
}

}
