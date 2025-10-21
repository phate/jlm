/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <algorithm>
#include <unordered_map>

namespace jlm::llvm
{

StronglyConnectedComponent::constiterator
StronglyConnectedComponent::begin() const
{
  return constiterator(nodes_.begin());
}

StronglyConnectedComponent::constiterator
StronglyConnectedComponent::end() const
{
  return constiterator(nodes_.end());
}

bool
StronglyConnectedComponentStructure::IsTailControlledLoop() const noexcept
{
  return NumEntryNodes() == 1 && NumRepetitionEdges() == 1 && NumExitEdges() == 1
      && (*RepetitionEdges().begin())->source() == (*ExitEdges().begin())->source();
}

std::unique_ptr<StronglyConnectedComponentStructure>
StronglyConnectedComponentStructure::Create(const StronglyConnectedComponent & scc)
{
  auto sccStructure = std::make_unique<StronglyConnectedComponentStructure>();

  for (auto & node : scc)
  {
    for (auto & inEdge : node.InEdges())
    {
      if (!scc.contains(inEdge.source()))
      {
        sccStructure->EntryEdges_.insert(&inEdge);
        if (!sccStructure->EntryNodes_.Contains(&node))
          sccStructure->EntryNodes_.insert(&node);
      }
    }

    for (auto & outEdge : node.OutEdges())
    {
      if (!scc.contains(outEdge.sink()))
      {
        sccStructure->ExitEdges_.insert(&outEdge);
        if (!sccStructure->ExitNodes_.Contains(outEdge.sink()))
          sccStructure->ExitNodes_.insert(outEdge.sink());
      }
    }
  }

  for (auto & node : scc)
  {
    for (auto & outEdge : node.OutEdges())
    {
      if (sccStructure->EntryNodes_.Contains(outEdge.sink()))
        sccStructure->RepetitionEdges_.insert(&outEdge);
    }
  }

  return sccStructure;
}

/**
 * Tarjan's SCC algorithm
 */
static void
strongconnect(
    ControlFlowGraphNode * node,
    ControlFlowGraphNode * exit,
    std::unordered_map<ControlFlowGraphNode *, std::pair<size_t, size_t>> & map,
    std::vector<ControlFlowGraphNode *> & node_stack,
    size_t & index,
    std::vector<StronglyConnectedComponent> & sccs)
{
  map.emplace(node, std::make_pair(index, index));
  node_stack.push_back(node);
  index++;

  if (node != exit)
  {
    for (auto & edge : node->OutEdges())
    {
      auto successor = edge.sink();
      if (map.find(successor) == map.end())
      {
        /* successor has not been visited yet; recurse on it */
        strongconnect(successor, exit, map, node_stack, index, sccs);
        map[node].second = std::min(map[node].second, map[successor].second);
      }
      else if (std::find(node_stack.begin(), node_stack.end(), successor) != node_stack.end())
      {
        /* successor is in stack and hence in the current SCC */
        map[node].second = std::min(map[node].second, map[successor].first);
      }
    }
  }

  if (map[node].second == map[node].first)
  {
    std::unordered_set<ControlFlowGraphNode *> set;
    ControlFlowGraphNode * w = nullptr;
    do
    {
      w = node_stack.back();
      node_stack.pop_back();
      set.insert(w);
    } while (w != node);

    if (set.size() != 1 || (*set.begin())->has_selfloop_edge())
      sccs.push_back(StronglyConnectedComponent(set));
  }
}

std::vector<StronglyConnectedComponent>
find_sccs(const ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_closed(cfg));

  return find_sccs(cfg.entry(), cfg.exit());
}

std::vector<StronglyConnectedComponent>
find_sccs(ControlFlowGraphNode * entry, ControlFlowGraphNode * exit)
{
  size_t index = 0;
  std::vector<StronglyConnectedComponent> sccs;
  std::vector<ControlFlowGraphNode *> node_stack;
  std::unordered_map<ControlFlowGraphNode *, std::pair<size_t, size_t>> map;
  strongconnect(entry, exit, map, node_stack, index, sccs);

  return sccs;
}

static std::unique_ptr<ControlFlowGraph>
copy_structural(const ControlFlowGraph & in)
{
  JLM_ASSERT(is_valid(in));

  std::unique_ptr<ControlFlowGraph> out(new ControlFlowGraph(in.module()));
  out->entry()->remove_outedge(0);

  /* create all nodes */
  std::unordered_map<const jlm::llvm::ControlFlowGraphNode *, jlm::llvm::ControlFlowGraphNode *>
      node_map({ { in.entry(), out->entry() }, { in.exit(), out->exit() } });

  for (const auto & node : in)
  {
    JLM_ASSERT(jlm::llvm::is<jlm::llvm::BasicBlock>(&node));
    node_map[&node] = jlm::llvm::BasicBlock::create(*out);
  }

  /* establish control flow */
  node_map[in.entry()]->add_outedge(node_map[in.entry()->OutEdge(0)->sink()]);
  for (const auto & node : in)
  {
    for (auto & outedge : node.OutEdges())
      node_map[&node]->add_outedge(node_map[outedge.sink()]);
  }

  return out;
}

static inline bool
is_loop(const jlm::llvm::ControlFlowGraphNode * node) noexcept
{
  return node->NumInEdges() == 2 && node->NumOutEdges() == 2 && node->has_selfloop_edge();
}

static inline bool
is_linear_reduction(const jlm::llvm::ControlFlowGraphNode * node) noexcept
{
  if (node->NumOutEdges() != 1)
    return false;

  if (node->OutEdge(0)->sink()->NumInEdges() != 1)
    return false;

  return true;
}

static inline jlm::llvm::ControlFlowGraphNode *
find_join(const jlm::llvm::ControlFlowGraphNode * split) noexcept
{
  JLM_ASSERT(split->NumOutEdges() > 1);
  auto s1 = split->OutEdge(0)->sink();
  auto s2 = split->OutEdge(1)->sink();

  jlm::llvm::ControlFlowGraphNode * join = nullptr;
  if (s1->NumOutEdges() == 1 && s1->OutEdge(0)->sink() == s2)
    join = s2;
  else if (s2->NumOutEdges() == 1 && s2->OutEdge(0)->sink() == s1)
    join = s1;
  else if (
      s1->NumOutEdges() == 1 && s2->NumOutEdges() == 1
      && (s1->OutEdge(0)->sink() == s2->OutEdge(0)->sink()))
    join = s1->OutEdge(0)->sink();

  return join;
}

static inline bool
is_branch(const jlm::llvm::ControlFlowGraphNode * split) noexcept
{
  if (split->NumOutEdges() < 2)
    return false;

  auto join = find_join(split);
  if (join == nullptr || join->NumInEdges() != split->NumOutEdges())
    return false;

  for (auto & outedge : split->OutEdges())
  {
    if (outedge.sink() == join)
      continue;

    auto node = outedge.sink();
    if (node->NumInEdges() != 1)
      return false;
    if (node->NumOutEdges() != 1 || node->OutEdge(0)->sink() != join)
      return false;
  }

  return true;
}

static inline bool
is_proper_branch(const jlm::llvm::ControlFlowGraphNode * split) noexcept
{
  if (split->NumOutEdges() < 2)
    return false;

  if (split->OutEdge(0)->sink()->NumOutEdges() != 1)
    return false;

  auto join = split->OutEdge(0)->sink()->OutEdge(0)->sink();
  for (auto & outedge : split->OutEdges())
  {
    if (outedge.sink()->NumInEdges() != 1)
      return false;
    if (outedge.sink()->NumOutEdges() != 1)
      return false;
    if (outedge.sink()->OutEdge(0)->sink() != join)
      return false;
  }

  return true;
}

static inline bool
is_T1(const jlm::llvm::ControlFlowGraphNode * node) noexcept
{
  for (auto & outedge : node->OutEdges())
  {
    if (outedge.source() == outedge.sink())
      return true;
  }

  return false;
}

static inline bool
is_T2(const jlm::llvm::ControlFlowGraphNode * node) noexcept
{
  if (node->NumInEdges() == 0)
    return false;

  auto source = node->InEdges().begin()->source();
  for (auto & inedge : node->InEdges())
  {
    if (inedge.source() != source)
      return false;
  }

  return true;
}

static inline void
reduce_loop(
    jlm::llvm::ControlFlowGraphNode * node,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  JLM_ASSERT(is_loop(node));
  auto & cfg = node->cfg();

  auto reduction = jlm::llvm::BasicBlock::create(cfg);
  for (auto & outedge : node->OutEdges())
  {
    if (outedge.is_selfloop())
    {
      node->remove_outedge(outedge.index());
      break;
    }
  }

  reduction->add_outedge(node->OutEdge(0)->sink());
  node->remove_outedges();
  node->divert_inedges(reduction);

  to_visit.erase(node);
  to_visit.insert(reduction);
}

static inline void
reduce_linear(
    jlm::llvm::ControlFlowGraphNode * entry,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  JLM_ASSERT(is_linear_reduction(entry));
  auto exit = entry->OutEdge(0)->sink();
  auto & cfg = entry->cfg();

  auto reduction = jlm::llvm::BasicBlock::create(cfg);
  entry->divert_inedges(reduction);
  for (auto & outedge : exit->OutEdges())
    reduction->add_outedge(outedge.sink());
  exit->remove_outedges();

  to_visit.erase(entry);
  to_visit.erase(exit);
  to_visit.insert(reduction);
}

static inline void
reduce_branch(
    jlm::llvm::ControlFlowGraphNode * split,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  JLM_ASSERT(is_branch(split));
  auto join = find_join(split);
  auto & cfg = split->cfg();

  auto reduction = jlm::llvm::BasicBlock::create(cfg);
  split->divert_inedges(reduction);
  reduction->add_outedge(join);
  for (auto & outedge : split->OutEdges())
  {
    if (outedge.sink() != join)
    {
      outedge.sink()->remove_outedges();
      to_visit.erase(outedge.sink());
    }
  }
  split->remove_outedges();

  to_visit.erase(split);
  to_visit.insert(reduction);
}

static inline void
reduce_proper_branch(
    jlm::llvm::ControlFlowGraphNode * split,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  JLM_ASSERT(is_proper_branch(split));
  auto join = split->OutEdge(0)->sink()->OutEdge(0)->sink();

  auto reduction = jlm::llvm::BasicBlock::create(split->cfg());
  split->divert_inedges(reduction);
  join->remove_inedges();
  reduction->add_outedge(join);
  for (auto & outedge : split->OutEdges())
    to_visit.erase(outedge.sink());

  to_visit.erase(split);
  to_visit.insert(reduction);
}

static inline void
reduce_T1(jlm::llvm::ControlFlowGraphNode * node)
{
  JLM_ASSERT(is_T1(node));

  for (auto & outedge : node->OutEdges())
  {
    if (outedge.source() == outedge.sink())
    {
      node->remove_outedge(outedge.index());
      break;
    }
  }
}

static inline void
reduce_T2(
    jlm::llvm::ControlFlowGraphNode * node,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  JLM_ASSERT(is_T2(node));

  auto p = node->InEdges().begin()->source();
  p->divert_inedges(node);
  p->remove_outedges();
  to_visit.erase(p);
}

static inline bool
reduce_proper_structured(
    jlm::llvm::ControlFlowGraphNode * node,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  if (is_loop(node))
  {
    reduce_loop(node, to_visit);
    return true;
  }

  if (is_proper_branch(node))
  {
    reduce_proper_branch(node, to_visit);
    return true;
  }

  if (is_linear_reduction(node))
  {
    reduce_linear(node, to_visit);
    return true;
  }

  return false;
}

static inline bool
reduce_structured(
    jlm::llvm::ControlFlowGraphNode * node,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  if (is_loop(node))
  {
    reduce_loop(node, to_visit);
    return true;
  }

  if (is_branch(node))
  {
    reduce_branch(node, to_visit);
    return true;
  }

  if (is_linear_reduction(node))
  {
    reduce_linear(node, to_visit);
    return true;
  }

  return false;
}

static inline bool
reduce_reducible(
    jlm::llvm::ControlFlowGraphNode * node,
    std::unordered_set<jlm::llvm::ControlFlowGraphNode *> & to_visit)
{
  if (is_T1(node))
  {
    reduce_T1(node);
    return true;
  }

  if (is_T2(node))
  {
    reduce_T2(node, to_visit);
    return true;
  }

  return false;
}

static bool
has_valid_phis(const BasicBlock & bb)
{
  for (auto it = bb.begin(); it != bb.end(); it++)
  {
    const auto tac = *it;
    const auto phi = dynamic_cast<const SsaPhiOperation *>(&tac->operation());
    if (!phi)
      continue;

    // Ensure all phi nodes are at the beginning of a basic block
    if (tac != bb.first() && !is<SsaPhiOperation>(*std::prev(it)))
      return false;

    // Ensure there are no duplicated incoming blocks in the phi node
    util::HashSet<ControlFlowGraphNode *> phiIncoming;
    for (size_t i = 0; i < phi->narguments(); i++)
    {
      phiIncoming.insert(phi->GetIncomingNode(i));
    }
    if (phiIncoming.Size() != phi->narguments())
      return false;

    // Ensure the set of incoming blocks matches the actual predecessors of this basic block
    util::HashSet<ControlFlowGraphNode *> predecessors;
    for (auto & inEdge : bb.InEdges())
    {
      predecessors.insert(inEdge.source());
    }
    if (phiIncoming != predecessors)
      return false;
  }

  return true;
}

static bool
is_valid_basic_block(const BasicBlock & bb)
{
  if (bb.no_successor())
    return false;

  if (!has_valid_phis(bb))
    return false;

  return true;
}

static bool
has_valid_entry(const ControlFlowGraph & cfg)
{
  if (!cfg.entry()->no_predecessor())
    return false;

  if (cfg.entry()->NumOutEdges() != 1)
    return false;

  return true;
}

static bool
has_valid_exit(const ControlFlowGraph & cfg)
{
  return cfg.exit()->no_successor();
}

bool
is_valid(const ControlFlowGraph & cfg)
{
  if (!has_valid_entry(cfg))
    return false;

  if (!has_valid_exit(cfg))
    return false;

  // check all basic blocks
  for (const auto & bb : cfg)
  {
    if (!is_valid_basic_block(bb))
      return false;
  }

  return true;
}

bool
is_closed(const ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_valid(cfg));

  for (const auto & node : cfg)
  {
    if (node.no_predecessor())
      return false;
  }

  return true;
}

bool
is_linear(const ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_closed(cfg));

  for (const auto & node : cfg)
  {
    if (!node.single_successor() || !node.single_predecessor())
      return false;
  }

  return true;
}

static inline bool
reduce(
    const ControlFlowGraph & cfg,
    const std::function<
        bool(llvm::ControlFlowGraphNode *, std::unordered_set<llvm::ControlFlowGraphNode *> &)> & f)
{
  JLM_ASSERT(is_closed(cfg));
  auto c = copy_structural(cfg);

  std::unordered_set<ControlFlowGraphNode *> to_visit({ c->entry(), c->exit() });
  for (auto & node : *c)
    to_visit.insert(&node);

  auto it = to_visit.begin();
  while (it != to_visit.end())
  {
    bool reduced = f(*it, to_visit);
    it = reduced ? to_visit.begin() : std::next(it);
  }

  return to_visit.size() == 1;
}

bool
is_structured(const ControlFlowGraph & cfg)
{
  return reduce(cfg, reduce_structured);
}

bool
is_proper_structured(const ControlFlowGraph & cfg)
{
  return reduce(cfg, reduce_proper_structured);
}

bool
is_reducible(const ControlFlowGraph & cfg)
{
  return reduce(cfg, reduce_reducible);
}

void
straighten(ControlFlowGraph & cfg)
{
  auto it = cfg.begin();
  while (it != cfg.end())
  {
    BasicBlock * bb = &*it;

    // Check if bb only has one successor, and that the successor only has one predecessor
    if (!is_linear_reduction(bb))
    {
      it++;
      continue;
    }

    auto successor = dynamic_cast<BasicBlock *>(it->OutEdge(0)->sink());
    if (!successor || successor->HasSsaPhiOperation())
    {
      it++;
      continue;
    }

    // successor becomes the single basic block
    successor->append_first(bb->tacs());
    bb->divert_inedges(successor);
    it = cfg.remove_node(it);
  }
}

void
purge(ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_valid(cfg));

  auto it = cfg.begin();
  while (it != cfg.end())
  {
    auto bb = &*it;

    /*
      Ignore basic blocks with instructions
    */
    if (bb->ntacs() != 0)
    {
      it++;
      continue;
    }

    JLM_ASSERT(bb->NumOutEdges() == 1);
    auto outedge = bb->OutEdge(0);
    /*
      Ignore endless loops
    */
    if (outedge->sink() == bb)
    {
      it++;
      continue;
    }

    bb->divert_inedges(outedge->sink());
    it = cfg.remove_node(it);
  }

  JLM_ASSERT(is_valid(cfg));
}

/*
 * @brief Find all nodes dominated by the entry node.
 */
static std::unordered_set<const ControlFlowGraphNode *>
compute_livenodes(const ControlFlowGraph & cfg)
{
  std::unordered_set<const ControlFlowGraphNode *> visited;
  std::unordered_set<ControlFlowGraphNode *> to_visit({ cfg.entry() });
  while (!to_visit.empty())
  {
    auto node = *to_visit.begin();
    to_visit.erase(to_visit.begin());
    JLM_ASSERT(visited.find(node) == visited.end());
    visited.insert(node);
    for (auto & outedge : node->OutEdges())
    {
      if (visited.find(outedge.sink()) == visited.end()
          && to_visit.find(outedge.sink()) == to_visit.end())
        to_visit.insert(outedge.sink());
    }
  }

  return visited;
}

/*
 * @brief Find all nodes that are NOT dominated by the entry node.
 */
static std::unordered_set<ControlFlowGraphNode *>
compute_deadnodes(ControlFlowGraph & cfg)
{
  auto livenodes = compute_livenodes(cfg);

  std::unordered_set<ControlFlowGraphNode *> deadnodes;
  for (auto & node : cfg)
  {
    if (livenodes.find(&node) == livenodes.end())
      deadnodes.insert(&node);
  }

  JLM_ASSERT(deadnodes.find(cfg.entry()) == deadnodes.end());
  JLM_ASSERT(deadnodes.find(cfg.exit()) == deadnodes.end());
  return deadnodes;
}

/*
 * @brief Returns all basic blocks that are live and a sink
 *	of a dead node.
 */
static std::unordered_set<BasicBlock *>
compute_live_sinks(const std::unordered_set<ControlFlowGraphNode *> & deadnodes)
{
  std::unordered_set<BasicBlock *> sinks;
  for (auto & node : deadnodes)
  {
    for (size_t n = 0; n < node->NumOutEdges(); n++)
    {
      auto sink = dynamic_cast<BasicBlock *>(node->OutEdge(n)->sink());
      if (sink && deadnodes.find(sink) == deadnodes.end())
        sinks.insert(sink);
    }
  }

  return sinks;
}

static void
update_phi_operands(
    llvm::ThreeAddressCode & phitac,
    const std::unordered_set<ControlFlowGraphNode *> & deadnodes)
{
  const auto phi = util::assertedCast<const SsaPhiOperation>(&phitac.operation());

  std::vector<ControlFlowGraphNode *> incomingNodes;
  std::vector<const Variable *> operands;
  for (size_t n = 0; n < phitac.noperands(); n++)
  {
    if (deadnodes.find(phi->GetIncomingNode(n)) == deadnodes.end())
    {
      operands.push_back(phitac.operand(n));
      incomingNodes.push_back(phi->GetIncomingNode(n));
    }
  }

  phitac.replace(SsaPhiOperation(std::move(incomingNodes), phi->Type()), operands);
}

static void
update_phi_operands(
    const std::unordered_set<BasicBlock *> & sinks,
    const std::unordered_set<ControlFlowGraphNode *> & deadnodes)
{
  for (auto & sink : sinks)
  {
    for (auto & tac : *sink)
    {
      if (!is<SsaPhiOperation>(tac))
        break;

      update_phi_operands(*tac, deadnodes);
    }
  }
}

static void
remove_deadnodes(const std::unordered_set<ControlFlowGraphNode *> & deadnodes)
{
  for (auto & node : deadnodes)
  {
    node->remove_inedges();
    JLM_ASSERT(is<BasicBlock>(node));
    node->cfg().remove_node(static_cast<BasicBlock *>(node));
  }
}

void
prune(ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_valid(cfg));

  auto deadnodes = compute_deadnodes(cfg);
  auto sinks = compute_live_sinks(deadnodes);
  update_phi_operands(sinks, deadnodes);
  remove_deadnodes(deadnodes);

  JLM_ASSERT(is_closed(cfg));
}

}
