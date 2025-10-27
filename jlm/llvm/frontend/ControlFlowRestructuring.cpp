/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <deque>
#include <unordered_map>

namespace jlm::llvm
{

struct TailControlledLoop
{
  TailControlledLoop(ControlFlowGraphNode * entry, BasicBlock * i, BasicBlock * r)
      : ne(entry),
        insert(i),
        replacement(r)
  {}

  ControlFlowGraphNode * ne;
  BasicBlock * insert;
  BasicBlock * replacement;
};

static TailControlledLoop
ExtractLoop(ControlFlowGraphNode & loopEntry, ControlFlowGraphNode & loopExit)
{
  JLM_ASSERT(loopExit.NumOutEdges() == 2);
  auto & cfg = loopEntry.cfg();

  auto er = loopExit.OutEdge(0);
  auto ex = loopExit.OutEdge(1);
  if (er->sink() != &loopEntry)
  {
    er = loopExit.OutEdge(1);
    ex = loopExit.OutEdge(0);
  }
  JLM_ASSERT(er->sink() == &loopEntry);

  auto exsink = BasicBlock::create(cfg);
  auto replacement = BasicBlock::create(cfg);
  loopEntry.divert_inedges(replacement);
  replacement->add_outedge(ex->sink());
  ex->divert(exsink);
  er->divert(&loopEntry);

  return TailControlledLoop(&loopEntry, exsink, replacement);
}

static void
ReinsertLoop(const TailControlledLoop & loop)
{
  JLM_ASSERT(loop.insert->NumInEdges() == 1);
  JLM_ASSERT(loop.replacement->NumOutEdges() == 1);
  auto & cfg = loop.ne->cfg();

  loop.replacement->divert_inedges(loop.ne);
  loop.insert->divert_inedges(loop.replacement->OutEdge(0)->sink());

  cfg.remove_node(loop.insert);
  cfg.remove_node(loop.replacement);
}

static const ThreeAddressCodeVariable *
CreateContinuationVariable(BasicBlock & bb, std::shared_ptr<const rvsdg::ControlType> type)
{
  static size_t c = 0;
  const auto name = util::strfmt("#p", c++, "#");
  return bb.insert_before_branch(UndefValueOperation::Create(std::move(type), name))->result(0);
}

static const ThreeAddressCodeVariable &
CreateLoopExitVariable(BasicBlock & bb, std::shared_ptr<const rvsdg::ControlType> type)
{
  static size_t c = 0;
  const auto name = util::strfmt("#q", c++, "#");

  auto exitVariable = UndefValueOperation::Create(std::move(type), name);
  return *bb.append_last(std::move(exitVariable))->result(0);
}

static const ThreeAddressCodeVariable &
CreateLoopEntryVariable(BasicBlock & bb, std::shared_ptr<const rvsdg::ControlType> type)
{
  static size_t c = 0;
  const auto name = util::strfmt("#q", c++, "#");

  auto entryVariable = UndefValueOperation::Create(std::move(type), name);
  return *bb.insert_before_branch(std::move(entryVariable))->result(0);
}

static const ThreeAddressCodeVariable &
CreateLoopRepetitionVariable(BasicBlock & basicBlock)
{
  static size_t c = 0;
  const auto name = util::strfmt("#r", c++, "#");

  auto repetitionVariable = UndefValueOperation::Create(rvsdg::ControlType::Create(2), name);
  return *basicBlock.append_last(std::move(repetitionVariable))->result(0);
}

static void
AppendBranch(BasicBlock & basicBlock, const Variable * operand)
{
  const auto numAlternatives =
      util::assertedCast<const rvsdg::ControlType>(&operand->type())->nalternatives();
  basicBlock.append_last(BranchOperation::create(numAlternatives, operand));
}

static void
AppendConstantAssignment(
    BasicBlock & basicBlock,
    const ThreeAddressCodeVariable & variable,
    const size_t value)
{
  const auto numAlternatives =
      util::assertedCast<const rvsdg::ControlType>(&variable.type())->nalternatives();

  const rvsdg::ControlConstantOperation op(
      rvsdg::ControlValueRepresentation(value, numAlternatives));
  basicBlock.append_last(ThreeAddressCode::create(op, {}));
  basicBlock.append_last(AssignmentOperation::create(basicBlock.last()->result(0), &variable));
}

static void
RestructureLoopEntry(
    const StronglyConnectedComponentStructure & sccStructure,
    BasicBlock * newEntryNode,
    const ThreeAddressCodeVariable * entryVariable)
{
  size_t n = 0;
  std::unordered_map<ControlFlowGraphNode *, size_t> indices;
  for (auto & node : sccStructure.EntryNodes())
  {
    newEntryNode->add_outedge(node);
    indices[node] = n++;
  }

  if (entryVariable)
    AppendBranch(*newEntryNode, entryVariable);

  for (auto & edge : sccStructure.EntryEdges())
  {
    auto os = edge->sink();
    edge->divert(newEntryNode);
    if (entryVariable)
      AppendConstantAssignment(*edge->split(), *entryVariable, indices[os]);
  }
}

static void
RestructureLoopExit(
    const StronglyConnectedComponentStructure & sccStructure,
    BasicBlock & newRepetitionNode,
    BasicBlock & newExitNode,
    ControlFlowGraphNode & regionExit,
    const ThreeAddressCodeVariable & repetitionVariable,
    const ThreeAddressCodeVariable * exitVariable)
{
  // It could be that an SCC has no exit edge. This can arise when the input CFG contains a
  // statically detectable endless loop, e.g., entry -> basic block  exit. Note the missing
  //                                                    ^_________|
  // edge to the exit node.
  //
  // Such CFGs do not play well with our restructuring algorithm, as the exit node does not
  // post-dominate the basic block. We circumvent this problem by inserting an additional
  // edge from the newly created exit basic block of the loop to the exit of the SESE region.
  // This edge is never taken at runtime, but fixes the CFGs structure at compile-time such
  // that we can create an RVSDG.
  if (sccStructure.NumExitEdges() == 0)
  {
    newExitNode.add_outedge(&regionExit);
    return;
  }

  size_t n = 0;
  std::unordered_map<ControlFlowGraphNode *, size_t> indices;
  for (auto & node : sccStructure.ExitNodes())
  {
    newExitNode.add_outedge(node);
    indices[node] = n++;
  }

  if (exitVariable)
    AppendBranch(newExitNode, exitVariable);

  for (auto & edge : sccStructure.ExitEdges())
  {
    auto os = edge->sink();
    edge->divert(&newRepetitionNode);
    auto bb = edge->split();
    if (exitVariable)
      AppendConstantAssignment(*bb, *exitVariable, indices[os]);
    AppendConstantAssignment(*bb, repetitionVariable, 0);
  }
}

static void
RestructureLoopRepetition(
    const StronglyConnectedComponentStructure & sccStructure,
    ControlFlowGraphNode & newRepetitionNode,
    const ThreeAddressCodeVariable * entryVariable,
    const ThreeAddressCodeVariable & repetitionVariable)
{
  size_t n = 0;
  std::unordered_map<ControlFlowGraphNode *, size_t> indices;
  for (auto & node : sccStructure.EntryNodes())
    indices[node] = n++;

  for (auto & edge : sccStructure.RepetitionEdges())
  {
    auto os = edge->sink();
    edge->divert(&newRepetitionNode);
    auto basicBlock = edge->split();
    if (entryVariable)
      AppendConstantAssignment(*basicBlock, *entryVariable, indices[os]);
    AppendConstantAssignment(*basicBlock, repetitionVariable, 1);
  }
}

static BasicBlock *
GetEntryVariableBlock(ControlFlowGraphNode * node)
{
  if (auto basicBlock = dynamic_cast<BasicBlock *>(node))
    return basicBlock;

  auto sink = node->OutEdge(0)->sink();
  JLM_ASSERT(is<BasicBlock>(sink));

  return static_cast<BasicBlock *>(sink);
}

static void
RestructureControlFlow(
    ControlFlowGraphNode &,
    ControlFlowGraphNode &,
    std::vector<TailControlledLoop> &);

static void
RestructureLoops(
    ControlFlowGraphNode & regionEntry,
    ControlFlowGraphNode & regionExit,
    std::vector<TailControlledLoop> & loops)
{
  if (&regionEntry == &regionExit)
    return;

  auto & cfg = regionEntry.cfg();

  const auto stronglyConnectedComponents = find_sccs(&regionEntry, &regionExit);
  for (auto & scc : stronglyConnectedComponents)
  {
    auto sccStructure = StronglyConnectedComponentStructure::Create(scc);

    if (sccStructure->IsTailControlledLoop())
    {
      auto loopEntry = *sccStructure->EntryNodes().begin();
      auto loopExit = (*sccStructure->ExitEdges().begin())->source();
      RestructureControlFlow(*loopEntry, *loopExit, loops);
      loops.push_back(ExtractLoop(*loopEntry, *loopExit));
      continue;
    }

    auto & newEntryNode = *BasicBlock::create(cfg);
    auto & newRepetitionNode = *BasicBlock::create(cfg);
    auto & newExitNode = *BasicBlock::create(cfg);
    newRepetitionNode.add_outedge(&newExitNode);
    newRepetitionNode.add_outedge(&newEntryNode);

    const ThreeAddressCodeVariable * entryVariable = nullptr;
    if (sccStructure->NumEntryNodes() > 1)
    {
      auto bb = GetEntryVariableBlock(&regionEntry);
      entryVariable =
          &CreateLoopEntryVariable(*bb, rvsdg::ControlType::Create(sccStructure->NumEntryNodes()));
    }

    auto & repetitionVariable = CreateLoopRepetitionVariable(newEntryNode);

    const ThreeAddressCodeVariable * exitVariable = nullptr;
    if (sccStructure->NumExitNodes() > 1)
      exitVariable = &CreateLoopExitVariable(
          newEntryNode,
          rvsdg::ControlType::Create(sccStructure->NumExitNodes()));

    AppendBranch(newRepetitionNode, &repetitionVariable);

    RestructureLoopEntry(*sccStructure, &newEntryNode, entryVariable);
    RestructureLoopExit(
        *sccStructure,
        newRepetitionNode,
        newExitNode,
        regionExit,
        repetitionVariable,
        exitVariable);
    RestructureLoopRepetition(*sccStructure, newRepetitionNode, entryVariable, repetitionVariable);

    RestructureControlFlow(newEntryNode, newRepetitionNode, loops);
    loops.push_back(ExtractLoop(newEntryNode, newRepetitionNode));
  }
}

static ControlFlowGraphNode &
ComputeHeadBranch(ControlFlowGraphNode & start, ControlFlowGraphNode & end)
{
  ControlFlowGraphNode * headBranch = &start;
  do
  {
    if (headBranch->is_branch() || headBranch == &end)
      break;

    headBranch = headBranch->OutEdge(0)->sink();
  } while (true);

  return *headBranch;
}

static util::HashSet<ControlFlowGraphNode *>
ComputeDominatorGraph(const ControlFlowGraphEdge * edge)
{
  util::HashSet<ControlFlowGraphNode *> nodes;
  util::HashSet edges({ edge });

  std::deque toVisit(1, edge->sink());
  while (toVisit.size() != 0)
  {
    ControlFlowGraphNode * node = toVisit.front();
    toVisit.pop_front();
    if (nodes.Contains(node))
      continue;

    bool accept = true;
    for (auto & inedge : node->InEdges())
    {
      if (!edges.Contains(&inedge))
      {
        accept = false;
        break;
      }
    }

    if (accept)
    {
      nodes.insert(node);
      for (auto & outedge : node->OutEdges())
      {
        edges.insert(&outedge);
        toVisit.push_back(outedge.sink());
      }
    }
  }

  return nodes;
}

struct Continuation
{
  util::HashSet<ControlFlowGraphNode *> points;
  std::unordered_map<ControlFlowGraphEdge *, util::HashSet<ControlFlowGraphEdge *>> edges;
};

static Continuation
ComputeContinuation(const ControlFlowGraphNode & headBranch)
{
  JLM_ASSERT(headBranch.NumOutEdges() > 1);

  std::unordered_map<ControlFlowGraphEdge *, util::HashSet<ControlFlowGraphNode *>> dominatorGraphs;
  for (auto & outedge : headBranch.OutEdges())
    dominatorGraphs[&outedge] = ComputeDominatorGraph(&outedge);

  Continuation c;
  for (auto & outedge : headBranch.OutEdges())
  {
    auto & dominatorGraph = dominatorGraphs[&outedge];
    if (dominatorGraph.IsEmpty())
    {
      c.edges[&outedge].insert(&outedge);
      c.points.insert(outedge.sink());
      continue;
    }

    for (const auto & node : dominatorGraph.Items())
    {
      for (auto & outedge2 : node->OutEdges())
      {
        if (!dominatorGraph.Contains(outedge2.sink()))
        {
          c.edges[&outedge].insert(&outedge2);
          c.points.insert(outedge2.sink());
        }
      }
    }
  }

  return c;
}

static void
RestructureBranches(ControlFlowGraphNode & entry, ControlFlowGraphNode & exit)
{
  auto & cfg = entry.cfg();

  auto & headBranch = ComputeHeadBranch(entry, exit);
  if (&headBranch == &exit)
    return;

  JLM_ASSERT(is<BasicBlock>(&headBranch));
  auto & hbb = *static_cast<BasicBlock *>(&headBranch);

  auto [continuationPoints, continuationEdgesDict] = ComputeContinuation(headBranch);
  JLM_ASSERT(!continuationPoints.IsEmpty());

  if (continuationPoints.Size() == 1)
  {
    const auto continuationPoint = *continuationPoints.Items().begin();
    for (auto & outedge : headBranch.OutEdges())
    {
      auto continuationEdges = continuationEdgesDict[&outedge];

      // Empty branch subgraph
      if (outedge.sink() == continuationPoint)
      {
        outedge.split();
        continue;
      }

      // only one continuation edge
      if (continuationEdges.Size() == 1)
      {
        const auto continuationEdge = *continuationEdges.Items().begin();
        JLM_ASSERT(continuationEdge != &outedge);
        RestructureBranches(*outedge.sink(), *continuationEdge->source());
        continue;
      }

      // more than one continuation edge
      auto nullNode = BasicBlock::create(cfg);
      nullNode->add_outedge(continuationPoint);
      for (const auto & e : continuationEdges.Items())
        e->divert(nullNode);
      RestructureBranches(*outedge.sink(), *nullNode);
    }

    // Restructure tail subgraph
    RestructureBranches(*continuationPoint, exit);
    return;
  }

  // insert new continuation point
  auto p = CreateContinuationVariable(hbb, rvsdg::ControlType::Create(continuationPoints.Size()));
  auto continuationNode = BasicBlock::create(cfg);
  AppendBranch(*continuationNode, p);
  std::unordered_map<ControlFlowGraphNode *, size_t> indices;
  for (const auto & cp : continuationPoints.Items())
  {
    continuationNode->add_outedge(cp);
    indices.insert({ cp, indices.size() });
  }

  // Restructure branch subgraphs
  for (auto & outedge : headBranch.OutEdges())
  {
    auto continuationEdges = continuationEdgesDict[&outedge];

    auto nullNode = BasicBlock::create(cfg);
    nullNode->add_outedge(continuationNode);
    for (const auto & e : continuationEdges.Items())
    {
      auto bb = BasicBlock::create(cfg);
      AppendConstantAssignment(*bb, *p, indices[e->sink()]);
      bb->add_outedge(nullNode);
      e->divert(bb);
    }

    RestructureBranches(*outedge.sink(), *nullNode);
  }

  // Restructure tail subgraph
  RestructureBranches(*continuationNode, exit);
}

void
RestructureLoops(ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_closed(cfg));

  std::vector<TailControlledLoop> loops;
  RestructureLoops(*cfg.entry(), *cfg.exit(), loops);

  for (const auto & loop : loops)
    ReinsertLoop(loop);
}

void
RestructureBranches(ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_acyclic(cfg));
  RestructureBranches(*cfg.entry(), *cfg.exit());
  JLM_ASSERT(is_proper_structured(cfg));
}

static void
RestructureControlFlow(
    ControlFlowGraphNode & entry,
    ControlFlowGraphNode & exit,
    std::vector<TailControlledLoop> & tailControlledLoops)
{
  RestructureLoops(entry, exit, tailControlledLoops);
  RestructureBranches(entry, exit);
}

void
RestructureControlFlow(ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_closed(cfg));

  std::vector<TailControlledLoop> loops;
  RestructureControlFlow(*cfg.entry(), *cfg.exit(), loops);

  for (const auto & loop : loops)
    ReinsertLoop(loop);

  JLM_ASSERT(is_proper_structured(cfg));
}

}
