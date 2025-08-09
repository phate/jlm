/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace jlm::llvm
{

struct tcloop
{
  inline tcloop(ControlFlowGraphNode * entry, BasicBlock * i, BasicBlock * r)
      : ne(entry),
        insert(i),
        replacement(r)
  {}

  ControlFlowGraphNode * ne;
  BasicBlock * insert;
  BasicBlock * replacement;
};

static inline tcloop
extract_tcloop(ControlFlowGraphNode * ne, ControlFlowGraphNode * nx)
{
  JLM_ASSERT(nx->NumOutEdges() == 2);
  auto & cfg = ne->cfg();

  auto er = nx->OutEdge(0);
  auto ex = nx->OutEdge(1);
  if (er->sink() != ne)
  {
    er = nx->OutEdge(1);
    ex = nx->OutEdge(0);
  }
  JLM_ASSERT(er->sink() == ne);

  auto exsink = BasicBlock::create(cfg);
  auto replacement = BasicBlock::create(cfg);
  ne->divert_inedges(replacement);
  replacement->add_outedge(ex->sink());
  ex->divert(exsink);
  er->divert(ne);

  return tcloop(ne, exsink, replacement);
}

static inline void
reinsert_tcloop(const tcloop & l)
{
  JLM_ASSERT(l.insert->NumInEdges() == 1);
  JLM_ASSERT(l.replacement->NumOutEdges() == 1);
  auto & cfg = l.ne->cfg();

  l.replacement->divert_inedges(l.ne);
  l.insert->divert_inedges(l.replacement->OutEdge(0)->sink());

  cfg.remove_node(l.insert);
  cfg.remove_node(l.replacement);
}

static const ThreeAddressCodeVariable *
create_pvariable(BasicBlock & bb, std::shared_ptr<const rvsdg::ControlType> type)
{
  static size_t c = 0;
  auto name = util::strfmt("#p", c++, "#");
  return bb.insert_before_branch(UndefValueOperation::Create(std::move(type), name))->result(0);
}

static const ThreeAddressCodeVariable *
create_qvariable(BasicBlock & bb, std::shared_ptr<const rvsdg::ControlType> type)
{
  static size_t c = 0;
  auto name = util::strfmt("#q", c++, "#");
  return bb.append_last(UndefValueOperation::Create(std::move(type), name))->result(0);
}

static const ThreeAddressCodeVariable *
create_tvariable(BasicBlock & bb, std::shared_ptr<const rvsdg::ControlType> type)
{
  static size_t c = 0;
  auto name = util::strfmt("#q", c++, "#");
  return bb.insert_before_branch(UndefValueOperation::Create(std::move(type), name))->result(0);
}

static const ThreeAddressCodeVariable *
create_rvariable(BasicBlock & bb)
{
  static size_t c = 0;
  auto name = util::strfmt("#r", c++, "#");

  return bb.append_last(UndefValueOperation::Create(rvsdg::ControlType::Create(2), name))
      ->result(0);
}

static inline void
append_branch(BasicBlock * bb, const Variable * operand)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ControlType *>(&operand->type()));
  auto nalternatives = static_cast<const rvsdg::ControlType *>(&operand->type())->nalternatives();
  bb->append_last(BranchOperation::create(nalternatives, operand));
}

static inline void
append_constant(BasicBlock * bb, const ThreeAddressCodeVariable * result, size_t value)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ControlType *>(&result->type()));
  auto nalternatives = static_cast<const rvsdg::ControlType *>(&result->type())->nalternatives();

  rvsdg::ctlconstant_op op(rvsdg::ControlValueRepresentation(value, nalternatives));
  bb->append_last(ThreeAddressCode::create(op, {}));
  bb->append_last(AssignmentOperation::create(bb->last()->result(0), result));
}

static inline void
restructure_loop_entry(
    const sccstructure & s,
    BasicBlock * new_ne,
    const ThreeAddressCodeVariable * ev)
{
  size_t n = 0;
  std::unordered_map<llvm::ControlFlowGraphNode *, size_t> indices;
  for (auto & node : s.enodes())
  {
    new_ne->add_outedge(node);
    indices[node] = n++;
  }

  if (ev)
    append_branch(new_ne, ev);

  for (auto & edge : s.eedges())
  {
    auto os = edge->sink();
    edge->divert(new_ne);
    if (ev)
      append_constant(edge->split(), ev, indices[os]);
  }
}

static inline void
restructure_loop_exit(
    const sccstructure & s,
    BasicBlock * new_nr,
    BasicBlock * new_nx,
    ControlFlowGraphNode * exit,
    const ThreeAddressCodeVariable * rv,
    const ThreeAddressCodeVariable * xv)
{
  /*
    It could be that an SCC has no exit edge. This can arise when the input CFG contains a
    statically detectable endless loop, e.g., entry -> basic block  exit. Note the missing
                                                       ^_________|
    edge to the exit node.

    Such CFGs do not play well with our restructuring algorithm, as the exit node does not
    post-dominate the basic block. We circumvent this problem by inserting an additional
    edge from the newly created exit basic block of the loop to the exit of the SESE region.
    This edge is never taken at runtime, but fixes the CFGs structure at compile-time such
    that we can create an RVSDG.
  */
  if (s.nxedges() == 0)
  {
    new_nx->add_outedge(exit);
    return;
  }

  size_t n = 0;
  std::unordered_map<llvm::ControlFlowGraphNode *, size_t> indices;
  for (auto & node : s.xnodes())
  {
    new_nx->add_outedge(node);
    indices[node] = n++;
  }

  if (xv)
    append_branch(new_nx, xv);

  for (auto & edge : s.xedges())
  {
    auto os = edge->sink();
    edge->divert(new_nr);
    auto bb = edge->split();
    if (xv)
      append_constant(bb, xv, indices[os]);
    append_constant(bb, rv, 0);
  }
}

static inline void
restructure_loop_repetition(
    const sccstructure & s,
    ControlFlowGraphNode * new_nr,
    const ThreeAddressCodeVariable * ev,
    const ThreeAddressCodeVariable * rv)
{
  size_t n = 0;
  std::unordered_map<llvm::ControlFlowGraphNode *, size_t> indices;
  for (auto & node : s.enodes())
    indices[node] = n++;

  for (auto & edge : s.redges())
  {
    auto os = edge->sink();
    edge->divert(new_nr);
    auto bb = edge->split();
    if (ev)
      append_constant(bb, ev, indices[os]);
    append_constant(bb, rv, 1);
  }
}

static BasicBlock *
find_tvariable_bb(ControlFlowGraphNode * node)
{
  if (auto bb = dynamic_cast<BasicBlock *>(node))
    return bb;

  auto sink = node->OutEdge(0)->sink();
  JLM_ASSERT(is<BasicBlock>(sink));

  return static_cast<BasicBlock *>(sink);
}

static void
restructure(ControlFlowGraphNode *, ControlFlowGraphNode *, std::vector<tcloop> &);

static void
restructure_loops(
    ControlFlowGraphNode * entry,
    ControlFlowGraphNode * exit,
    std::vector<tcloop> & loops)
{
  if (entry == exit)
    return;

  auto & cfg = entry->cfg();

  auto sccs = find_sccs(entry, exit);
  for (auto & scc : sccs)
  {
    auto sccstruct = sccstructure::create(scc);

    if (sccstruct->is_tcloop())
    {
      auto tcloop_entry = *sccstruct->enodes().begin();
      auto tcloop_exit = (*sccstruct->xedges().begin())->source();
      restructure(tcloop_entry, tcloop_exit, loops);
      loops.push_back(extract_tcloop(tcloop_entry, tcloop_exit));
      continue;
    }

    auto new_ne = BasicBlock::create(cfg);
    auto new_nr = BasicBlock::create(cfg);
    auto new_nx = BasicBlock::create(cfg);
    new_nr->add_outedge(new_nx);
    new_nr->add_outedge(new_ne);

    const ThreeAddressCodeVariable * ev = nullptr;
    if (sccstruct->nenodes() > 1)
    {
      auto bb = find_tvariable_bb(entry);
      ev = create_tvariable(*bb, rvsdg::ControlType::Create(sccstruct->nenodes()));
    }

    auto rv = create_rvariable(*new_ne);

    const ThreeAddressCodeVariable * xv = nullptr;
    if (sccstruct->nxnodes() > 1)
      xv = create_qvariable(*new_ne, rvsdg::ControlType::Create(sccstruct->nxnodes()));

    append_branch(new_nr, rv);

    restructure_loop_entry(*sccstruct, new_ne, ev);
    restructure_loop_exit(*sccstruct, new_nr, new_nx, exit, rv, xv);
    restructure_loop_repetition(*sccstruct, new_nr, ev, rv);

    restructure(new_ne, new_nr, loops);
    loops.push_back(extract_tcloop(new_ne, new_nr));
  }
}

static ControlFlowGraphNode *
find_head_branch(ControlFlowGraphNode * start, ControlFlowGraphNode * end)
{
  do
  {
    if (start->is_branch() || start == end)
      break;

    start = start->OutEdge(0)->sink();
  } while (1);

  return start;
}

static std::unordered_set<llvm::ControlFlowGraphNode *>
find_dominator_graph(const ControlFlowGraphEdge * edge)
{
  std::unordered_set<llvm::ControlFlowGraphNode *> nodes;
  std::unordered_set<const ControlFlowGraphEdge *> edges({ edge });

  std::deque<llvm::ControlFlowGraphNode *> to_visit(1, edge->sink());
  while (to_visit.size() != 0)
  {
    ControlFlowGraphNode * node = to_visit.front();
    to_visit.pop_front();
    if (nodes.find(node) != nodes.end())
      continue;

    bool accept = true;
    for (auto & inedge : node->InEdges())
    {
      if (edges.find(&inedge) == edges.end())
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
        to_visit.push_back(outedge.sink());
      }
    }
  }

  return nodes;
}

struct continuation
{
  std::unordered_set<ControlFlowGraphNode *> points;
  std::unordered_map<ControlFlowGraphEdge *, std::unordered_set<ControlFlowGraphEdge *>> edges;
};

static inline continuation
compute_continuation(ControlFlowGraphNode * hb)
{
  JLM_ASSERT(hb->NumOutEdges() > 1);

  std::unordered_map<ControlFlowGraphEdge *, std::unordered_set<ControlFlowGraphNode *>> dgraphs;
  for (auto & outedge : hb->OutEdges())
    dgraphs[&outedge] = find_dominator_graph(&outedge);

  continuation c;
  for (auto & outedge : hb->OutEdges())
  {
    auto & dgraph = dgraphs[&outedge];
    if (dgraph.empty())
    {
      c.edges[&outedge].insert(&outedge);
      c.points.insert(outedge.sink());
      continue;
    }

    for (const auto & node : dgraph)
    {
      for (auto & outedge2 : node->OutEdges())
      {
        if (dgraph.find(outedge2.sink()) == dgraph.end())
        {
          c.edges[&outedge].insert(&outedge2);
          c.points.insert(outedge2.sink());
        }
      }
    }
  }

  return c;
}

static inline void
restructure_branches(ControlFlowGraphNode * entry, ControlFlowGraphNode * exit)
{
  auto & cfg = entry->cfg();

  auto hb = find_head_branch(entry, exit);
  if (hb == exit)
    return;

  JLM_ASSERT(is<BasicBlock>(hb));
  auto & hbb = *static_cast<BasicBlock *>(hb);

  auto c = compute_continuation(hb);
  JLM_ASSERT(!c.points.empty());

  if (c.points.size() == 1)
  {
    auto cpoint = *c.points.begin();
    for (auto & outedge : hb->OutEdges())
    {
      auto cedges = c.edges[&outedge];

      /* empty branch subgraph */
      if (outedge.sink() == cpoint)
      {
        outedge.split();
        continue;
      }

      /* only one continuation edge */
      if (cedges.size() == 1)
      {
        auto e = *cedges.begin();
        JLM_ASSERT(e != &outedge);
        restructure_branches(outedge.sink(), e->source());
        continue;
      }

      /* more than one continuation edge */
      auto null = BasicBlock::create(cfg);
      null->add_outedge(cpoint);
      for (const auto & e : cedges)
        e->divert(null);
      restructure_branches(outedge.sink(), null);
    }

    /* restructure tail subgraph */
    restructure_branches(cpoint, exit);
    return;
  }

  /* insert new continuation point */
  auto p = create_pvariable(hbb, rvsdg::ControlType::Create(c.points.size()));
  auto cn = BasicBlock::create(cfg);
  append_branch(cn, p);
  std::unordered_map<ControlFlowGraphNode *, size_t> indices;
  for (const auto & cp : c.points)
  {
    cn->add_outedge(cp);
    indices.insert({ cp, indices.size() });
  }

  /* restructure branch subgraphs */
  for (auto & outedge : hb->OutEdges())
  {
    auto cedges = c.edges[&outedge];

    auto null = BasicBlock::create(cfg);
    null->add_outedge(cn);
    for (const auto & e : cedges)
    {
      auto bb = BasicBlock::create(cfg);
      append_constant(bb, p, indices[e->sink()]);
      bb->add_outedge(null);
      e->divert(bb);
    }

    restructure_branches(outedge.sink(), null);
  }

  /* restructure tail subgraph */
  restructure_branches(cn, exit);
}

void
RestructureLoops(ControlFlowGraph * cfg)
{
  JLM_ASSERT(is_closed(*cfg));

  std::vector<tcloop> loops;
  restructure_loops(cfg->entry(), cfg->exit(), loops);

  for (const auto & l : loops)
    reinsert_tcloop(l);
}

void
RestructureBranches(ControlFlowGraph * cfg)
{
  JLM_ASSERT(is_acyclic(*cfg));
  restructure_branches(cfg->entry(), cfg->exit());
  JLM_ASSERT(is_proper_structured(*cfg));
}

static inline void
restructure(
    ControlFlowGraphNode * entry,
    ControlFlowGraphNode * exit,
    std::vector<tcloop> & tcloops)
{
  restructure_loops(entry, exit, tcloops);
  restructure_branches(entry, exit);
}

void
RestructureControlFlow(ControlFlowGraph * cfg)
{
  JLM_ASSERT(is_closed(*cfg));

  std::vector<tcloop> tcloops;
  restructure(cfg->entry(), cfg->exit(), tcloops);

  for (const auto & l : tcloops)
    reinsert_tcloop(l);

  JLM_ASSERT(is_proper_structured(*cfg));
}

}
