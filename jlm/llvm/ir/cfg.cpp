/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/util/Program.hpp>

#include <algorithm>
#include <deque>
#include <fstream>
#include <unordered_map>

namespace jlm::llvm
{

Argument::~Argument() noexcept = default;

EntryNode::~EntryNode() noexcept = default;

ExitNode::~ExitNode() noexcept = default;

ControlFlowGraph::ControlFlowGraph(InterProceduralGraphModule & im)
    : module_(im)
{
  entry_ = std::make_unique<EntryNode>(*this);
  exit_ = std::make_unique<ExitNode>(*this);
  entry_->add_outedge(exit_.get());
}

ControlFlowGraph::iterator
ControlFlowGraph::remove_node(ControlFlowGraph::iterator & nodeit)
{
  auto & cfg = nodeit->cfg();

  for (auto & inedge : nodeit->InEdges())
  {
    if (inedge.source() != &*nodeit)
      throw util::Error("cannot remove node. It has still incoming edges.");
  }

  nodeit->remove_outedges();
  std::unique_ptr<BasicBlock> tmp(&*nodeit);
  auto rit = iterator(std::next(cfg.nodes_.find(tmp)));
  cfg.nodes_.erase(tmp);
  tmp.release();
  return rit;
}

ControlFlowGraph::iterator
ControlFlowGraph::remove_node(BasicBlock * bb)
{
  auto & cfg = bb->cfg();

  auto it = cfg.find_node(bb);
  return remove_node(it);
}

util::graph::Graph &
ControlFlowGraph::toDot(util::graph::Writer & writer, const ControlFlowGraph & controlFlowGraph)
{
  util::graph::Graph & dotGraph = writer.CreateGraph();
  dotGraph.SetProgramObject(controlFlowGraph);

  // Handle entry node
  const auto entryNode = controlFlowGraph.entry();
  auto & dotEntryNode = dotGraph.CreateInOutNode(0, entryNode->NumOutEdges());
  dotEntryNode.SetProgramObject(*entryNode);

  std::string label = "Entry\n";
  for (const auto argument : entryNode->arguments())
  {
    label += argument->name() + " <" + argument->type().debug_string() + ">\n";
  }
  dotEntryNode.SetLabel(label);

  // Handle exit node
  const auto exitNode = controlFlowGraph.exit();
  auto & dotExitNode = dotGraph.CreateInOutNode(exitNode->NumInEdges(), 0);
  dotExitNode.SetProgramObject(*exitNode);

  label = "Exit\n";
  for (const auto result : exitNode->results())
  {
    label += result->name() + " <" + result->type().debug_string() + ">\n";
  }
  dotExitNode.SetLabel(label);

  // Handle basic blocks
  for (auto & basicBlock : controlFlowGraph)
  {
    auto & dotBasicBlock =
        dotGraph.CreateInOutNode(basicBlock.NumInEdges(), basicBlock.NumOutEdges());
    dotBasicBlock.SetProgramObject(basicBlock);

    label = util::strfmt("BasicBlock ", &basicBlock, "\n");
    for (const auto & tac : basicBlock.tacs())
      label += ThreeAddressCode::ToAscii(*tac) + "\n";
    dotBasicBlock.SetLabel(label);
  }

  // Handle edges
  auto createEdge = [&dotGraph](const ControlFlowGraphEdge & edge, const size_t index)
  {
    auto & dotSourceNode = dotGraph.GetFromProgramObject<util::graph::InOutNode>(*edge.source());
    auto & dotSinkNode = dotGraph.GetFromProgramObject<util::graph::InOutNode>(*edge.sink());
    auto & sourcePort = dotSourceNode.GetOutputPort(edge.index());
    sourcePort.SetLabel(util::strfmt(edge.index()));
    auto & sinkPort = dotSinkNode.GetInputPort(index);
    dotGraph.CreateDirectedEdge(sourcePort, sinkPort);
  };

  auto createEdges = [&createEdge](const ControlFlowGraphNode & node)
  {
    size_t index = 0;
    for (auto & edge : node.InEdges())
    {
      createEdge(edge, index++);
    }
  };

  for (auto & basicBlock : controlFlowGraph)
  {
    createEdges(basicBlock);
  }
  createEdges(*exitNode);

  return dotGraph;
}

void
ControlFlowGraph::view() const
{
  util::graph::Writer graphWriter;
  toDot(graphWriter, *this);

  const util::FilePath outputFilePath =
      util::FilePath::createUniqueFileName(util::FilePath::TempDirectoryPath(), "cfg-", ".dot");

  std::ofstream outputFile(outputFilePath.to_str());
  graphWriter.outputAllGraphs(outputFile, util::graph::OutputFormat::Dot);

  util::executeProgramAndWait(util::getDotViewer(), { outputFilePath.to_str() });
}

std::string
ControlFlowGraph::ToAscii(const ControlFlowGraph & controlFlowGraph)
{
  std::string str;
  auto nodes = breadth_first(controlFlowGraph);
  auto labels = CreateLabels(nodes);
  for (const auto & node : nodes)
  {
    str += labels.at(node) + ":";
    str += (is<BasicBlock>(node) ? "\n" : " ");

    if (const auto entryNode = dynamic_cast<const EntryNode *>(node))
    {
      str += ToAscii(*entryNode);
    }
    else if (const auto exitNode = dynamic_cast<const ExitNode *>(node))
    {
      str += ToAscii(*exitNode);
    }
    else if (auto basicBlock = dynamic_cast<const BasicBlock *>(node))
    {
      str += ToAscii(*basicBlock, labels);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled control flow graph node type!");
    }
  }

  return str;
}

std::string
ControlFlowGraph::ToAscii(const EntryNode & entryNode)
{
  std::string str;
  for (size_t n = 0; n < entryNode.narguments(); n++)
  {
    str += entryNode.argument(n)->debug_string() + " ";
  }

  return str + "\n";
}

std::string
ControlFlowGraph::ToAscii(const ExitNode & exitNode)
{
  std::string str;
  for (size_t n = 0; n < exitNode.nresults(); n++)
  {
    str += exitNode.result(n)->debug_string() + " ";
  }

  return str;
}

std::string
ControlFlowGraph::ToAscii(
    const BasicBlock & basicBlock,
    const std::unordered_map<ControlFlowGraphNode *, std::string> & labels)
{
  auto & threeAddressCodes = basicBlock.tacs();

  std::string str;
  for (const auto & tac : threeAddressCodes)
  {
    str += "\t" + ThreeAddressCode::ToAscii(*tac);
    if (tac != threeAddressCodes.last())
      str += "\n";
  }

  if (threeAddressCodes.last())
  {
    if (is<BranchOperation>(threeAddressCodes.last()->operation()))
      str += " " + CreateTargets(basicBlock, labels);
    else
      str += "\n\t" + CreateTargets(basicBlock, labels);
  }
  else
  {
    str += "\t" + CreateTargets(basicBlock, labels);
  }

  return str + "\n";
}

std::string
ControlFlowGraph::CreateTargets(
    const ControlFlowGraphNode & node,
    const std::unordered_map<ControlFlowGraphNode *, std::string> & labels)
{
  size_t n = 0;
  std::string str("[");
  for (auto & outedge : node.OutEdges())
  {
    str += labels.at(outedge.sink());
    if (n != node.NumOutEdges() - 1)
      str += ", ";
  }
  str += "]";

  return str;
}

std::unordered_map<ControlFlowGraphNode *, std::string>
ControlFlowGraph::CreateLabels(const std::vector<ControlFlowGraphNode *> & nodes)
{
  std::unordered_map<ControlFlowGraphNode *, std::string> map;
  for (size_t n = 0; n < nodes.size(); n++)
  {
    auto node = nodes[n];
    if (is<EntryNode>(node))
    {
      map[node] = "entry";
    }
    else if (is<ExitNode>(node))
    {
      map[node] = "exit";
    }
    else if (is<BasicBlock>(node))
    {
      map[node] = util::strfmt("bb", n);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled control flow graph node type!");
    }
  }

  return map;
}

std::vector<ControlFlowGraphNode *>
postorder(const ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_closed(cfg));

  std::function<void(
      ControlFlowGraphNode *,
      std::unordered_set<ControlFlowGraphNode *> &,
      std::vector<ControlFlowGraphNode *> &)>
      traverse = [&](ControlFlowGraphNode * node,
                     std::unordered_set<ControlFlowGraphNode *> & visited,
                     std::vector<ControlFlowGraphNode *> & nodes)
  {
    visited.insert(node);
    for (size_t n = 0; n < node->NumOutEdges(); n++)
    {
      auto edge = node->OutEdge(n);
      if (visited.find(edge->sink()) == visited.end())
        traverse(edge->sink(), visited, nodes);
    }

    nodes.push_back(node);
  };

  std::vector<ControlFlowGraphNode *> nodes;
  std::unordered_set<ControlFlowGraphNode *> visited;
  traverse(cfg.entry(), visited, nodes);

  return nodes;
}

std::vector<ControlFlowGraphNode *>
reverse_postorder(const ControlFlowGraph & cfg)
{
  auto nodes = postorder(cfg);
  std::reverse(nodes.begin(), nodes.end());
  return nodes;
}

std::vector<ControlFlowGraphNode *>
breadth_first(const ControlFlowGraph & cfg)
{
  std::deque<ControlFlowGraphNode *> next({ cfg.entry() });
  std::vector<ControlFlowGraphNode *> nodes({ cfg.entry() });
  std::unordered_set<ControlFlowGraphNode *> visited({ cfg.entry() });
  while (!next.empty())
  {
    auto node = next.front();
    next.pop_front();

    for (auto & outedge : node->OutEdges())
    {
      if (visited.find(outedge.sink()) == visited.end())
      {
        visited.insert(outedge.sink());
        next.push_back(outedge.sink());
        nodes.push_back(outedge.sink());
      }
    }
  }

  return nodes;
}

size_t
ntacs(const ControlFlowGraph & cfg)
{
  size_t ntacs = 0;
  for (auto & node : cfg)
  {
    if (auto bb = dynamic_cast<const BasicBlock *>(&node))
      ntacs += bb->ntacs();
  }

  return ntacs;
}

}
