/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>

#include <algorithm>
#include <cmath>
#include <regex>

namespace jlm::hls
{

bool
isForbiddenChar(char c)
{
  if (('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9') || '_' == c)
  {
    return false;
  }
  return true;
}

BaseHLS::~BaseHLS() = default;

std::string
BaseHLS::get_node_name(const jlm::rvsdg::Node * node)
{
  auto found = node_map.find(node);
  if (found != node_map.end())
  {
    return found->second;
  }
  std::string append = "";
  auto inPorts = node->ninputs();
  auto outPorts = node->noutputs();
  if (inPorts)
  {
    append.append("_IN");
    append.append(std::to_string(inPorts));
    append.append("_W");
    append.append(std::to_string(JlmSize(node->input(inPorts - 1)->Type().get())));
  }
  if (outPorts)
  {
    append.append("_OUT");
    append.append(std::to_string(outPorts));
    append.append("_W");
    append.append(std::to_string(JlmSize(node->output(outPorts - 1)->Type().get())));
  }
  auto name = util::strfmt("op_", node->DebugString(), append, "_", node_map.size());
  // remove chars that are not valid in firrtl module names
  std::replace_if(name.begin(), name.end(), isForbiddenChar, '_');
  // verilator seems to throw a fit if there are too many underscores in some scenarios
  name = std::regex_replace(name, std::regex("_+"), "_");
  node_map[node] = name;
  return name;
}

std::string
BaseHLS::get_port_name(jlm::rvsdg::Input * port)
{
  std::string result;
  if (jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*port))
  {
    result += "i";
  }
  else if (jlm::rvsdg::TryGetOwnerRegion(*port))
  {
    result += "r";
  }
  else
  {
    throw std::logic_error(port->debug_string() + " not implemented!");
  }
  result += util::strfmt(port->index());
  return result;
}

std::string
BaseHLS::get_port_name(jlm::rvsdg::Output * port)
{
  if (port == nullptr)
  {
    throw std::logic_error("nullptr!");
  }
  std::string result;
  if (dynamic_cast<const jlm::rvsdg::RegionArgument *>(port))
  {
    result += "a";
  }
  else if (dynamic_cast<const rvsdg::NodeOutput *>(port))
  {
    result += "o";
  }
  else if (dynamic_cast<const rvsdg::StructuralOutput *>(port))
  {
    result += "so";
  }
  else
  {
    throw std::logic_error(port->debug_string() + " not implemented!");
  }
  result += util::strfmt(port->index());
  return result;
}

int
BaseHLS::JlmSize(const jlm::rvsdg::Type * type)
{
  return jlm::hls::JlmSize(type);
}

void
BaseHLS::create_node_names(rvsdg::Region * r)
{
  for (auto & node : r->Nodes())
  {
    if (dynamic_cast<jlm::rvsdg::SimpleNode *>(&node))
    {
      get_node_name(&node);
    }
    else if (auto oln = dynamic_cast<LoopNode *>(&node))
    {
      create_node_names(oln->subregion());
    }
    else
    {
      throw util::Error("Unimplemented op (unexpected structural node) : " + node.DebugString());
    }
  }
}

const jlm::rvsdg::LambdaNode *
BaseHLS::get_hls_lambda(llvm::LlvmRvsdgModule & rm)
{
  auto region = &rm.Rvsdg().GetRootRegion();
  auto ln = dynamic_cast<const rvsdg::LambdaNode *>(region->Nodes().begin().ptr());
  if (region->numNodes() == 1 && ln)
  {
    return ln;
  }
  else
  {
    throw util::Error("Root should have only one lambda node now");
  }
}

std::string
BaseHLS::get_base_file_name(const llvm::LlvmRvsdgModule & rm)
{
  auto source_file_name = rm.SourceFileName().name();
  auto base_file_name = source_file_name.substr(0, source_file_name.find_last_of('.'));
  return base_file_name;
}

}
