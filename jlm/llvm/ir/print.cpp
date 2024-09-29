/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/aggregation.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/tac.hpp>

#include <deque>

namespace jlm::llvm
{

/* string converters */

static std::string
emit_tacs(const tacsvector_t & tacs)
{
  std::string str;
  for (const auto & tac : tacs)
    str += tac::ToAscii(*tac) + ", ";

  return "[" + str + "]";
}

static std::string
emit_function_node(const ipgraph_node & clg_node)
{
  JLM_ASSERT(dynamic_cast<const function_node *>(&clg_node));
  auto & node = *static_cast<const function_node *>(&clg_node);

  const auto & fcttype = node.fcttype();

  /* convert result types */
  std::string results("<");
  for (size_t n = 0; n < fcttype.NumResults(); n++)
  {
    results += fcttype.ResultType(n).debug_string();
    if (n != fcttype.NumResults() - 1)
      results += ", ";
  }
  results += ">";

  /* convert operand types */
  std::string operands("<");
  for (size_t n = 0; n < fcttype.NumArguments(); n++)
  {
    operands += fcttype.ArgumentType(n).debug_string();
    if (n != fcttype.NumArguments() - 1)
      operands += ", ";
  }
  operands += ">";

  std::string cfg = node.cfg() ? cfg::ToAscii(*node.cfg()) : "";
  std::string exported = !is_externally_visible(node.linkage()) ? "static" : "";

  return exported + results + " " + node.name() + " " + operands + "\n{\n" + cfg + "}\n";
}

static std::string
emit_data_node(const ipgraph_node & clg_node)
{
  JLM_ASSERT(dynamic_cast<const data_node *>(&clg_node));
  auto & node = *static_cast<const data_node *>(&clg_node);
  auto init = node.initialization();

  std::string str = node.name();
  if (init)
    str += " = " + emit_tacs(init->tacs());

  return str;
}

std::string
to_str(const ipgraph & clg)
{
  static std::unordered_map<std::type_index, std::function<std::string(const ipgraph_node &)>> map(
      { { typeid(function_node), emit_function_node }, { typeid(data_node), emit_data_node } });

  std::string str;
  for (const auto & node : clg)
  {
    JLM_ASSERT(map.find(typeid(node)) != map.end());
    str += map[typeid(node)](node) + "\n";
  }

  return str;
}

/* dot converters */

static inline std::string
emit_entry_dot(const cfg_node & node)
{
  JLM_ASSERT(is<entry_node>(&node));
  auto en = static_cast<const entry_node *>(&node);

  std::string str;
  for (size_t n = 0; n < en->narguments(); n++)
  {
    auto argument = en->argument(n);
    str += "<" + argument->type().debug_string() + "> " + argument->name() + "\\n";
  }

  return str;
}

static inline std::string
emit_exit_dot(const cfg_node & node)
{
  JLM_ASSERT(is<exit_node>(&node));
  auto xn = static_cast<const exit_node *>(&node);

  std::string str;
  for (size_t n = 0; n < xn->nresults(); n++)
  {
    auto result = xn->result(n);
    str += "<" + result->type().debug_string() + "> " + result->name() + "\\n";
  }

  return str;
}

static inline std::string
emit_basic_block(const cfg_node & node)
{
  JLM_ASSERT(is<basic_block>(&node));
  auto & tacs = static_cast<const basic_block *>(&node)->tacs();

  std::string str;
  for (const auto & tac : tacs)
    str += tac::ToAscii(*tac) + "\\n";

  return str;
}

static inline std::string
emit_header(const cfg_node & node)
{
  if (is<entry_node>(&node))
    return "ENTRY";

  if (is<exit_node>(&node))
    return "EXIT";

  return util::strfmt(&node);
}

static inline std::string
emit_node(const cfg_node & node)
{
  static std::unordered_map<std::type_index, std::string (*)(const cfg_node &)> map(
      { { typeid(entry_node), emit_entry_dot },
        { typeid(exit_node), emit_exit_dot },
        { typeid(basic_block), emit_basic_block } });

  JLM_ASSERT(map.find(typeid(node)) != map.end());
  std::string body = map[typeid(node)](node);

  return emit_header(node) + "\\n" + body;
}

std::string
to_dot(const llvm::cfg & cfg)
{
  auto entry = cfg.entry();
  auto exit = cfg.exit();

  std::string dot("digraph cfg {\n");

  /* emit entry node */
  dot += util::strfmt(
      "{ rank = source; ",
      (intptr_t)entry,
      "[shape=box, label = \"",
      emit_node(*entry),
      "\"]; }\n");
  dot += util::strfmt(
      (intptr_t)entry,
      " -> ",
      (intptr_t)entry->outedge(0)->sink(),
      "[label=\"0\"];\n");

  /* emit exit node */
  dot += util::strfmt(
      "{ rank = sink; ",
      (intptr_t)exit,
      "[shape=box, label = \"",
      emit_node(*exit),
      "\"]; }\n");

  for (const auto & node : cfg)
  {
    dot += util::strfmt("{", (intptr_t)&node);
    dot += util::strfmt("[shape = box, label = \"", emit_node(node), "\"]; }\n");
    for (auto it = node.begin_outedges(); it != node.end_outedges(); it++)
    {
      dot += util::strfmt((intptr_t)it->source(), " -> ", (intptr_t)it->sink());
      dot += util::strfmt("[label = \"", it->index(), "\"];\n");
    }
  }
  dot += "}\n";

  return dot;
}

std::string
to_dot(const ipgraph & clg)
{
  std::string dot("digraph clg {\n");
  for (const auto & node : clg)
  {
    dot += util::strfmt((intptr_t)&node);
    dot += util::strfmt("[label = \"", node.name(), "\"];\n");

    for (const auto & call : node)
      dot += util::strfmt((intptr_t)&node, " -> ", (intptr_t)call, ";\n");
  }
  dot += "}\n";

  return dot;
}

/* aggregation node */

std::string
to_str(const aggnode & n, const AnnotationMap & dm)
{
  std::function<std::string(const aggnode &, size_t)> f = [&](const aggnode & n, size_t depth)
  {
    std::string subtree(depth, '-');
    subtree += n.debug_string();

    if (dm.Contains(n))
      subtree += " " + dm.Lookup<AnnotationSet>(n).DebugString() + "\n";

    for (const auto & child : n)
      subtree += f(child, depth + 1);

    return subtree;
  };

  return f(n, 0);
}

void
print(const aggnode & n, const AnnotationMap & dm, FILE * out)
{
  fputs(to_str(n, dm).c_str(), out);
  fflush(out);
}

}
