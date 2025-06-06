/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM_LLVM_IR_PRINT_HPP
#define JLM_JLM_LLVM_IR_PRINT_HPP

/* FIXME: I would rather like to forward declare AnnotationMap and demand_set */
#include <jlm/llvm/ir/Annotation.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

#include <string>

namespace jlm::llvm
{

class ControlFlowGraph;
class InterProceduralGraph;
class ipgraph_module;

/* control flow graph */

std::string
to_dot(const ControlFlowGraph & cfg);

static inline void
print_dot(const ControlFlowGraph & cfg, FILE * out)
{
  fputs(to_dot(cfg).c_str(), out);
  fflush(out);
}

/* inter-procedural graph */

std::string
to_str(const InterProceduralGraph & ipg);

std::string
to_dot(const InterProceduralGraph & ipg);

static inline void
print_ascii(const InterProceduralGraph & ipg, FILE * out)
{
  fputs(to_str(ipg).c_str(), out);
  fflush(out);
}

static inline void
print_dot(const InterProceduralGraph & ipg, FILE * out)
{
  fputs(to_dot(ipg).c_str(), out);
  fflush(out);
}

/* ipgraph module */

static inline std::string
to_str(const ipgraph_module & im)
{
  return to_str(im.ipgraph());
}

static inline void
print(const ipgraph_module & im, FILE * out)
{
  fputs(to_str(im).c_str(), out);
  fflush(out);
}

/* aggregation tree */

std::string
to_str(const AggregationNode & n, const AnnotationMap & dm);

static inline std::string
to_str(const AggregationNode & n)
{
  return to_str(n, {});
}

void
print(const AggregationNode & n, const AnnotationMap & dm, FILE * out);

static inline void
print(const AggregationNode & n, FILE * out)
{
  print(n, {}, out);
}

}

#endif
