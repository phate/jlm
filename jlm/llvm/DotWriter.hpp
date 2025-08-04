/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_DOTWRITER_HPP
#define JLM_LLVM_DOTWRITER_HPP

#include <jlm/rvsdg/region.hpp>
#include <jlm/util/GraphWriter.hpp>

namespace jlm::llvm::dot
{
/**
 * Recursively converts a region and all sub-regions into graphs and sub-graphs.
 * All nodes in each region become InOutNodes, with edges showing data and state dependencies.
 * Arguments and results are represented using ArgumentNode and ResultNode, respectively.
 * All created nodes, inputs, and outputs, get associated to the rvsdg nodes, inputs and outputs.
 *
 * @param writer the GraphWriter to use
 * @param region the RVSDG region to recursively traverse
 * @param emitTypeGraph if true, an additional graph containing nodes for all types is emitted
 * @return a reference to the top-level graph corresponding to the region
 */
util::graph::Graph &
WriteGraphs(util::graph::Writer & writer, rvsdg::Region & region, bool emitTypeGraph);
}

#endif // JLM_LLVM_DOTWRITER_HPP
