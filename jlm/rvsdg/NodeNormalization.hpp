/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NODENORMALIZATION_HPP
#define JLM_RVSDG_NODENORMALIZATION_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/util/common.hpp>

#include <functional>
#include <optional>
#include <vector>

namespace jlm::rvsdg
{

class Output;

template<class TOperation>
using NodeNormalization = std::function<
    std::optional<std::vector<Output *>>(const TOperation &, const std::vector<Output *> &)>;

template<class TOperation>
std::optional<std::vector<Output *>>
NormalizeSequence(
    const std::vector<NodeNormalization<TOperation>> & nodeNormalizations,
    const TOperation & operation,
    const std::vector<Output *> & operands)
{
  for (auto & nodeNormalization : nodeNormalizations)
  {
    if (auto results = nodeNormalization(operation, operands))
    {
      return results;
    }
  }

  return std::nullopt;
}

template<class TOperation>
bool
ReduceNode(const NodeNormalization<TOperation> & nodeNormalization, Node & node)
{
  auto operation = util::AssertedCast<const TOperation>(&node.GetOperation());
  auto operands = rvsdg::operands(&node);

  if (auto results = nodeNormalization(*operation, operands))
  {
    divert_users(&node, *results);
    remove(&node);
    return true;
  }

  return false;
}

}

#endif
