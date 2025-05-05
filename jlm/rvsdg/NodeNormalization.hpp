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

class output;

template<class TOperation>
using NodeNormalization = std::function<
    std::optional<std::vector<output *>>(const TOperation &, const std::vector<output *> &)>;

template<class TOperation>
std::optional<std::vector<output *>>
NormalizeSequence(
    const std::vector<NodeNormalization<TOperation>> & nodeNormalizations,
    const TOperation & operation,
    const std::vector<output *> & operands)
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
ReduceNode(const NodeNormalization<TOperation> & nodeNormalization, SimpleNode & node)
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
