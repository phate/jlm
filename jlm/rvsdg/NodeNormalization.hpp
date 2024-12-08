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
using NodeNormalizationSequenceType = std::function<NodeNormalization<TOperation>(
    const std::vector<NodeNormalization<TOperation>> &)>;

template<class TOperation>
NodeNormalizationSequenceType<TOperation> NodeNormalizationSequence =
    [](const std::vector<NodeNormalization<TOperation>> & nodeNormalizations)
    -> NodeNormalization<TOperation>
{
  return [&](const TOperation & operation,
             const std::vector<output *> & operands) -> std::optional<std::vector<output *>>
  {
    for (auto & nodeNormalization : nodeNormalizations)
    {
      if (auto results = nodeNormalization(operation, operands))
      {
        return results;
      }
    }

    return std::nullopt;
  };
};

using NodeReduction = std::function<bool(Node &)>;

template<class TOperation>
using NodeNormalizationReductionType = std::function<NodeReduction(NodeNormalization<TOperation>)>;

template<class TOperation>
NodeNormalizationReductionType<TOperation> NodeNormalizationReduction =
    [](NodeNormalization<TOperation> nodeNormalization) -> NodeReduction
{
  return [&](Node & node) -> bool
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
  };
};

}

#endif
