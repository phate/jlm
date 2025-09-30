/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <stdint.h>
#include <jlm/rvsdg/node.hpp>

namespace jlm::rvsdg
{
class DeltaNode;
class GammaNode;
class Graph;
class LambdaNode;
class Output;
class StructuralNode;
class ThetaNode;
class Region;
}

namespace jlm::llvm
{

/** \brief Partial Redundancy Elimination
 *
 * Todo: description here
 *
 *
 */
class PartialRedundancyElimination final : public rvsdg::Transformation
{
  class Context;
  class Statistics;

public:
  ~PartialRedundancyElimination() noexcept override;

  PartialRedundancyElimination();

  PartialRedundancyElimination(const PartialRedundancyElimination &) = delete;
  PartialRedundancyElimination(PartialRedundancyElimination &&) = delete;

  PartialRedundancyElimination &
  operator=(const PartialRedundancyElimination &) = delete;
  PartialRedundancyElimination &
  operator=(PartialRedundancyElimination &&) = delete;

  /*void
  run(rvsdg::Region & region);*/

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:


  void TraverseSubRegions(rvsdg::Region& reg,          void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node& node));

  static void dump_region(        PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void dump_node(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void register_leaf_hash( PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_bin(           PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_gamma(         PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_node(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  //static void hash_call(          PartialRedundancyElimination *pe, rvsdg::Node& node);

  inline void register_hash(jlm::rvsdg::Output* k, size_t h)
  {
    output_hashes.insert({k, h});
    register_hash(h);
  }

  inline void register_hash_for_output(jlm::rvsdg::Output* out, std::string base, int index)
  {
    const std::hash<std::string> hasher;

    size_t h = hasher(base);
    h ^= index;
    output_hashes.insert({out, h});
    register_hash(h);
  }

  //todo rename to gvn_hashes
  std::unordered_map<jlm::rvsdg::Output*, size_t> output_hashes;

  /* Debug data */
  std::unordered_map<size_t, size_t> hash_counts;

  inline bool output_has_hash(rvsdg::Output* out){return output_hashes.find(out) != output_hashes.end();}

  inline void register_hash(size_t h)
  {
    if (hash_counts.find(h) == hash_counts.end())
    {
      hash_counts.insert({h, 1});
    }else
    {
      hash_counts[h] = hash_counts[h] + 1;
    }
  }

  inline size_t hash_count(size_t h)
  {
    if (hash_counts.find(h) != hash_counts.end())
    {
      return hash_counts[h];
    }else
    {
      return 0;
    }
  }

};

}

#endif
