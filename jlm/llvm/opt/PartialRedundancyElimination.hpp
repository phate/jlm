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
  std::unordered_map<jlm::rvsdg::Output*, size_t> gvn_hashes_;

  /* Debug data */
  std::unordered_map<size_t, size_t> dbg_hash_counts_;

  void TraverseTopDownRecursively(rvsdg::Region& reg,          void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node& node));

  static void dump_region(        PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void dump_node(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void register_leaf_hash( PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_bin(           PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_gamma(         PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_node(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_call(          PartialRedundancyElimination *pe, rvsdg::Node& node);

  /**
   * Insert the hash into a map of {hash => count} for debugging purposes.
   * \link dbg_hash_counts_
   */

  inline void register_hash(size_t h)
  {
    if (dbg_hash_counts_.find(h) == dbg_hash_counts_.end())
    {
      dbg_hash_counts_.insert({h, 1});
    }else
    {
      dbg_hash_counts_[h] = dbg_hash_counts_[h] + 1;
    }
  }

inline void register_hash(jlm::rvsdg::Output* k, size_t h)
  {
    gvn_hashes_.insert({k, h});
    register_hash(h);
  }

  /**
   * Convenience method for annotating outputs such as function arguments with hashes.
   *
   * @param out output to annotate with gvn value
   * @param base a string to base the hash on
   * @param index an index which is hashed together with the string hash
   */
  inline void register_hash(jlm::rvsdg::Output* out, std::string base, int index)
  {
    const std::hash<std::string> hasher;

    size_t h = hasher(base);
    h ^= index;
    gvn_hashes_.insert({out, h});
    register_hash(h);
  }

  inline bool OutputHasHash(rvsdg::Output* out){return gvn_hashes_.find(out) != gvn_hashes_.end();}

  inline size_t DBG_HashCount(size_t h)
  {
    if (dbg_hash_counts_.find(h) != dbg_hash_counts_.end())
    {
      return dbg_hash_counts_[h];
    }else
    {
      return 0;
    }
  }

};

}

#endif
