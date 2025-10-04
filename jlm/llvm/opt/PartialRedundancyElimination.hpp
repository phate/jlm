/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
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

struct GVN_Hash
{
  size_t value;
  inline GVN_Hash(){this->value = 0;}
  inline GVN_Hash(size_t v){this->value = v;}
  inline static GVN_Hash None()   {return GVN_Hash(0);}
  inline static GVN_Hash Tainted(){return GVN_Hash(1);}
  inline bool IsValid(){return value >= 2;}
  inline bool IsSome(){return value != 0;}
};

/** Boiler plate for making the struct compatible with std::unordered_map **/
struct GVN_Map_Hash{
  size_t operator()(const GVN_Hash& v) const{return v.value;}
};

struct GVN_Map_Eq
{
  bool operator()(const GVN_Hash& a, const GVN_Hash& b) const{return a.value == b.value;}
};


/** \brief Partial Redundancy Elimination
 *
 * A pass for doing partial redundancy analysis and elimination
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

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  /** \brief A mapping from Input and Output to gvn hashes **/
  std::unordered_map<void* , GVN_Hash> gvn_hashes_;

  /* Debug data */
  std::unordered_map<GVN_Hash, size_t, GVN_Map_Hash, GVN_Map_Eq> dbg_hash_counts_;

  void TraverseTopDownRecursively(rvsdg::Region& reg,          void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node& node));

  static void dump_region(        PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void dump_node(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void register_leaf_hash( PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_bin(           PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_gamma(         PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_node(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_call(          PartialRedundancyElimination *pe, rvsdg::Node& node);
  static void hash_theta_pre(     PartialRedundancyElimination *pe, rvsdg::Node& node);
  //static void hash_theta_post(    PartialRedundancyElimination *pe, rvsdg::Node& ndoe);

  /**
   * Insert the hash into a map of {hash => count} for debugging purposes.
   * \ref dbg_hash_counts_
   */

  inline void DBG_CountGVN_HashCounts(GVN_Hash h)
  {
    if (dbg_hash_counts_.find(h) == dbg_hash_counts_.end()){
      dbg_hash_counts_.insert({h, 1});
    }else{
      dbg_hash_counts_[h] = dbg_hash_counts_[h] + 1;
    }
  }

inline void AssignGVN(jlm::rvsdg::Output* k, GVN_Hash h)
  {
    gvn_hashes_.insert({k, h});
    DBG_CountGVN_HashCounts(h);
  }

  inline void AssignGVN(jlm::rvsdg::Input* k, GVN_Hash h)
  {
    gvn_hashes_.insert({k, h});
    DBG_CountGVN_HashCounts(h);
  }

  /**
   * Convenience method for annotating outputs such as function arguments with hashes.
   *
   * @param out output to annotate with gvn value
   * @param base a string to base the hash on
   * @param index an index which is hashed together with the string hash
   */
  inline void AssignGVN(jlm::rvsdg::Output* out, std::string base, int index)
  {
    const std::hash<std::string> hasher;

    size_t h = hasher(base);
    h ^= index;
    gvn_hashes_.insert({out, GVN_Hash(h)});
    DBG_CountGVN_HashCounts(h);
  }

  /** Safely returns a valid hash value or None **/
  GVN_Hash GetHash(void* input_or_output)
  {
    if (gvn_hashes_.find(input_or_output) != gvn_hashes_.end()){return gvn_hashes_[input_or_output];}
    return GVN_Hash::None();
  }

  inline size_t DBG_HashCount(GVN_Hash h)
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

namespace std
{
  inline std::string to_string(jlm::llvm::GVN_Hash h)
  {
    switch (h.value)
    {
    case 0:  return std::string("none");
    case 1:  return std::string("tainted");
    default: return std::to_string(h.value);
    }
  }
}


#endif
