/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include "jlm/rvsdg/MatchType.hpp"
#include "jlm/rvsdg/traverser.hpp"
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <stdint.h>

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


namespace jlm::llvm::flows{
const size_t  PRE_PTR_TAG_OUTPUT = 0x1;
const size_t  PRE_PTR_TAG_INPUT  = 0x2;
const size_t  PRE_PTR_TAG_NODE   = 0x3;
const size_t  PRE_PTR_TAG_MASK   = 0x3;

/** \brief A pointer to either rvsdg::Node*, rvsdg::Input or rvsdg::Output using the low bits as inline tags. **/
union Flow
{
  Flow(rvsdg::Input* i){SetInput(i);}
  Flow(rvsdg::Output* o){SetOutput(o);}
  Flow(rvsdg::Node* n){SetNode(n);}

  inline void SetInput(rvsdg::Input* i)
  {
    this->input_ = reinterpret_cast<rvsdg::Input*>(reinterpret_cast<size_t>(i) | PRE_PTR_TAG_INPUT);
  }
  inline void SetOutput(rvsdg::Output* o)
  {
    this->output_ = reinterpret_cast<rvsdg::Output*>(reinterpret_cast<size_t>(o) | PRE_PTR_TAG_OUTPUT);
  }
  inline void SetNode(rvsdg::Node* n)
  {
    this->node_ = reinterpret_cast<rvsdg::Node*>(reinterpret_cast<size_t>(n) | PRE_PTR_TAG_NODE);
  }

  inline rvsdg::Input*  GetInput(){return (reinterpret_cast<size_t>(input_)   & PRE_PTR_TAG_MASK) == PRE_PTR_TAG_INPUT ? input_ : NULL;}
  inline rvsdg::Output* GetOutput(){return (reinterpret_cast<size_t>(output_) & PRE_PTR_TAG_MASK) == PRE_PTR_TAG_OUTPUT ? output_ : NULL;}
  inline rvsdg::Node*   GetNode(){return (reinterpret_cast<size_t>(output_)   & PRE_PTR_TAG_MASK) == PRE_PTR_TAG_NODE ? node_ : NULL;}
  inline void* UnsafeGetRaw() const {return static_cast<void*>(input_);}
private:
  rvsdg::Input* input_;
  rvsdg::Output* output_;
  rvsdg::Node* node_;
};

struct Flow_Hash{void* operator()(const Flow& v) const{return v.UnsafeGetRaw();}};
struct Flow_Eq{bool operator()(const Flow& a, const Flow& b) const{return a.UnsafeGetRaw() == b.UnsafeGetRaw();}};

template<typename D>
struct AnalysisData
{
  std::unordered_map<Flow, D, Flow_Hash, Flow_Eq> annot_flows;
  D Get(Output o)
  {
    Flow f(o);
    if (annot_flows.find(f) != annot_flows.end())
    {
      return annot_flows[f];
    }else{
      return D(); /** Empty value **/
    }
  }
};

/** \brief the type of flow. **/
enum class FlowType
{
  //Optionally differentiate between single and multiple cases in order to catch logic errors
  //   where a single value is expected to flow from a node, but there are multiple exits
  INPUT,
  OUTPUT,
  NODE,
  PARAMETER,
};

template<typename D>
struct FlowValue
{
  FlowType type;
  D value;
  size_t index;
  FlowValue(FlowType type, D value) : type(type), value(value), index(0){}
  FlowValue(D value, size_t index) : type(FlowType::PARAMETER),value(value), index(index){}
};

/** TODO: one data structure for storing annotations and another for storing reactive state **/
/** TODO: no need for intermediate buffers. Project flows onto edges and update reactive state as needed **/
/** Avoid duplicating data by storing maps elsewhere and provide mapping functions **/

/** \brief FlowsCtx a context used as a proxy inside graph traversers
 * D must be comparable
 * **/



/** \brief a buffer for the inputs and output flows from a node or region **/
template<typename D>
class FlowsBuffer
{
public:
  FlowsBuffer(){}
  void Resize(size_t ninputs){
    values.resize(ninputs);
    flow_direction.resize(ninputs);
  }
  void Clear(){values.clear(); values.clear();}
  std::vector<D> values;
private:
  std::vector<FlowType> flow_direction;
};

}

namespace std
{
inline std::string to_string(jlm::llvm::flows::FlowType ft)
{
  switch (ft)
  {
  case jlm::llvm::flows::FlowType::NODE:  return std::string("NODE");
  case jlm::llvm::flows::FlowType::INPUT: return std::string("INPUT");
  case jlm::llvm::flows::FlowType::OUTPUT: return std::string("OUTPUT");
  case jlm::llvm::flows::FlowType::PARAMETER: return std::string("PARAMETER");
  default: return std::string("Invalid flow type");
  }
}
}

namespace jlm::llvm::flows
{
using namespace jlm;

/** \brief reactively update annotated data for nodes and edges until a fixed point is reached **/
template<typename D, typename Fn>
void RecurseTopDownWithFlows(AnalysisData<D>& analysis, rvsdg::Region& reg, Fn cb_producer)
{
  using namespace jlm::rvsdg;

  for (Node& node : reg.TopNodes())
  {
    MatchType(node,
      /** Split and merge values for each branch **/
      [&analysis](GammaNode& gn)
      {
        FlowsBuffer<D> outputs;
        outputs.Resize(gn.noutputs() );

        FlowsBuffer<D> inputs;
        inputs.Resize(gn.ninputs());

        for (auto i = 0; i )

        for (size_t i = 0; i < gn.ninputs() ; i++)
        {
          if (gn.input(i)){inputs[i] = analysis.GetValue(gn.input(i)->origin());}
        }


        /* Split flows */
        size_t reg_count =    gn.nsubregions();
        size_t input_count =  gn.ninputs();
        size_t output_count = gn.noutputs();



      }
    );
  }
}

template<typename D, typename Fn>
void RecurseTopDown(AnalysisData<D>& analysis, rvsdg::Region& reg, Fn cb_node)
{
  for ( rvsdg::Node* node : jlm::rvsdg::TopDownTraverser(&reg) )
  {
    if (node){
      FlowValue<D> fl = cb_node(*node);
      std::cout << "The flow type is: " << std::to_string(fl.type) << std::endl;
      std::cout << std::to_string(fl.value) << std::endl;
      analysis.annot_node[node] = fl.value;
    }
  }
}
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
  inline GVN_Hash Merge(GVN_Hash& other)
  {
    if (other.IsSome())
    {
      return (this->value == other.value) ? *this : GVN_Hash::Tainted();
    }else{
      return this->IsSome() ? *this : other;
    }
  }
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
class PartialRedundancyElimination final : public jlm::rvsdg::Transformation
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
  Run(jlm::rvsdg::RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector) override;

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
  //static void hash_theta_post(    PartialRedundancyElimination *pe, rvsdg::Node& node);

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
