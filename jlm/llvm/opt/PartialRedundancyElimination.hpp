/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include "jlm/rvsdg/MatchType.hpp"
#include "jlm/rvsdg/traverser.hpp"
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/delta.hpp>
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

/*
const size_t  PRE_PTR_TAG_OUTPUT = 0x1;
const size_t  PRE_PTR_TAG_INPUT  = 0x2;
const size_t  PRE_PTR_TAG_NODE   = 0x3;
const size_t  PRE_PTR_TAG_MASK   = 0x3;
*/
/** \brief A pointer to either rvsdg::Node*, rvsdg::Input or rvsdg::Output using the low bits as inline tags. **/
/*union Flow
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

*/

enum class FlowType
{
  //Optionally differentiate between single and multiple cases in order to catch logic errors
  //   where a single value is expected to flow from a node, but there are multiple exits
  INPUT,
  OUTPUT,
  NODE,
  PARAMETER,
};




/** TODO: one data structure for storing annotations and another for storing reactive state **/
/** TODO: no need for intermediate buffers. Project flows onto edges and update reactive state as needed **/
/** Avoid duplicating data by storing maps elsewhere and provide mapping functions **/

/** \brief FlowsCtx a context used as a proxy inside graph traversers
 * D must be comparable
 * **/

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

  /** A view into the data to manipulate by flows **/
  template<typename D>
  class FlowData
  {
  public:
    FlowData(std::unordered_map<void*, D> * output_values)
    {
      JLM_ASSERT(output_values);   //should be non-null
      output_values_ = output_values;
    }
    std::optional<D> Get(rvsdg::Output* k)
    {
      if (k == NULL){return std::nullopt;}
      bool present = output_values_->find(k) == output_values_->end();
      return present ? std::optional<D>((*output_values_)[k]) : std::nullopt;
    }
    void Set(rvsdg::Output* k, std::optional<D> v)
    {
      if (v){
        output_values_->insert({k,*v});
      }
    }
  private:
    std::unordered_map<void*, D>* output_values_;
  };

  enum class WorkItemType
  {
    REGION,
    DELTA,
    NODE,
    GAMMA,
    GAMMA_END,
    THETA,
    THETA_END,
    LAMBDA,
  };
  struct WorkItemValue
  {
    // implicit conversions are ok here
    WorkItemValue(rvsdg::Region* r)                 {this->type = WorkItemType::REGION; this->region = r;}
    WorkItemValue(WorkItemType type, rvsdg::Node* n){this->type = type;                 this->node = n;}
    WorkItemValue(rvsdg::Node* n)
    {
      this->node = n;
      this->type = WorkItemType::NODE;
      rvsdg::MatchType(*n,
        [this](jlm::rvsdg::GammaNode& gn){this->type = WorkItemType::GAMMA;},
        [this](jlm::rvsdg::ThetaNode& tn){this->type = WorkItemType::THETA;},
        [this](jlm::rvsdg::LambdaNode& lm)     {this->type = WorkItemType::LAMBDA;},
        [this](jlm::rvsdg::DeltaNode& dl)     {this->type = WorkItemType::DELTA;}
      );
    }

    WorkItemType type;
    union{
      rvsdg::DeltaNode* dl;
      rvsdg::Region* region;
      rvsdg::Node* node;
      rvsdg::GammaNode* gn;
      rvsdg::ThetaNode* tn;
      rvsdg::LambdaNode* lm;
    };
  };

  /** \brief abstracts away the walking of the rvsdg graph for data flows of a reactive nature
   *  Theta and Gamma nodes have special semantics as they must sometimes merge values.
   *  Merges at gamma and theta nodes combine flows.
   *  The merge at theta nodes represent values with multiple alternatives from
   *    different loop iterations.
   *  Merges at gamma nodes represent the combined value from switch cases
   *  The merges may be different or the same
   * */
  template<typename D, typename GaMerger, typename ThMerger, typename Prod>
  void ApplyDataFlowsTopDown(rvsdg::Region& scope, FlowData<D>& fd, GaMerger mrGa, ThMerger mrTh,  Prod cb){
    // mrGa: represent the intersection of values from one data flow out from a gamma node
    // mrTh: represent the merging of output of theta node with the input
    //
    // A queue of nodes and regions to visit or equivalently a continuation
    //    of instruction to be executed by the interpreter below.
    std::vector<WorkItemValue> workItems;
    // A buffer for flow values.
    // The flows function handles lookup of values from the fd map.
    std::vector< std::optional<D> > flows_in;
    std::vector< std::optional<D> > flows_out;

    workItems.push_back(WorkItemValue(&scope));
    size_t max_iter = 500;
    while (workItems.size() && max_iter)
    {
      max_iter--;
      auto w = workItems.back();  workItems.pop_back();
      switch (w.type)
      {
        case WorkItemType::DELTA:{
          for (auto& reg : w.lm->Subregions()){
            workItems.push_back(WorkItemValue(&reg));
          }
          std::cout << "WL:DELTA"<<std::endl;
          if (workItems.size() > 1000){std::cout<<"Stack overflow" << std::endl; return;}

        }break;

        case WorkItemType::REGION:{
          std::cout << "WL:REGION"<<std::endl;
          // Push all nodes inside a region in topological order onto the queue
          std::vector<WorkItemValue> tmp;
          for (auto node : rvsdg::TopDownTraverser(w.region)){tmp.push_back(WorkItemValue(node));}
          while (tmp.size()){workItems.push_back(tmp.back()); tmp.pop_back();}
        }break;
        case WorkItemType::NODE:{
          std::cout << "WL:NODE:"<<w.node->DebugString()<< std::endl;
          // initialize input buffer
          flows_in.clear();
          for (size_t i=0;i < w.node->ninputs() ; i++){
            auto ni = w.node->input(i);  //check just in case Input is NULL
            std::optional<D> val = ni ? fd.Get(ni->origin()) : std::nullopt;
            flows_in.push_back(std::optional<D>(val));
          }
          // initialize output buffer
          flows_out.clear();
          flows_out.resize(w.node->noutputs(), std::nullopt);

          // visit node
          cb(*(w.node), flows_in, flows_out);

          // update map
          for ( size_t i = 0; i < w.node->noutputs(); i++){
            fd.Set(w.node->output(i), flows_out[i]);
          }
          //DEBUG
          for (size_t i = 0; i < flows_out.size(); i++){
            if (flows_out[i]){
              std::cout << "Flow out ["<<i<<"] : " << std::endl;
            }
          }
        }break;
        case WorkItemType::LAMBDA:{
          // This case only handles params out
          // Add an enum for visit type later
          // initialize input buffer
          flows_in.clear();
          auto f_args = w.lm->GetFunctionArguments();
          flows_out.clear(); flows_out.resize(f_args.size(), std::nullopt);

          // visit node
          cb(*(w.node), flows_in, flows_out);

          // update map
          for ( size_t i = 0; i < f_args.size(); i++){
            fd.Set(f_args[i], flows_out[i]);
          }
          //DEBUG
          for (size_t i = 0; i < flows_out.size(); i++){
            if (flows_out[i]){
              std::cout << "Lambda: Flow out [" << i << "] : " << std::endl;
            }
          }
          // Todododo : visit lambda after body has been visited once.

          // Finally iterate over lambda body
          workItems.push_back(w.lm->subregion());
        }break;

        case WorkItemType::GAMMA:
        {
          //Split flows
          for (auto ev : w.gn->GetEntryVars()){
            auto tmp = ev.input;
            auto flow_from = tmp ? fd.Get(tmp->origin()) : std::nullopt;
            if (flow_from){
              for (rvsdg::Output* brarg : ev.branchArgument){
                fd.Set(brarg, *flow_from);
              }
            }
          }
          //Push tasks in LIFO order
          workItems.push_back(WorkItemValue(WorkItemType::GAMMA_END, w.gn));
          for (size_t i = 0; i < w.gn->nsubregions() ; i++){
            workItems.push_back(w.gn->subregion(i));
          }
        }break;
        case WorkItemType::GAMMA_END:{
          // Reduce all outputs from exitVars with mrGa
          auto ex_vars = w.gn->GetExitVars();
          JLM_ASSERT(ex_vars.size() == w.gn->noutputs());
          for (size_t v = 0; v < ex_vars.size(); v++){
            auto br_fst = ex_vars[v].branchResult[0];
            auto merged_val = fd.Get( br_fst ? br_fst->origin() : NULL );

            for (size_t b = 1;b < ex_vars[v].branchResult.size();b++){  // !!! from 1
              auto next_br = ex_vars[v].branchResult[b];
              auto next_br_value = fd.Get( next_br ? next_br->origin() : NULL );
              merged_val = mrGa(merged_val, next_br_value);
            }

            fd.Set(ex_vars[v].output, merged_val);
            JLM_ASSERT(ex_vars.size() == w.gn->noutputs());
            JLM_ASSERT(ex_vars[v].output == w.gn->output(v));
          }
        }break;

        case WorkItemType::THETA:{
          // At entry into a theta node check if the flow into each loop variable
          // is compatible with the last output previous loop iterations
          // This ensures theta bodies are only visited twice during GVN
          // It is the responsibility of merge callbacks to ensure values
          //    reach a fixpoint.
          auto loopvars = w.tn->GetLoopVars();
          bool fixed_point_reached = true;

          for (size_t i = 0;i < loopvars.size(); i++){
            auto lv = loopvars[i];
            auto lv_input = fd.Get( lv.input ? lv.input->origin() : NULL );

            auto lv_post = fd.Get( lv.post ? lv.post->origin() : NULL );
            auto merged = mrTh( lv_input, lv_post );

            auto lv_pre = fd.Get( lv.pre );
            if (
              (merged && !lv_pre) || (!merged && lv_pre) || (*merged != *lv_pre)
            ){fixed_point_reached = false;}

            fd.Set( lv.pre, merged );
          }

          if (!fixed_point_reached){
            workItems.push_back( WorkItemValue(WorkItemType::THETA_END, w.tn) );
            workItems.push_back( w.tn->subregion() );
          }
        }break;

        case WorkItemType::THETA_END:{
          auto loopvars = w.tn->GetLoopVars();
          for (size_t i = 0;i < loopvars.size(); i++){
            auto lv = loopvars[i];
            fd.Set( lv.output, fd.Get( lv.post ? lv.post->origin() : NULL ) );   // Required for downstream nodes
          }
          workItems.push_back( WorkItemValue(w.tn) ); // Attempt another loop iteration
        }break;

        default: std::cout << static_cast<int>(w.type) <<"Ignoring work item..."<<std::endl;
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
  inline bool operator==(const GVN_Hash &other){return this->value == other.value;}
  inline bool operator!=(const GVN_Hash &other){return this->value != other.value;}
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
