/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include "jlm/llvm/ir/operators/IntegerOperations.hpp"
#include "jlm/rvsdg/MatchType.hpp"
#include "jlm/rvsdg/traverser.hpp"
#include <jlm/rvsdg/delta.hpp>
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

namespace jlm::llvm::flows
{
  /** Generic functions for some data flow analyses.
   *  Nodes are treated as part of a reactive flow network.
   *  This makes it possible to write more generic analyses as some common tasks
   *  such as ensuring the data is passed to all downstream (or upstream) usage sites.
   *  Currently only a top-down flow is implemented. Usable for GVN.
   */
  using namespace jlm;

  /** A view into the data to manipulate by flows **/
  /** This represents the value output from nodes **/
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
      bool present = output_values_->find(k) != output_values_->end();
      return present ? std::optional<D>((*output_values_)[k]) : std::nullopt;
    }
    void Set(rvsdg::Output* k, std::optional<D> v)
    {
      output_values_->insert({k,*v});
    }
  private:
    std::unordered_map<void*, D>* output_values_;
  };

  /** \ref WorkItemType    Union type tag.
   *  \ref WorkItemValue   Used internally by the reactive interpreter to keep track of nodes and regions yet to be visited.
   *  **/
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
    /* Fields should be made const */
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
   *
   *  The style of traversing the graph can be adapted to more complex cases.
   * */
  template<typename D, typename GaMerger, typename ThMerger, typename Prod>
  void ApplyDataFlowsTopDown(rvsdg::Region& scope, FlowData<D>& fd, GaMerger mrGa, ThMerger mrTh,  Prod cb){
    // mrGa: represent the intersection of values from one data flow out from a gamma node
    // mrTh: represent the merging of output of theta node with the input. Third arg must be non-null if second is something.
    //       args :  theta node
    //               value from older iteration
    //               value from last iteration
    //
    //
    // A FIFO queue of nodes and regions to visit or equivalently a continuation
    //    of instruction to be executed by the interpreter below.
    std::vector<WorkItemValue> workItems;

    // A buffer for flow values.
    // The flows function handles lookup of values from the fd map.
    std::vector< std::optional<D> > flows_in;
    std::vector< std::optional<D> > flows_out;

    workItems.push_back(WorkItemValue(&scope));
    size_t max_iter = 500;
    while (workItems.size() )
    {
      max_iter--;  if (!max_iter){std::cout<<"ApplyDataFlowsTopDownMaximum iteration count reached"<<std::endl; return;}
      auto w = workItems.back();  workItems.pop_back();

      switch (w.type){
        case WorkItemType::DELTA:{
          for (auto& reg : w.lm->Subregions()){
            workItems.push_back(WorkItemValue(&reg));
          }
          if (workItems.size() > 1000){std::cout<<"Stack overflow" << std::endl; return;}

        }break;

        case WorkItemType::REGION:{
          // Push all nodes inside a region in topological order onto the queue
          std::vector<WorkItemValue> tmp;
          for (auto node : rvsdg::TopDownTraverser(w.region)){tmp.push_back(WorkItemValue(node));}
          while (tmp.size()){
            workItems.push_back(tmp.back());
            tmp.pop_back();
          }
        }break;
        case WorkItemType::NODE:{
          // initialize input buffer
          flows_in.resize(w.node->ninputs(), std::nullopt);

          for (size_t i=0;i < w.node->ninputs() ; i++){
            auto ni = w.node->input(i);  //check just in case Input is NULL
            auto v = fd.Get(ni ? ni->origin() : NULL);
            flows_in[i] = v;
          }
          // initialize output buffer
          flows_out.clear();
          flows_out.resize(w.node->noutputs(), std::nullopt);

          // visit node
          if (flows_out.size()){
            cb(*(w.node), flows_in, flows_out);
          }

          // update map
          for ( size_t i = 0; i < w.node->noutputs(); i++){
            fd.Set(w.node->output(i), flows_out[i]);
          }
        }break;
        case WorkItemType::LAMBDA:{
          // This case only handles params out
          // Add an enum for other visit types later
          // initialize input buffer
          flows_in.clear();
          auto f_args = w.lm->GetFunctionArguments();
          flows_out.clear(); flows_out.resize(f_args.size(), std::nullopt);

          // visit node
          if (flows_out.size()){  cb(*(w.node), flows_in, flows_out);  }

          // update map
          for ( size_t i = 0; i < f_args.size(); i++){
            //if (!flows_out[i]){std::cout << "LAM MISSING OUT" << std::endl;}
            //std::cout << "lambda: Flow out ["<<i<<"] : " << std::endl;
            fd.Set(f_args[i], flows_out[i]);
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

          /* Try to push inputs into the loop */
          /* values from previous iterations stored on Output* of .pre field */
          for (size_t i = 0;i < loopvars.size(); i++){
            auto lv = loopvars[i];
            JLM_ASSERT(lv.input->origin() != lv.pre);

            auto lv_input = fd.Get( lv.input ? lv.input->origin() : NULL );
            auto lv_pre   = fd.Get( lv.pre );
            auto merged = mrTh( *(w.tn), lv_pre, lv_input );

            if (
              (merged && !lv_pre) || (!merged && lv_pre) || (*merged != *lv_pre)
            ){fixed_point_reached = false;}

            fd.Set( lv.pre, merged );
          }

          if (!fixed_point_reached){
            workItems.push_back( WorkItemValue(WorkItemType::THETA_END, w.node) );
            workItems.push_back( w.tn->subregion() );
          }
        }break;

        case WorkItemType::THETA_END:{
          auto loopvars = w.tn->GetLoopVars();
          for (size_t i = 0;i < loopvars.size(); i++){
            auto lv = loopvars[i];
            auto lv_pre  = fd.Get( lv.pre );
            auto lv_post = fd.Get( lv.post ? lv.post->origin() : NULL );
            auto merged = mrTh(*(w.tn), lv_pre, lv_post );

            fd.Set( lv.output,  lv_post );   // Required for downstream nodes
            fd.Set( lv.pre,  merged);        // Required for blocking new iterations
          }
          workItems.push_back( WorkItemValue(w.node) ); // Attempt another loop iteration
        }break;

        default: std::cout << static_cast<int>(w.type) <<"Ignoring work item..."<<std::endl;
      }
    }
  }

}

namespace jlm::llvm
{

typedef size_t GVN_Val;

/** \brief A collection of all data required to compute a gvn value for an Output* */
struct GVN_Deps{
  GVN_Deps() : op(nullptr), producers(), literal(0), has_literal(false){}
  GVN_Deps(const GVN_Deps& from)
  {
    std::cout << "CP called" << std::endl;
    op = from.op;
    producers = from.producers;
    literal = from.literal;
    has_literal = from.has_literal;
  }
  rvsdg::Operation* op;   // Some edges have
  std::vector<rvsdg::Output*> producers;
  GVN_Val literal;
  bool has_literal;
  void reset(){
    op = nullptr;
    producers.clear();
    literal = 0;
    has_literal = false;
  }
};

class GVN_Manager
{
  /** \brief A class for creating GVN hashes and keeping track of GVN values already in use. */

public:
  GVN_Manager() = default;
  /** \brief GVN values are created using a simple builder pattern. Remember to call End().*/
  /** \param output : The output edge the GVN value applies to. */
  /** \param op : The operator which produced the value. */
  inline GVN_Manager* Start(rvsdg::Output* output, rvsdg::Operation* op){
    CheckDepsEmpty(); //was the last GVN value completed?
    if (!output){throw std::runtime_error("Output pointer was null");}
    build_output_ = output;
    build_deps_.op = op;

    if (op_to_gvn_.find(op) == op_to_gvn_.end()){
      op_to_gvn_.insert({op, CreateUniqueGVN()});
    }

    return this;
  }

  inline GVN_Manager* WithEdge(rvsdg::Output* source){
    build_deps_.producers.push_back(source);
    return this;
  }

  inline GVN_Manager* WithStr(std::string& str){
    CheckLitEmpty();
    build_deps_.has_literal = true;
    bool not_created = lit_to_gvn_.find(str) == lit_to_gvn_.end();
    if (not_created){lit_to_gvn_.insert( {str, CreateUniqueGVN() });}
    build_deps_.literal = lit_to_gvn_[str];
    return this;
  }

  inline GVN_Manager* WithIndex(size_t index){
    CheckLitEmpty();
    build_deps_.has_literal = true;
    bool not_created = index_to_gvn_.find(index) == index_to_gvn_.end();
    if (not_created){index_to_gvn_.insert( {index, CreateUniqueGVN() });}
    build_deps_.literal = index_to_gvn_[index];
    return this;
  }

  inline void End(){
    // This method is the only place where non-literal nodes should allocate a new gvn value.
    auto fresh_value = HashDeps(build_deps_);
    if (!fresh_value){
      // The value either has missing dependencies or has been explicitly marked as unique.
      fresh_value = std::optional<GVN_Val>(CreateUniqueGVN());
    }

    build_deps_.reset();
    return;

    std::cout << "GVN:" << std::endl;
    std::cout << "   op: " << build_deps_.op->debug_string() << std::endl;
    std::cout << "   literal: " << build_deps_.literal << std::endl;
    std::cout << ".........................." << std::endl;
/*
    //Check if gvn value already exists
    //  if this is the case compare it with existing values
    if (occurrences_.find(*fresh_value) != occurrences_.end()){
      if (!CompareDeps(gvn_to_deps_[*fresh_value], build_deps_)){
        std::cout << "COLLISION DETECTED" << std::endl;
        fresh_value = CreateUniqueGVN();
        gvn_to_deps_.insert({*fresh_value, build_deps_});
      }
    }else{
      std::cout << "CP?? [[";
      gvn_to_deps_.insert({*fresh_value, build_deps_});
      std::cout  << "]]";
    }
*/
    if (occurrences_.find(*fresh_value) == occurrences_.end()){
      occurrences_.insert(*fresh_value);
    }

    edges_to_gvn_.insert({build_output_, *fresh_value});

    build_deps_.reset();
  }

  /** \brief Extend the flow of a value into a structural node */
  inline void ExtendFlow(rvsdg::Output* from, rvsdg::Output* downto)
  {
    if (traceback_flows_.find(downto) != traceback_flows_.end()){throw std::runtime_error("Incorrect extension of flow. Cannot have two origins.");}
    //This map can also be used when redirecting to available expressions in outer scopes.
    traceback_flows_.insert({downto, from});
  }
  inline void ExtendFlow(rvsdg::Input* from, rvsdg::Output* downto)
  {
    if (!from || !from->origin()){throw std::runtime_error("Input lacks source");}
    ExtendFlow(from->origin(), downto);
  }

  std::optional<GVN_Val> GetGVN(rvsdg::Output* output)
  {
    if (edges_to_gvn_.find(output) != edges_to_gvn_.end()){return edges_to_gvn_[output];}
    return std::nullopt;
  }

private:
  // Multiple output edges may map to the same gvn

  std::unordered_map<rvsdg::Output*,    GVN_Val>    edges_to_gvn_;
  std::unordered_map<GVN_Val,           GVN_Deps>   gvn_to_deps_; // For collision detection.
  std::unordered_map<std::string,       GVN_Val>    lit_to_gvn_;
  std::unordered_map<std::size_t,       GVN_Val>    index_to_gvn_;
  std::unordered_map<rvsdg::Operation*, GVN_Val>    op_to_gvn_;

  std::unordered_map<rvsdg::Output*, rvsdg::Output*> traceback_flows_;

  std::unordered_set<GVN_Val> occurrences_;

  /* ********************************************************************** */
  GVN_Val CreateUniqueGVN()
  {
    // All GVN values start here
    GVN_Val v = random();
    while (occurrences_.count(v) != 0){v = random();}
    occurrences_.insert(v);
    return v;
  }

  /* ********************************************************************** */

  bool CompareDeps(GVN_Deps& a, GVN_Deps& b)
  {
    // Strict structural equality
    bool m_lit = a.literal == b.literal && a.has_literal == b.has_literal;
    bool m_op = a.op == b.op;
    bool m_prods = false;
    if (a.producers.size() == b.producers.size()){
      m_prods = true;
      for (size_t i = 0; i < a.producers.size(); i++){
        auto ea = ResolveEdge(a.producers[i]);
        auto eb = ResolveEdge(b.producers[i]);
        if (!ea){m_prods = false;break;}
        if (!eb){m_prods = false;break;}
        if (*ea != *eb){m_prods = false;break;}
      }
    }
    return m_lit && m_op && m_prods;
  }
  // GVN values are computed one at the time using
  // the builder interface above.
  // These fields hold the data required
  GVN_Deps       build_deps_;
  rvsdg::Output* build_output_;

  static bool CanHashAsAssociativeCumulativeOp(const GVN_Deps& deps)
  {
    bool hash_as_ca = false;
    MatchType(*(deps.op), [&hash_as_ca](const rvsdg::BinaryOperation& bin_op){
      hash_as_ca = bin_op.is_associative() && bin_op.is_commutative();
    });
    return hash_as_ca;
  }

  bool DepsAreEquivalent(const GVN_Deps& a, const GVN_Deps& b)
  {
    // Expensive check for detecting collisions
    if (a.op != b.op){return false;}
    if (a.literal != b.literal){return false;}

    //TODO: handle ca and gamma here
    if (a.producers.size() != b.producers.size()){return false;}

    for (size_t i = 0; i < a.producers.size(); i++){
      auto g_a = ResolveEdge(a.producers[i]);
      auto g_b = ResolveEdge(b.producers[i]);
      if (g_a != g_b){return false;}
    }

    return true;
  }

  std::optional<GVN_Val> HashDeps(GVN_Deps& deps)
  {
    GVN_Val h = op_to_gvn_[deps.op] ^ deps.literal;
    if (!deps.has_literal && !deps.producers.size()){
      throw std::runtime_error("Logic error: missing data sources for hashing.");
    }
    if (CanHashAsAssociativeCumulativeOp(deps)){
      for (size_t i = 0 ; i < deps.producers.size() ; i++){
        auto g = ResolveEdge(deps.producers[i]);
        if (!g){ return std::nullopt; }
        size_t a = static_cast<size_t>(*g);
        h ^= a;                                // independent of order
      }
      return std::optional<GVN_Val>(h);
    }

    // Default hashing, in order with each edge treated differently based on order
    for (size_t i = 0 ; i < deps.producers.size() ; i++){
      auto g = ResolveEdge(deps.producers[i]);
      if (!g){ return std::nullopt; }
      size_t a = static_cast<size_t>(*g);
      h ^= a * (i+1) + a;                     // dependent on order
    }

    return std::optional<GVN_Val>(h);
  }

  std::optional<GVN_Val> ResolveEdge(rvsdg::Output* producer){
    //trace the edge all the way back to the top, crossing into lambdas(context vars), gammas and theta nodes
    while (traceback_flows_.find(producer) != traceback_flows_.end()){
      producer = traceback_flows_[producer];
    }
    if (edges_to_gvn_.find(producer) == edges_to_gvn_.end()){return std::nullopt;}
    return edges_to_gvn_[producer];
  }

  void CheckLitEmpty() const {if (build_deps_.has_literal){throw std::runtime_error("Maximum one literal supported per Output*");}}
  void CheckDepsEmpty() const {if (build_deps_.op){throw std::runtime_error("Previous GVN value not flushed.");}}
};


};

namespace jlm::llvm{

struct GVN_Hash
{
#define GVN_LV_BIT 0x1000
  size_t value;
  inline GVN_Hash(){this->value = 0;}
  inline GVN_Hash(size_t v){this->value = v &(~GVN_LV_BIT);}
  inline static GVN_Hash LoopVar(size_t v){auto h = GVN_Hash(v); h.value |= GVN_LV_BIT;
    std::cout << "LP" << h.value << " !! " << h.IsLoopVar() << std::endl;
    return h;
  }
  inline bool IsLoopVar() const {return value & GVN_LV_BIT;}
  inline static GVN_Hash None()   {return GVN_Hash(0);}
  inline static GVN_Hash Tainted(){return GVN_Hash(1);}
  inline bool IsValid(){return value >= 2;}
  inline bool IsSome(){return value != 0;}
  inline bool operator==(const GVN_Hash &other){return this->value == other.value;}
  inline bool operator!=(const GVN_Hash &other){return this->value != other.value;}
  #undef GVN_LV_BIT
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

  GVN_Manager gvn_man_;

  void TraverseTopDownRecursively(rvsdg::Region& reg,          void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node& node));
  void GVN_Compute(rvsdg::Region& reg);

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
      default:{
        if (h.IsLoopVar()){ return std::string("lv") + std::to_string(h.value); }
        return std::to_string(h.value);
      }
    }
  }
}

#endif
