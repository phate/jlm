/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
 * See COPYING for terms of redistribution.
 */

#define  TR_CMD  "\033"
#define  TR_RESET TR_CMD "[0m"

#define  TR_FG(r,g,b)  TR_CMD "[38;2;" #r ";" #g ";" #b "m"
#define  TR_RED     TR_FG(255,64,64)
#define  TR_GREEN   TR_FG(64, 255, 64)

#define  TR_PURPLE  TR_FG(128,0,128)
#define  TR_YELLOW  TR_FG(255, 255, 64)
#define  TR_ORANGE  TR_FG(255, 128, 0)
#define  TR_BLUE    TR_FG(64, 64, 255)
#define  TR_PINK    TR_FG(255,128,128)
#define  TR_CYAN    TR_FG(64, 255, 255)
#define  TR_GRAY    TR_FG(52,52,52)

#include "../../../tests/test-operation.hpp"
#include "../../rvsdg/gamma.hpp"
#include "../../rvsdg/lambda.hpp"
#include "../../rvsdg/delta.hpp"
#include "../../rvsdg/MatchType.hpp"
#include "../../rvsdg/node.hpp"
#include "../../rvsdg/nullary.hpp"
#include "../../rvsdg/structural-node.hpp"
#include "../../rvsdg/theta.hpp"
#include "../../util/GraphWriter.hpp"
#include "../ir/operators/call.hpp"
#include "../ir/operators/IntegerOperations.hpp"
#include "../ir/operators/operators.hpp"
#include "PartialRedundancyElimination.hpp"
#include "../../rvsdg/control.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/PartialRedundancyElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>
#include <ostream>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/llvm/DotWriter.hpp>
#include <jlm/util/GraphWriter.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include "../../rvsdg/binary.hpp"
#include "../../util/common.hpp"
#include "../ir/types.hpp"
#include "gvn.hpp"
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>

#include <typeinfo>

/**
 *  Partial redundancy elimination:
 *  -invariants: after each insertion of a new GVN all GVN shall have unique values
 *               -including op clusters
 *  -GVN source:
 *        -create a variant type for flows into an operator and gvn from constants
 *            -GVN map responsible for comparing constants
 *  -Collisions:
 *        -a GVN value is always generated when a new tuple of (op, flow0, flow1, ...) is inserted
 *        -on a collision generate a unique symbol
 *            -this prevents accidental matches downstream and keep the invariant of one gvn per
 * value -it should be possible to recompute tuples from edges -outputs -operators: -keep around
 * vectors of leaves with a count after performing the operation -vecs only required for internal
 * nodes in op clusters -compare vectors rather than GVN arguments -in effect operator nodes whose
 * args have the same op return a vector of leaf inputs to the operator dag -tracebacks from op
 * nodes will thus bypasses non-leaf nodes -vectors of leaf counts stored per output edge -possible
 * to discard for too large vectors, preventing excess memory usage -Theta nodes: -Complicated:
 *          -initial values passed into thetas must not match
 *          -the outputs might depend on other loop variables transitively
 *        -Solution:
 *          -dynamically switch between which table to insert values into
 *          -compute a GVN value by setting each loop input to a simple (OP-LOOP-PARAM, index)
 *               -these are placed in a separate table
 *          -this value unique identifies a loop
 *          -this scales well as each loop body is only scanned once for identifying structure
 *              and once afterwards to trickle down values from outer contexts
 *          -for a given set of loop inputs and loop hash, outputs are unique
 *              -that is treat theta nodes as operators from the outside
 *          -once such hashes have been calculated for all thetas proceed by passing
 *              -values into thetas hashed with loop hashes
 *              -two identical loops called with the same values give the same outputs
 */

namespace jlm::rvsdg
{
class Operation;
}

/** This might be moved to util if proven useful elsewhere **/
static int indentation_level = 0;

inline std::string ind()
{
  std::string acc = "";
  for (int i = 0;i<indentation_level;i++)
  {
    acc += "    ";
  }
  return acc;
}

class IndentMan
{

public:
  IndentMan()
  {
    indentation_level++;
  }
  ~IndentMan()
  {
    indentation_level--;
  }
};

/** -------------------------------------------------------------------------------------------- **/

namespace jlm::llvm
{

/** \brief PRE statistics
 *
 */
class PartialRedundancyElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * SweepTimerLabel_ = "SweepTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::PartialRedundancyElimination, sourceFile)
  {}

  void
  StartMarkStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(MarkTimerLabel_).start();
  }

  void
  StopMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

/** -------------------------------------------------------------------------------------------- **/



PartialRedundancyElimination::~PartialRedundancyElimination() noexcept {}
PartialRedundancyElimination::PartialRedundancyElimination() :
  jlm::rvsdg::Transformation("PartialRedundancyElimination"),
  stat_theta_count(0),
  stat_gamma_count(0)
{

}

void PartialRedundancyElimination::TraverseTopDownRecursively(rvsdg::Region& reg, void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node* node))
{
  IndentMan indenter = IndentMan();
  for (rvsdg::Node* node : rvsdg::TopDownTraverser(&reg))
  {
    cb(this, node);
    MatchType(*node, [this,cb](rvsdg::StructuralNode& sn)
    {
      for (auto& reg : sn.Subregions())
      {
        this->TraverseTopDownRecursively(reg, cb);
        std::cout << ind() << TR_GRAY << "..........................." << TR_RESET << std::endl;
      }
    });
  }
}

void
PartialRedundancyElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  std::cout << TR_BLUE << "Hello JLM its me." << TR_RESET << std::endl;

  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  auto& root = rvsdg.GetRootRegion();
  this->TraverseTopDownRecursively(root, initialize_stats);

  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;
  std::cout <<TR_CYAN << "Gamma node count:"  << stat_gamma_count << TR_RESET << std::endl;
  std::cout <<TR_CYAN << "Theta node count:"  << stat_theta_count << TR_RESET << std::endl;
  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;

  GVN_VisitRegion(root);
  TraverseTopDownRecursively(root, dump_node);
  TraverseTopDownRecursively(root, dump_region);
  std::cout << TR_PURPLE << "=================================================" << TR_RESET << std::endl;

  for (auto kv : thetas_){
    auto ic = kv.second.stat_iteration_count;
    std::cout << TR_ORANGE << kv.first->DebugString() << kv.first->GetNodeId() << " Iteration count: " << ic << " Checksum inputs: " << kv.second.checksum_inputs << TR_RESET << std::endl;
  }

  if (gvn_.stat_collisions){
    throw std::runtime_error("gvn_.stat_collisions");
  }
 // PassDownRegion(root);
}

/** -------------------------------------------------------------------------------------------- **/
static size_t reg_counter = 0;
void PartialRedundancyElimination::dump_region(PartialRedundancyElimination* pe, rvsdg::Node* node)
{
  std::string name = node->DebugString() + std::to_string(node->GetNodeId());

  MatchType(*node, [&name](rvsdg::StructuralNode& sn)
  {
    for (auto& reg : sn.Subregions())
    {
      auto my_graph_writer = jlm::util::graph::Writer();

      jlm::llvm::LlvmDotWriter my_dot_writer;
      my_dot_writer.WriteGraphs(my_graph_writer , reg, false);

      std::string full_name = "reg_dump/" + name+"__"+std::to_string(reg_counter)+"__.dot"; reg_counter++;
      std::cout<< TR_RED<<full_name<<TR_RESET<<std::endl;

      std::ofstream my_dot_oss (full_name);
      my_graph_writer.OutputAllGraphs(my_dot_oss, jlm::util::graph::OutputFormat::Dot);
      std::cout << TR_GREEN << "-------------------------------------" << TR_RESET << std::endl;
    }
  });
}

void PartialRedundancyElimination::initialize_stats(PartialRedundancyElimination *pe, rvsdg::Node* node)
{
  MatchType(*node, [&pe, &node](rvsdg::ThetaNode& tn){
    pe->stat_theta_count++;
  });

  MatchType(*node, [&pe, &node](rvsdg::GammaNode& tn){
    pe->stat_gamma_count++;
  });
}

void PartialRedundancyElimination::GVN_VisitRegion(rvsdg::Region& reg)
{
  IndentMan indenter = IndentMan();
  for (rvsdg::Node* node : rvsdg::TopDownTraverser(&reg)){GVN_VisitNode(node);}
}

void PartialRedundancyElimination::GVN_VisitAllSubRegions(rvsdg::Node* node)
{
  MatchType(*node,[this](rvsdg::StructuralNode& sn){
    for (auto& reg : sn.Subregions()){GVN_VisitRegion(reg);}
  });
}

struct StateEdgesAt
{
  std::optional< std::pair<size_t, size_t> > io_state;
  std::optional< std::pair<size_t, size_t> > mem_state;
};
/*
static StateEdgesAt findStateIndices(rvsdg::Node* node)
{
  StateEdgesAt edge_indices;
  edge_indices.io_state = std::make_pair(0,0);
  edge_indices.mem_state = std::make_pair(0,0);

  bool found_io_state = false;
  bool found_mem_state = false;

  for (size_t i = 0 ; i < node->ninputs() ; i++){
    if ( rvsdg::is<MemoryStateType>(node->input(i)->Type()) ){
      edge_indices.mem_state->first = i;
      found_io_state = true;
    }
    if ( rvsdg::is<IOStateType>(node->input(i)->Type()) ){
      edge_indices.io_state->first = i;
      found_mem_state = true;
    }
  }

  for (size_t o = 0 ; o < node->noutputs() ; o++){
    if ( rvsdg::is<MemoryStateType>(node->output(o)->Type()) ){
      edge_indices.mem_state->second = o;
    }
    if ( rvsdg::is<IOStateType>(node->input(o)->Type()) ){
      edge_indices.io_state->second = o;
    }
  }

  if (!found_io_state){edge_indices.io_state = std::nullopt;}
  if (!found_mem_state){edge_indices.mem_state = std::nullopt;}

  return edge_indices;
}
*/
void PartialRedundancyElimination::GVN_VisitLeafNode(rvsdg::Node* node)
{
  std::cout << ind() << "Leaf node: " << node->DebugString() << node->GetNodeId() << std::endl;
  // Treats state edges just like any other value
  if (node->ninputs() && node->noutputs()){
    auto op_opaque = gvn_.FromStr( node->DebugString() );
    auto hash_call_out = gvn_.FromStr("hash_call_out");
    gvn_.Op(op_opaque);
    for (size_t i = 0 ; i <node->ninputs() ; i++){
      auto from = node->input(i)->origin();
      gvn_.Arg( GVNOrWarn(from, node) );
    }
    auto hash_all_inputs = gvn_.End();
    for (size_t i = 0 ; i < node->noutputs() ; i++){
      auto arg_pos = gvn_.FromWord(i);
      auto g_out = gvn_.Op(hash_call_out).Arg(hash_all_inputs).Arg(arg_pos).End();
      RegisterGVN( node->output(i), g_out );
    }
  }

  MatchType(node->GetOperation(),
    [this, node](const jlm::llvm::IntegerConstantOperation& iconst){
      //std::cout << TR_RED << "FOUND: " << iconst.Representation().str() << TR_RESET << std::endl;
      size_t value = 0;
      auto istr = iconst.Representation().str();
      for (size_t i = 0 ; i < istr.length() ; i++){
        value += istr[i] == '1' ? 1 << i : 0;
      }
      RegisterGVN( node->output(0), gvn_.FromWord(value) );
    },
    [this, node](const jlm::rvsdg::BinaryOperation& binop){
      using namespace jlm::rvsdg::gvn;
      JLM_ASSERT(node->ninputs() >= 2);
      auto op    = gvn_.FromStr(binop.debug_string());
      if (binop.debug_string() == "IAdd"){op = GVN_OP_ADDITION; }
      if (binop.debug_string() == "IMul"){op = GVN_OP_MULTIPLY; }
      if (binop.debug_string() == "INe"){op = GVN_OP_NEQ; }
      if (binop.debug_string() == "IEq"){op = GVN_OP_EQ; }

      auto a     = GVNOrWarn( node->input(0)->origin(), node);
      auto b     = GVNOrWarn( node->input(1)->origin(), node);

      //std::cout << TR_RED << "BINOP" << to_string(op) << "[" << to_string(a) << " , " << to_string(b) << "]" << TR_RESET << std::endl;

      RegisterGVN(node->output(0), gvn_.Op(op).Arg(a).Arg(b).End());
    }
  );

  for (size_t i = 0; i < node->noutputs(); i++){
    if (output_to_gvn_.find(node->output(i)) == output_to_gvn_.end()){
      if (node->DebugString() == std::string("undef")){
        RegisterGVN( node->output(i), rvsdg::gvn::GVN_NO_VALUE );
      }
      auto& op = node->GetOperation();
      if ( rvsdg::is_ctlconstant_op( op ) ){
        RegisterGVN( node->output(i), rvsdg::gvn::GVN_NO_VALUE );
      }
    }
  }

  MatchType(node->GetOperation(), [this, node](const rvsdg::MatchOperation& mop){
    using namespace jlm::rvsdg::gvn;
    auto pred = GVNOrPanic( node->input(0)->origin(), node );
    if (pred == GVN_TRUE) {pred = 1;}
    if (pred == GVN_FALSE){pred = 0;}
    if (jlm::rvsdg::gvn::GVN_IsSmallValue(pred)){
      RegisterGVN( node->output(0), mop.alternative(static_cast<size_t>(pred)) );
    }
    //std::cout << "MAPPED MATCH" << std::endl;
    //dump_node(this, node);
  });
}

void PartialRedundancyElimination::GVN_VisitGammaNode(rvsdg::Node * node)
{
  using namespace jlm::rvsdg::gvn;

  MatchType(*node, [this,node](rvsdg::GammaNode& gn){
    std::cout << ind() << TR_YELLOW << node->DebugString() << node->GetNodeId() << TR_RESET << std::endl;

    //Route gvn values into alternatives
    for (auto br : gn.GetEntryVars()){
      auto out = br.input->origin();
      for (auto into_branch : br.branchArgument){
        RegisterGVN(into_branch, GVNOrWarn(out, node));
      }
    }

    auto mv_edge = gn.GetMatchVar().input->origin();
    auto mv_val  = GVNOrWarn( mv_edge, node );
    for (auto mv : gn.GetMatchVar().matchContent){
      RegisterGVN(mv, mv_val);
    }

    //  --------------------------------------------------------------------------------------------
    GVN_VisitAllSubRegions(node);
    //  --------------------------------------------------------------------------------------------

    auto GAMMA_VARIABLE_OUT = gvn_.FromStr("GAMMA_VARIABLE_OUT");
    auto BRANCHES_CONDITIONALLY = gvn_.FromStr("CONDITIONAL_BRANCHING");

    auto entry_vars = gn.GetEntryVars();
    auto exit_vars = gn.GetExitVars();
    auto match_var = gn.GetMatchVar();
    auto predicate = GVNOrPanic(gn.predicate()->origin(), node);

    for (auto ev : exit_vars){
      auto any_branch_value = GVN_NO_VALUE;
      auto value_from_branch_always_taken = BRANCHES_CONDITIONALLY;
      gvn_.Op(GVN_OP_ANY_ORDERED);         // Returns input if all inputs are the same.
      for (size_t b = 0; b < ev.branchResult.size(); b++){
        auto leaving_branch = ev.branchResult[b];
        auto from_inner = GVNOrZero(leaving_branch->origin());
        gvn_.Arg( from_inner );
        any_branch_value = from_inner;

        // --------------------- check if predicate is known and maps to this branch -------------
        auto match_node = rvsdg::TryGetOwnerNode<rvsdg::Node>( *(match_var.input->origin()) );
        if (match_node){
          MatchType(match_node->GetOperation(), [&value_from_branch_always_taken, predicate, b, from_inner](const rvsdg::MatchOperation& mop){
            if (jlm::rvsdg::gvn::GVN_IsSmallValue(predicate) && mop.alternative(static_cast<size_t>(predicate)) == b){
              value_from_branch_always_taken = from_inner;
            }
          });
        }
        // ---------------------
      }
      auto branches_merged = gvn_.End();

      RegisterGVN( ev.output,
        gvn_.Op(GAMMA_VARIABLE_OUT).Arg(predicate).Arg(branches_merged).End()
      );

      // If all branches output the same value
      if (any_branch_value == branches_merged){ RegisterGVN(ev.output, any_branch_value); }
      // If the match variable has a known value
      if (value_from_branch_always_taken != BRANCHES_CONDITIONALLY){ RegisterGVN(ev.output, value_from_branch_always_taken); }
    }



  });
}

jlm::rvsdg::gvn::GVN_Val PartialRedundancyElimination::ComputeInputCheckSumForTheta(rvsdg::ThetaNode& tn)
{
  auto CHECKSUM = gvn_.FromStr("CHECKSUM");
  gvn_.Op(CHECKSUM);
  for (auto v : tn.GetLoopVars()){
    auto from_outer = GVNOrZero( v.input->origin() );
    gvn_.Arg( from_outer );
  }
  return gvn_.End();
}

void PartialRedundancyElimination::GVN_VisitThetaNode(rvsdg::Node * node)
{

  MatchType(*node, [this,node](rvsdg::ThetaNode& tn){
    using namespace jlm::rvsdg::gvn;
    auto GVN_INVARIANT = gvn_.FromStr("INVARIANT");
    auto LOOP_EXIT = gvn_.FromStr("LOOP_EXIT");
    auto lv = tn.GetLoopVars();
    auto OP_PRISM   = gvn_.FromStr("prism");

    /** ----------------------------------- LOAD INPUTS INTO PRE.disruptors ------------------- */

    if (thetas_.find(node) == thetas_.end()){
      thetas_.insert({node, ThetaData()});
      thetas_[node].prism = 0;           // This value capture the behavior of thetas given their context
      thetas_[node].stat_iteration_count = 0;
      thetas_[node].checksum_inputs = ComputeInputCheckSumForTheta(tn);
      for (auto v : tn.GetLoopVars()){
        thetas_[node].pre.Add( GVNOrWarn( v.input->origin() , node ) );
        thetas_[node].post.Add( 0 );
      }
    }else{
      // Only evaluate loop body once per unique set of inputs
      auto check_new = ComputeInputCheckSumForTheta(tn);
      if (thetas_[node].checksum_inputs == check_new){return;}
      thetas_[node].checksum_inputs = check_new;

      ///// Only called for nested loops
      {
        for (size_t i = 0; i < lv.size(); i++){
          auto from_outer = GVNOrWarn( lv[i].input->origin(), node );
          auto merged = gvn_.Op(GVN_OP_ANY_ORDERED).Arg(from_outer).Arg(thetas_[node].pre.elements[i].disruptor).End();
          if ( thetas_[node].post.elements[i].disruptor == GVN_INVARIANT){
            thetas_[node].pre.elements[i].disruptor = from_outer;
            thetas_[node].pre.elements[i].partition = from_outer;
            thetas_[node].pre.elements[i].original_partition = from_outer;
          }else{
            thetas_[node].pre.elements[i].disruptor = merged;
          }
        }
      }
    }
    auto& td = thetas_[node];

    do{
      // ================= PERFORM GVN IN LOOP BODY ========================
      for (size_t i = 0; i < lv.size(); i++){
        RegisterGVN( lv[i].pre, td.pre.elements[i].disruptor );
      }

      GVN_VisitAllSubRegions(node);

      for (size_t i = 0; i < lv.size(); i++){
        td.post.elements[i].disruptor = GVNOrWarn( lv[i].post->origin(), node );
      }
      // ================= END LOOP BODY ====================================

      for (size_t i = 0; i < lv.size(); i++){
        auto input  = td.pre.elements[i].disruptor;
        auto output = td.post.elements[i].disruptor;
        auto merged_io = gvn_.Op(GVN_OP_ANY_ORDERED).Arg(input).Arg(output).End();
        td.post.elements[i].partition = merged_io;
      }

      GVN_Val predicate = GVNOrPanic( tn.predicate()->origin(), node );
      /// This hash depends on the mapping between inputs to ouputs as well as the predicate
      /// Does not depend on order of loop variables or duplicate loop variables
      td.prism = gvn_.Op(OP_PRISM).Arg(predicate).FromPartitions(td.post).End();
      td.post.OrderByOriginal();

      for (size_t i = 0; i < lv.size(); i++){
        auto pi = td.post.elements[i].partition;
        auto merged_with_prism = gvn_.Op(GVN_OP_ANY_ORDERED).Arg(pi).Arg(td.prism).End();
        bool was_invariant = td.pre.elements[i].disruptor == td.post.elements[i].disruptor;

        // Note: this doesn't use rvsdg::ThetaLoopVarIsInvariant()
        //       state edges can be made invariant inside thetas only
        //       if they pass through referentially transparent nodes.
        //       Computing referential transparency

        if (was_invariant){
          td.post.elements[i].partition = GVN_INVARIANT;
        }else{
          td.post.elements[i].partition = merged_with_prism;
          auto loop_old = td.pre.elements[i].disruptor;
          td.pre.elements[i].disruptor = gvn_.Op(GVN_OP_ANY_ORDERED).Arg(merged_with_prism).Arg(loop_old).End();
        }
      }

      td.stat_iteration_count++;
    } while (thetas_[node].pre.Fracture());
    //Loop when either a loop variable changed from its initial value or loop variables
    //   which were the same at the start of the iteration diverged (partition splits or fractures).

    /** ----------------------------------- COMPUTE LOOP OUTPUTS -------------------------------------- */

    for (size_t i = 0; i < lv.size(); i++){
      auto inv = thetas_[node].post.elements[i].partition == GVN_INVARIANT;
      if (inv){
        RegisterGVN(lv[i].output, GVNOrWarn(lv[i].input->origin(), node));
      }else{
        // The gvn for loop output variables must be different from the value exiting the loop.
        auto superimposed_values_inside_loop = thetas_[node].pre.elements[i].disruptor;
        auto g = gvn_.Op(LOOP_EXIT).Arg(superimposed_values_inside_loop).End();
        RegisterGVN(lv[i].output, g);
      }
    }
  });
}

void PartialRedundancyElimination::GVN_VisitLambdaNode(rvsdg::Node * node)
{
  MatchType(*node, [this,node](rvsdg::LambdaNode& ln){
    size_t i = 0;
    for (auto arg : ln.GetFunctionArguments()){
      RegisterGVN(arg, gvn_.FromStr("Param:" + std::to_string(i)) ); i++;
    }
    for (auto arg : ln.GetContextVars())
    {
      auto from = arg.input->origin();
      RegisterGVN(arg.inner, GVNOrWarn(from, node));
    }
    std::cout << ind() << TR_PURPLE << node->DebugString() << node->GetNodeId() << TR_RESET << std::endl;
    GVN_VisitAllSubRegions(node);
  });
}

void PartialRedundancyElimination::GVN_VisitNode(rvsdg::Node* node)
{
  MatchTypeWithDefault(*node,
  [this,node](rvsdg::DeltaNode& dn){
      std::cout << ind() << TR_CYAN << node->DebugString() << node->GetNodeId() << TR_RESET << std::endl;
      GVN_VisitAllSubRegions(node);
    },
    [this,node](rvsdg::ThetaNode& tn){
      GVN_VisitThetaNode(node);
    },
    [this,node](rvsdg::GammaNode& gn){
      GVN_VisitGammaNode(node);
    },
    [this,node](rvsdg::LambdaNode& ln){
      GVN_VisitLambdaNode(node);
    },
    //DEFAULT
    [this, node](){
      GVN_VisitLeafNode(node);
    }
  );
}

void PartialRedundancyElimination::dump_node(PartialRedundancyElimination* pe, rvsdg::Node* node)
{
  using namespace jlm::rvsdg::gvn;
  std::cout << ind() << TR_BLUE << node->DebugString() << "<"<<node->GetNodeId() <<">"<< TR_RESET;

  MatchType(*node, [&pe, &node](rvsdg::LambdaNode& ln){
    std::cout <<TR_ORANGE;
    for (auto arg : ln.GetFunctionArguments()){
      std::cout << " : " << pe->GVNOrZero(arg);
    }
    std::cout << TR_RESET;
  });

  MatchType(node->GetOperation(), [&pe, node](const jlm::llvm::IntegerConstantOperation& iconst)
  {
    JLM_ASSERT(node->noutputs() == 1);
    std::cout << TR_CYAN;
    std::cout << to_string(pe->GVNOrZero( node->output(0) ));
    std::cout << TR_RESET;
  });

  for (size_t i = 0; i < node->ninputs(); i++){
    std::cout << TR_GREEN;
    std::cout << " : " << to_string(pe->GVNOrZero( node->input(i)->origin() ));
  }
  std::cout << TR_GRAY << " => ";
  for (size_t i = 0; i < node->noutputs(); i++){
    std::cout << TR_RED;
    std::cout << " : " << to_string(pe->GVNOrZero( node->output(i) ));
  }
  std::cout << std::endl;
}

/** ********************************************************************************************* */
/** ********************************************************************************************* */
/** ********************************************************************************************* */

void PartialRedundancyElimination::PassDownRegion(rvsdg::Region& reg)
{
  IndentMan indenter = IndentMan();

  for (rvsdg::Node* node : rvsdg::TopDownTraverser(&reg)){PassDownNode(node);}
}

void PartialRedundancyElimination::PassDownAllSubRegions(rvsdg::Node* node)
{
  MatchType(*node,[this](rvsdg::StructuralNode& sn){
    for (auto& reg : sn.Subregions()){PassDownRegion(reg);}
  });
}

void PartialRedundancyElimination::PassDownNode(rvsdg::Node* node)
{
  MatchTypeWithDefault(*node,
  [this,node](rvsdg::DeltaNode& dn){
      std::cout << ind() << TR_CYAN << node->DebugString() << node->GetNodeId() << TR_RESET << std::endl;
      PassDownAllSubRegions(node);
    },
    [this,node](rvsdg::ThetaNode& tn){
      PassDownThetaNode(node);
    },
    [this, node](rvsdg::GammaNode& gn)
    {
      PassDownGammaNode(node);
    },
    [this,node](rvsdg::LambdaNode& ln){
      PassDownLambdaNode(node);
    },
    //DEFAULT
    [this, node](){
      PassDownLeafNode(node);
    }
  );
}

void PartialRedundancyElimination::PassDownThetaNode(rvsdg::Node* node)
{
  PassDownAllSubRegions(node);
}

void PartialRedundancyElimination::PassDownGammaNode(rvsdg::Node* node)
{
  PassDownAllSubRegions(node);
}

void PartialRedundancyElimination::PassDownLeafNode(rvsdg::Node* node)
{
   std::cout << ind() <<TR_CYAN << node->DebugString()<<node->GetNodeId() << TR_RESET;
}

void
PartialRedundancyElimination::PassDownLambdaNode(rvsdg::Node * ln)
{
  std::cout << ind() << TR_GREEN << ln->GetNodeId() << TR_RESET;
  PassDownAllSubRegions(ln);
}


}


