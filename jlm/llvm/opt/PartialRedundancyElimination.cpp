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
  this->TraverseTopDownRecursively(root, dump_region);

  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;
  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;
  this->TraverseTopDownRecursively(root, dump_node);
  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;
  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;
  this->TraverseTopDownRecursively(root, initialize_stats);

  this->TraverseTopDownRecursively(root, dump_node);

  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;
  std::cout <<TR_CYAN << "Gamma node count:"  << stat_gamma_count << TR_RESET << std::endl;
  std::cout <<TR_CYAN << "Theta node count:"  << stat_theta_count << TR_RESET << std::endl;
  std::cout << TR_GRAY << "=================================================" << TR_RESET << std::endl;

  /*jlm::rvsdg::gvn::gvn_verbose = false;
  jlm::rvsdg::gvn::RunAllTests();*/

  std::cout << TR_PURPLE << "=================================================" << TR_RESET << std::endl;
  TraverseTopDownRecursively(root, dump_node);
  GVN_VisitRegion(root);
  TraverseTopDownRecursively(root, dump_node);
  std::cout << TR_PURPLE << "=================================================" << TR_RESET << std::endl;

 // PassDownRegion(root);
}

/** -------------------------------------------------------------------------------------------- **/

void PartialRedundancyElimination::dump_region(PartialRedundancyElimination* pe, rvsdg::Node* node)
{
  std::string name = node->DebugString() + std::to_string(node->GetNodeId());
  size_t reg_counter = 0;

  MatchType(*node, [&name, &reg_counter](rvsdg::StructuralNode& sn)
  {
    for (auto& reg : sn.Subregions())
    {
      auto my_graph_writer = jlm::util::graph::Writer();

      jlm::llvm::LlvmDotWriter my_dot_writer;
      my_dot_writer.WriteGraphs(my_graph_writer , reg, false);

      std::string full_name = name+std::to_string(reg_counter++)+".dot";
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

void PartialRedundancyElimination::GVN_VisitLeafNode(rvsdg::Node* node)
{
  std::cout << ind() << "Leaf node: " << node->DebugString() << node->GetNodeId() << std::endl;

  if (node->ninputs() && node->noutputs()){
    auto op_opaque = gvn_.FromPtr(&(node->GetOperation()) );
    auto hash_call_out = gvn_.FromStr("hash_call_out");
    gvn_.Op(op_opaque);
    for (size_t i = 0 ; i <node->ninputs() ; i++){
      auto from = node->input(i)->origin();
      gvn_.Arg( GVNOrFail(from, node) );
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
      RegisterGVN( node->output(0), gvn_.FromStr(iconst.Representation().str()) );
    },
    [this, node](const jlm::rvsdg::BinaryOperation& binop){
      if (binop.is_associative() && binop.is_commutative()){
        JLM_ASSERT(node->ninputs() >= 2);
        auto op    = gvn_.FromStr(binop.debug_string(), rvsdg::gvn::GVN_OPERATOR_IS_CA);
        auto a     = GVNOrFail( node->input(0)->origin(), node);
        auto b     = GVNOrFail( node->input(1)->origin(), node);
        RegisterGVN(node->output(0), gvn_.Op(op).Arg(a).Arg(b).End());
      }
    }
  );

  for (size_t i = 0; i < node->noutputs(); i++){
    if (output_to_gvn_.find(node->output(i)) == output_to_gvn_.end()){
      std::cout <<ind() << TR_RED << "Warning: Missing out from:" << node->DebugString() << node->GetNodeId() << std::endl;
      RegisterGVN( node->output(i), gvn_.Leaf() );
    }
  }
}

void PartialRedundancyElimination::GVN_VisitGammaNode(rvsdg::Node * node)
{
  MatchType(*node, [this,node](rvsdg::GammaNode& gn){
    std::cout << ind() << TR_YELLOW << node->DebugString() << node->GetNodeId() << TR_RESET << std::endl;

    //Route gvn values into alternatives
    for (auto br : gn.GetEntryVars()){
      auto out = br.input->origin();
      for (auto into_branch : br.branchArgument){
        RegisterGVN(into_branch, GVNOrFail(out, node));
      }
    }
    auto selector = gn.GetMatchVar().input->origin();
    auto match_var = GVNOrFail(selector, node);
    for (auto mv : gn.GetMatchVar().matchContent){
      RegisterGVN( mv, GVNOrFail(selector, node));
    }

    GVN_VisitAllSubRegions(node);

    //route results out and handle cases where result is always the same
    for (auto ev : gn.GetExitVars())
    {
      using namespace jlm::rvsdg::gvn;
      auto any_val = GVN_NO_VALUE;
      gvn_.Op(GVN_OP_ANY_ORDERED);
      for (auto leaving_branch : ev.branchResult){
        auto from_inner = GVNOrZero(leaving_branch->origin());
        gvn_.Arg( from_inner );
        any_val = from_inner;
      }
      auto branches_merged = gvn_.End();

      if (any_val == branches_merged){
        RegisterGVN(ev.output, branches_merged); // If all branches output the same value
      }else{
        auto sel_op = gvn_.FromStr("selector");
        auto hash_with_selector = gvn_.Op(sel_op).Arg(match_var).Arg(branches_merged).End();
        RegisterGVN( ev.output, hash_with_selector); // Typical case. Note: branch order matters.
      }
    }
  });
}

void PartialRedundancyElimination::GVN_VisitThetaNode(rvsdg::Node * node)
{
  MatchType(*node, [this,node](rvsdg::ThetaNode& tn)
  {
    std::cout << ind() << TR_ORANGE << node->DebugString() << node->GetNodeId() << TR_RESET << std::endl;
    GVN_VisitAllSubRegions(node);
    throw std::runtime_error("Not implemented");
  });
}

void PartialRedundancyElimination::GVN_VisitLambdaNode(rvsdg::Node * node)
{
  MatchType(*node, [this,node](rvsdg::LambdaNode& ln){
    for (auto arg : ln.GetFunctionArguments()){
      RegisterGVN(arg, gvn_.Leaf());
    }
    for (auto arg : ln.GetContextVars())
    {
      auto from = arg.input->origin();
      RegisterGVN(arg.inner, GVNOrFail(from, node));
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
    std::cout << pe->GVNOrZero( node->output(0) );
    std::cout << TR_RESET;
  });

  for (size_t i = 0; i < node->noutputs(); i++){
    std::cout << TR_GREEN;
    std::cout << " : " << pe->GVNOrZero( node->output(i) );
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


