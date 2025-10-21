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
#include "../../rvsdg/theta.hpp"
#include "../../rvsdg/MatchType.hpp"
#include "../../rvsdg/node.hpp"
#include "../../rvsdg/nullary.hpp"
#include "../../rvsdg/structural-node.hpp"
#include "../../rvsdg/theta.hpp"
#include "../../util/GraphWriter.hpp"
#include "../ir/operators/call.hpp"
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



class  LambdaParameterOperation : public jlm::rvsdg::Operation
{
public:
  LambdaParameterOperation(){} //trivial constructor
  bool operator==(const Operation & other) const noexcept;
  std::string debug_string() const override;
  virtual ~LambdaParameterOperation() noexcept override;
  [[nodiscard]] virtual std::unique_ptr<jlm::rvsdg::Operation>copy() const override;
};

LambdaParameterOperation::~LambdaParameterOperation() noexcept = default;

bool LambdaParameterOperation::operator==(const Operation & other) const noexcept {return &other == this;}
std::string LambdaParameterOperation::debug_string() const { return "LambdaParameterOperation";}
std::unique_ptr<jlm::rvsdg::Operation> LambdaParameterOperation::copy() const { std::cout<<"Attempt to copy singleton"; JLM_ASSERT(false); return nullptr; }
static LambdaParameterOperation lambdaParamOp;


PartialRedundancyElimination::~PartialRedundancyElimination() noexcept {}

PartialRedundancyElimination::PartialRedundancyElimination(): jlm::rvsdg::Transformation("PartialRedundancyElimination"), gvn_man_(){}


void PartialRedundancyElimination::TraverseTopDownRecursively(rvsdg::Region& reg, void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node& node))
{
  IndentMan indenter = IndentMan();
  for (rvsdg::Node* node : rvsdg::TopDownTraverser(&reg))
  {
    cb(this, *node);
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

void PartialRedundancyElimination::GVN_Compute(rvsdg::Region& reg)
{
  IndentMan indenter = IndentMan();
  for (rvsdg::Node* node : rvsdg::TopDownTraverser(&reg))
  {
    /* setup flows into regions of structural nodes */
    MatchType(*node,
      [this](const rvsdg::GammaNode& gn){
        for (auto v : gn.GetEntryVars()){
          for (auto ba : v.branchArgument){
            gvn_man_.ExtendFlow(v.input, ba);
          }
        }
      },
      [this](const rvsdg::ThetaNode& tn){
        for (auto v : tn.GetLoopVars()){
          if (rvsdg::ThetaLoopVarIsInvariant(v)){
            gvn_man_.ExtendFlow(v.input, v.pre);
          }
        }
      },
      [this](rvsdg::LambdaNode& lm){
        for ( auto cv : lm.GetContextVars() ){
          gvn_man_.ExtendFlow(cv.input, cv.inner );
        }
        auto params = lm.GetFunctionArguments();
        for (size_t i = 0; i < params.size() ; i++){
          gvn_man_.Start(params[i], &lambdaParamOp)->WithIndex(i)->End();
        }
      }
    );

    MatchType(*node, [this](rvsdg::StructuralNode& sn)
    {
      for (auto& subreg : sn.Subregions())
      {
        this->GVN_Compute(subreg );
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
  //
  // // NEW CODE USING REACTIVE STYLE CALLBACKS
  // // SHOULD HANDLE THETA NODES, BUT MORE TESTING IS REQUIRED
  //
  // jlm::llvm::flows::FlowData<GVN_Hash> fd(&gvn_hashes_);
  //
  //
  // //cb for handling two flows coming from different sub-regions of a gamma node.
  // //  This reprents the GVN of expressions such as (a > b ? a : b)
  // //  Thus, the branch order matters.
  // //  The net hash of outputs from gamma nodes is computed by reducing each output across branches
  // //    with this callback.
  // auto merge_gvn_ga = [](std::optional<GVN_Hash>& a, std::optional<GVN_Hash>& b)
  // {
  //   if (!a){return b;}  if (!b){return a;}
  //   if (*a == GVN_Hash::Tainted() || *b == GVN_Hash::Tainted()){ return std::optional(GVN_Hash::Tainted()); }
  //   if ( a->value == b->value ){ return a; }
  //   size_t h = a->value ^ (b->value << 3); //Hash branches differently.
  //   return std::optional( GVN_Hash(h) );
  // };
  //
  // //cb for merging flows coming into a theta node for the first time and from previous iterations.
  // //  on the second iteration all sub-nodes will have their values overwritten, except
  // //  loop invariant nodes.
  // auto merge_gvn_th = [](rvsdg::ThetaNode& tn, std::optional<GVN_Hash>& a, std::optional<GVN_Hash>& b)
  // {
  //   if (!a){return b;}  if (!b){std::cout<<"UNREACHABLE"; exit(-1);}
  //   if (*a == GVN_Hash::Tainted() || *b == GVN_Hash::Tainted()){ return std::optional(GVN_Hash::Tainted() ); }
  //   if (a->value == b->value){return a;}
  //   if (a && a->IsLoopVar()){return a;} // This is required for fixed points to be reached by gvn
  //                                       // Values are identified as variant on exit of first iteration
  //                                       // LoopVar hashes trickle through the loop body once more
  //                                       // Subsequent iterations are blocked as loopvar hashes are never overwritten
  //   return std::optional( GVN_Hash::LoopVar( a->value ^ b->value ));
  // };
  //
  //
  // jlm::llvm::flows::ApplyDataFlowsTopDown(rvsdg.GetRootRegion(), fd, merge_gvn_ga, merge_gvn_th,
  // [](rvsdg::Node& node,
  //   std::vector<std::optional<GVN_Hash>>& flows_in,
  //   std::vector<std::optional<GVN_Hash>>& flows_out
  //   )
  //   {
  //     // The gvn hashes are stored automatically by the argument fd, when the flows_out vector
  //     // is written to.
  //     std::cout << TR_GREEN << node.GetNodeId() << ":" << node.DebugString() << TR_RESET << std::endl;
  //
  //     rvsdg::MatchType(node.GetOperation(),
  //       // -----------------------------------------------------------------------------------------
  //       [&flows_out](const jlm::llvm::IntegerConstantOperation& iconst){
  //         std::hash<std::string> hasher;
  //         flows_out[0] = GVN_Hash( hasher(iconst.Representation().str()) );
  //       },
  //
  //       // -----------------------------------------------------------------------------------------
  //       [&flows_in, &flows_out](const rvsdg::BinaryOperation& op){
  //         JLM_ASSERT(flows_in.size() == 2);
  //         if (!(flows_in[0]) || !(flows_in[1])){
  //           std::cout<< TR_RED << "Expected some input" << TR_RESET << std::endl;return;
  //         }
  //
  //         std::hash<std::string> hasher;
  //         size_t h = hasher(op.debug_string() );
  //
  //         size_t a = hasher(std::to_string(flows_in[0]->value));
  //         size_t b = hasher(std::to_string(flows_in[1]->value));
  //         bool c_and_a = op.is_commutative() && op.is_associative();
  //         h ^= c_and_a ? (a + b) : (a ^ (b << 3));
  //         flows_out[0] = std::optional<GVN_Hash>(h);
  //       },
  //       // -----------------------------------------------------------------------------------------
  //       [&flows_in, &flows_out](const rvsdg::UnaryOperation& op){
  //         if (!(flows_in.size())){
  //           std::cout<< TR_RED << "Expected some input" << TR_RESET << std::endl;return;
  //         }
  //         std::hash<std::string> hasher;
  //         size_t h = hasher(op.debug_string() );
  //         size_t a = hasher(std::to_string(flows_in[0]->value));
  //         h ^= a;
  //         flows_out[0] = std::optional<GVN_Hash>(h);
  //       },
  //       // -----------------------------------------------------------------------------------------
  //       [&node, &flows_in, &flows_out](const jlm::llvm::CallOperation& op){
  //         std::string s = node.DebugString() + "CALL";
  //         std::hash<std::string> hasher;
  //         size_t h = hasher(s);
  //         for (size_t i = 0; i < flows_out.size(); i++){
  //           flows_out[i] = std::optional<GVN_Hash>( h + i);
  //         }
  //       }
  //     );
  //
  //     rvsdg::MatchType(node,
  //       // -----------------------------------------------------------------------------------------
  //       [&flows_out](rvsdg::LambdaNode& lm){
  //         //std::cout << TR_PINK << "LAMBDA PARAMS" << flows_out.size() << TR_RESET << std::endl;
  //         auto s = lm.DebugString();
  //         std::hash<std::string> hasher;
  //         size_t h = hasher(s);
  //         for (size_t i = 0; i < flows_out.size(); i++){
  //           flows_out[i] = std::optional( GVN_Hash( h+i ) );
  //         }
  //       }
  //     );
  //   }
  // );
  //

  std::cout << TR_RED << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;


  this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_region);

  this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_node);
  this->GVN_Compute( rvsdg.GetRootRegion() );
  /*
  {

    std::cout << TR_RED << "================================================================" << TR_RESET << std::endl;
    this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::register_leaf_hash);
    std::cout << TR_RED << "================================================================" << TR_RESET << std::endl;
    this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_node);
    std::cout << TR_BLUE << "================================================================" << TR_RESET << std::endl;
    this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::hash_node);
    std::cout << TR_PINK << "================================================================" << TR_RESET << std::endl;
  }
  */

  this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_node);

  std::cout << TR_GREEN << "=================================================" << TR_RESET << std::endl;
}

/** -------------------------------------------------------------------------------------------- **/

void PartialRedundancyElimination::dump_region(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  std::string name = node.DebugString() + std::to_string(node.GetNodeId());
  size_t reg_counter = 0;

  MatchType(node, [&name, &reg_counter](rvsdg::StructuralNode& sn)
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

void PartialRedundancyElimination::dump_node(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  std::cout << ind() << TR_BLUE << node.DebugString() << "<"<<node.GetNodeId() <<">"<< TR_RESET;
  for (size_t i = 0; i < node.noutputs(); i++)
  {
    if (pe->gvn_man_.GetGVN(node.output(i))){
      std::cout << TR_PURPLE << "gvn" << *(pe->gvn_man_.GetGVN(node.output(i))) << TR_RESET;
    }
    auto h = pe->GetHash(node.output(i));
    if (h.IsSome())
    {
      std::string color = (pe->DBG_HashCount(h) > 1 ? TR_GREEN : TR_YELLOW);

      MatchType(node.GetOperation(),[&color](const jlm::llvm::CallOperation& op){color = TR_CYAN;});
      MatchType(node, [&color](rvsdg::LambdaNode& lm){color = TR_ORANGE;});

      std::cout << " : " << color << std::to_string(h) << TR_RESET;
    }
  }
  MatchType(node, [pe](rvsdg::LambdaNode& lm)
  {
    for (auto& param : lm.GetFunctionArguments())
    {
      if (pe->gvn_hashes_.find(param) != pe->gvn_hashes_.end())
      {
        GVN_Hash h = pe->gvn_hashes_[param];
        std::cout << TR_ORANGE << " : " << std::to_string(h) << TR_RESET;
      }
    }
  });
  std::cout << std::endl;
}

void PartialRedundancyElimination::register_leaf_hash(PartialRedundancyElimination *pe, rvsdg::Node& node)
{
  MatchType(node.GetOperation(),
    [pe, &node](const jlm::llvm::IntegerConstantOperation& iconst)
    {
      std::hash<std::string> hasher;
      size_t h = hasher(iconst.Representation().str());
      pe->AssignGVN(node.output(0), GVN_Hash(h));
    }
  );

  /* Add each lambda parameter as a leaf hash for hashing within its body */
  MatchType(node, [pe, &node](rvsdg::LambdaNode& lm)
  {
    auto fargs = lm.GetFunctionArguments();
    auto s = node.DebugString() + std::to_string(node.GetNodeId());
    for (size_t i = 0; i < fargs.size(); i++){
      pe->AssignGVN(fargs[i], node.DebugString(), i);
    }
    pe->AssignGVN(node.output(0), node.DebugString() + "LM_NODE", 0);
  });
}

void PartialRedundancyElimination::hash_call(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  MatchTypeOrFail(node.GetOperation(), [pe, &node](const jlm::llvm::CallOperation& op)
  {
    std::string s = node.DebugString() + std::to_string(node.GetNodeId()); // + op.GetLambdaOutput();
    //std::cout << TR_PINK << s << TR_RESET << std::endl;
    pe->AssignGVN(node.output(0), s, 0);
  });
}

void PartialRedundancyElimination::hash_gamma(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  MatchTypeOrFail(node, [pe](rvsdg::GammaNode& node)
  {
    for (auto ev : node.GetEntryVars()){
      rvsdg::Output* origin = ev.input->origin();
      GVN_Hash h = pe->GetHash(origin);
      if (pe->GetHash(origin).IsSome())
      {
        for (rvsdg::Output* brarg : ev.branchArgument){
          pe->AssignGVN(brarg, h);
        }
      }
    }
  });
}

void PartialRedundancyElimination::hash_node(PartialRedundancyElimination *pe, rvsdg::Node& node)
{
  /*Match by operation*/
  MatchType(node.GetOperation(),
    [pe, &node](const rvsdg::BinaryOperation& op){hash_bin(pe, node);},
    [pe, &node](const jlm::llvm::CallOperation& op){hash_call(pe, node);}
  );
  /*Match by node type*/
  MatchType(node,
    [pe](rvsdg::GammaNode& node){hash_gamma(pe, node);}
  );
}

void PartialRedundancyElimination::hash_bin(PartialRedundancyElimination *pe, rvsdg::Node& node)
{
  MatchType(node.GetOperation(), [pe, &node](const rvsdg::BinaryOperation& op)
  {

    std::hash<std::string> hasher;
    size_t h = hasher(op.debug_string());
    bool was_hashable = true;
    for (size_t i = 0 ; i < node.ninputs(); i++)
    {
      if (pe->gvn_hashes_.find(node.input(i)->origin()) != pe->gvn_hashes_.end())
      {
        auto hash_in = pe->GetHash(node.input(i)->origin());
        if (op.is_commutative() && op.is_associative())
        {
          h += hasher(std::to_string(hash_in.value));
        }else
        {
          h ^= hasher(std::to_string(hash_in.value)) * (i+1);
        }
      }else
      {
        std::cout << TR_RED << node.DebugString() << node.GetNodeId()<< "MISSING INPUT HASH" << TR_RESET << std::endl;
        auto input_source = node.input(i)->origin()->GetOwner();
        try
        {
          auto src_node = std::get<rvsdg::Node*>(input_source);
          std::cout << TR_RED << "origin: " << src_node->DebugString() << src_node->GetNodeId() << TR_RESET<< std::endl;
        }catch (std::bad_variant_access& ex)
        {
          std::cout<<TR_RED<<"Found a hash coming from a region. Todo: error info here."<<TR_RESET<<std::endl;
        }
        was_hashable = false;
      }
    }
    if (was_hashable){
      pe->AssignGVN(node.output(0), h);
    }

  });
}

}
