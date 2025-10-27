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
PartialRedundancyElimination::PartialRedundancyElimination(): jlm::rvsdg::Transformation("PartialRedundancyElimination"){}

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

void
PartialRedundancyElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  std::cout << TR_BLUE << "Hello JLM its me." << TR_RESET << std::endl;

  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  this->TraverseTopDownRecursively(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_region);
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
  std::cout << std::endl;
}

}

