/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#define  TR_CMD  "\033"
#define  TR_RESET TR_CMD "[0m"

#define  TR_FG(r,g,b)  TR_CMD "[38;2;" #r ";" #g ";" #b "m"
#define  TR_RED     TR_FG(255,64,64)
#define  TR_GREEN   TR_FG(64, 255, 64)

#define  TR_YELLOW  TR_FG(255, 255, 64)
#define  TR_ORANGE  TR_FG(255, 128, 0)
#define  TR_BLUE    TR_FG(64, 64, 255)
#define  TR_PINK    TR_FG(255,128,128)
#define  TR_CYAN    TR_FG(64, 255, 255)

#include "../../../tests/test-operation.hpp"
#include "../../rvsdg/gamma.hpp"
#include "../../rvsdg/lambda.hpp"
#include "../../rvsdg/MatchType.hpp"
#include "../../rvsdg/node.hpp"
#include "../../rvsdg/nullary.hpp"
#include "../../rvsdg/structural-node.hpp"
#include "../../util/GraphWriter.hpp"
#include "../ir/operators/call.hpp"
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
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>
#include <ostream>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/llvm/DotWriter.hpp>
#include <jlm/util/GraphWriter.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>

#include <typeinfo>

namespace jlm::llvm
{

class PartialRedundancyElimination::Context final
{
public:
  /*void
  MarkAlive(const jlm::rvsdg::Output & output)
  {
    if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
    {
      SimpleNodes_.Insert(simpleNode);
      return;
    }

    Outputs_.Insert(&output);
  }*/

  /*bool
  IsAlive(const jlm::rvsdg::Output & output) const noexcept
  {
    if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
    {
      return SimpleNodes_.Contains(simpleNode);
    }

    return Outputs_.Contains(&output);
  }*/

  /*bool
  IsAlive(const rvsdg::Node & node) const noexcept
  {
    if (auto simpleNode = dynamic_cast<const jlm::rvsdg::SimpleNode *>(&node))
    {
      return SimpleNodes_.Contains(simpleNode);
    }

    for (size_t n = 0; n < node.noutputs(); n++)
    {
      if (IsAlive(*node.output(n)))
      {
        return true;
      }
    }

    return false;
  }*/

  static std::unique_ptr<Context>
  Create()
  {
    return std::make_unique<Context>();
  }

private:
  util::HashSet<const jlm::rvsdg::SimpleNode *> SimpleNodes_;
  util::HashSet<const jlm::rvsdg::Output *> Outputs_;
};



/** \brief Dead Node Elimination statistics class
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

PartialRedundancyElimination::~PartialRedundancyElimination() noexcept = default;

PartialRedundancyElimination::PartialRedundancyElimination()
    : Transformation("PartialRedundancyElimination")
{}



void PartialRedundancyElimination::dump_region(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  std::string name = node.DebugString() + std::to_string(node.GetNodeId());
  size_t reg_counter = 0;

  MatchType(node, [&name, &reg_counter](rvsdg::StructuralNode& sn)
  {
    for (auto& reg : sn.Subregions())
    {
      auto my_graph_writer = jlm::util::graph::Writer();

      jlm::llvm::dot::LlvmDotWriter my_dot_writer;
      my_dot_writer.WriteGraphs(my_graph_writer , reg, false);

      std::string full_name = name+std::to_string(reg_counter++)+".dot";
      std::cout<< TR_RED<<full_name<<TR_RESET<<std::endl;

      std::ofstream my_dot_oss (full_name);
      my_graph_writer.OutputAllGraphs(my_dot_oss, jlm::util::graph::OutputFormat::Dot);
      std::cout << TR_GREEN << "-------------------------------------" << TR_RESET << std::endl;
    }
  });

}


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

void PartialRedundancyElimination::TraverseSubRegions(rvsdg::Region& reg, void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node& node))
{
  IndentMan indenter = IndentMan();
  for (rvsdg::Node* node : rvsdg::TopDownTraverser(&reg))
  {
    cb(this, *node);
    MatchType(*node, [this,cb](rvsdg::StructuralNode& sn)
    {
      for (auto& reg : sn.Subregions())
      {
        this->TraverseSubRegions(reg, cb);
      }
    });
  }
}


/*
void TraverseSubTrees(rvsdg::StructuralNode& node, void(*cb)(PartialRedundancyElimination* pe, rvsdg::StructuralNode& node));
*/

void PartialRedundancyElimination::dump_node(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  std::cout << ind() << TR_BLUE << node.DebugString() << TR_CYAN<<node.GetNodeId() << TR_RESET;
  for (size_t i = 0; i < node.noutputs(); i++)
  {
    auto k_present = pe->output_hashes.find(node.output(i));
    if (k_present != pe->output_hashes.end() )
    {
      size_t h = pe->output_hashes[node.output(i)];
      std::string color = (pe->hash_count(h) > 1 ? TR_GREEN : TR_YELLOW);

      MatchType(node.GetOperation(),[&color](const jlm::llvm::CallOperation& op){color = TR_CYAN;});
      MatchType(node, [&color](rvsdg::LambdaNode& lm){color = TR_ORANGE;});

      std::cout << " : " << color << h << TR_RESET;
    }
  }
  MatchType(node, [pe](rvsdg::LambdaNode& lm)
  {
    for (auto& param : lm.GetFunctionArguments())
    {
      if (pe->output_hashes.find(param) != pe->output_hashes.end())
      {
        size_t h = pe->output_hashes[param];
        std::cout << ( pe->hash_count(h) > 1 ? TR_RED : TR_ORANGE) << " : " << h << TR_RESET;
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
      pe->register_hash(node.output(0), h);
    }
    ,
    [pe, &node](const jlm::llvm::CallOperation& op)
    {
      auto s = node.DebugString() + std::to_string(node.GetNodeId() );
      for (size_t i = 0; i < node.noutputs(); i++){
        pe->register_hash_for_output(node.output(i), s, i);
      }
    }
  );

  /* Add each lambda parameter as a leaf hash for hashing within its body */
  MatchType(node, [pe, &node](rvsdg::LambdaNode& lm)
  {
    auto fargs = lm.GetFunctionArguments();
    auto s = node.DebugString() + std::to_string(node.GetNodeId());
    for (size_t i = 0; i < fargs.size(); i++){
      pe->register_hash_for_output(fargs[i], node.DebugString(), i);
    }
  });
}

void PartialRedundancyElimination::hash_gamma(PartialRedundancyElimination* pe, rvsdg::Node& node)
{
  MatchType(node, [pe](rvsdg::GammaNode& node)
  {
    for (auto ev : node.GetEntryVars())
    {
      rvsdg::Output* origin = ev.input->origin();
      if (pe->output_has_hash(origin))
      {
        size_t h = pe->output_hashes[origin];
        for (rvsdg::Output* brarg : ev.branchArgument)
        {
          pe->register_hash(brarg, h);
        }
      }
    }
  });
}

void PartialRedundancyElimination::hash_node(PartialRedundancyElimination *pe, rvsdg::Node& node)
{
  /*Match by operation*/
  MatchType(node.GetOperation(),
    [pe, &node](const rvsdg::BinaryOperation& op){hash_bin(pe, node);}
  );
  /*Match by node type*/
  MatchType(node, [pe](rvsdg::GammaNode& node){hash_gamma(pe, node);});
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
      if (pe->output_hashes.find(node.input(i)->origin()) != pe->output_hashes.end())
      {
        size_t hash_in = pe->output_hashes[node.input(i)->origin()];
        if (op.is_commutative() && op.is_associative())
        {
          h += hasher(std::to_string(hash_in));
        }else
        {
          h |= hasher(std::to_string(hash_in)) * (i+1);
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
    if (was_hashable)
    {
      pe->output_hashes.insert( {node.output(0), h} );
      pe->register_hash(h);
    }

  });
}

void
PartialRedundancyElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  std::cout << TR_BLUE << "Hello JLM its me." << TR_RESET << std::endl;






  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  this->TraverseSubRegions(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_region);
  this->TraverseSubRegions(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_node);
  std::cout << TR_RED << "================================================================" << TR_RESET << std::endl;
  this->TraverseSubRegions(rvsdg.GetRootRegion(), PartialRedundancyElimination::register_leaf_hash);
  std::cout << TR_RED << "================================================================" << TR_RESET << std::endl;
  this->TraverseSubRegions(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_node);
  std::cout << TR_BLUE << "================================================================" << TR_RESET << std::endl;
  this->TraverseSubRegions(rvsdg.GetRootRegion(), PartialRedundancyElimination::hash_node);
  std::cout << TR_PINK << "================================================================" << TR_RESET << std::endl;
  this->TraverseSubRegions(rvsdg.GetRootRegion(), PartialRedundancyElimination::dump_node);

  std::cout << TR_GREEN << "=================================================" << TR_RESET << std::endl;
}

}
