/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>
#include <queue>
#include <unordered_set>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/control.hpp>

namespace
{

using namespace jlm::llvm;
using namespace jlm::rvsdg;
using namespace jlm::util;

void
CompareRegions(const Region & region1, const Region & region2);

/**
 * \brief Compares two RVSDG types for equality.
 *
 * \param type1 The first type to compare.
 * \param type2 The second type to compare.
 */
void
CompareTypes(const Type & type1, const Type & type2)
{
  ASSERT_EQ(type1, type2) << "Type mismatch: expected " << type1.debug_string() << " but got "
                          << type2.debug_string();
}

/**
 * \brief Compares two RVSDG nodes for equality.
 *
 * \param node1 The first node to compare.
 * \param node2 The second node to compare.
 */
void
CompareNodes(const Node & node1, const Node & node2)
{
  // Check if structural node
  if (auto * snode1 = dynamic_cast<const StructuralNode *>(&node1))
  {
    auto * snode2 = assertedCast<const StructuralNode>(&node2);

    ASSERT_EQ(snode1->GetOperation(), snode2->GetOperation())
        << "StructuralNode operation mismatch: node1=" << &node1 << ", node2=" << &node2;
    ASSERT_EQ(snode1->nsubregions(), snode2->nsubregions())
        << "StructuralNode nsubregions mismatch: node1=" << &node1 << ", node2=" << &node2 << ": "
        << snode1->nsubregions() << " vs " << snode2->nsubregions();

    // Compare each region recursively
    for (size_t r = 0; r < snode1->nsubregions(); ++r)
    {
      CompareRegions(*snode1->subregion(r), *snode2->subregion(r));
    }

    // Compare inputs
    ASSERT_EQ(snode1->ninputs(), snode2->ninputs())
        << "StructuralNode ninputs mismatch: node1=" << &node1 << ", node2=" << &node2 << ": "
        << snode1->ninputs() << " vs " << snode2->ninputs();
    for (size_t i = 0; i < snode1->ninputs(); ++i)
    {
      CompareTypes(*snode1->input(i)->Type(), *snode2->input(i)->Type());
    }

    // Compare outputs
    ASSERT_EQ(snode1->noutputs(), snode2->noutputs())
        << "StructuralNode noutputs mismatch: node1=" << &node1 << ", node2=" << &node2 << ": "
        << snode1->noutputs() << " vs " << snode2->noutputs();
    for (size_t i = 0; i < snode1->noutputs(); ++i)
    {
      CompareTypes(*snode1->output(i)->Type(), *snode2->output(i)->Type());
    }
    return;
  }

  // Check if simple node
  if (auto * simp1 = dynamic_cast<const SimpleNode *>(&node1))
  {
    auto * simp2 = assertedCast<const SimpleNode>(&node2);

    ASSERT_EQ(simp1->GetOperation(), simp2->GetOperation())
        << "SimpleNode operation mismatch: node1=" << &node1 << ", node2=" << &node2;

    // Compare inputs
    ASSERT_EQ(simp1->ninputs(), simp2->ninputs())
        << "SimpleNode ninputs mismatch: " << simp1->ninputs() << " vs " << simp2->ninputs();
    for (size_t i = 0; i < simp1->ninputs(); ++i)
    {
      CompareTypes(*simp1->input(i)->Type(), *simp2->input(i)->Type());
    }

    // Compare outputs
    ASSERT_EQ(simp1->noutputs(), simp2->noutputs())
        << "SimpleNode noutputs mismatch: " << simp1->noutputs() << " vs " << simp2->noutputs();
    for (size_t i = 0; i < simp1->noutputs(); ++i)
    {
      CompareTypes(*simp1->output(i)->Type(), *simp2->output(i)->Type());
    }
    return;
  }
  ADD_FAILURE() << "Node comparison failed: could not identify node type";
}

/**
 * \brief Compares two RVSDG regions for equality by traversing through results
 * and verifying the same graph structure exists in both regions.
 *
 * \param region1 The first region to compare.
 * \param region2 The second region to compare.
 */
void
CompareRegions(const Region & region1, const Region & region2)
{
  // Check number of arguments
  ASSERT_EQ(region1.narguments(), region2.narguments())
      << "Region narguments mismatch: " << region1.narguments() << " vs " << region2.narguments();
  for (size_t i = 0; i < region1.narguments(); ++i)
  {
    CompareTypes(*region1.argument(i)->Type(), *region2.argument(i)->Type());
  }

  // Check number of results
  ASSERT_EQ(region1.nresults(), region2.nresults())
      << "Region nresults mismatch: " << region1.nresults() << " vs " << region2.nresults();
  for (size_t i = 0; i < region1.nresults(); ++i)
  {
    CompareTypes(*region1.result(i)->Type(), *region2.result(i)->Type());
  }

  // Check node count
  ASSERT_EQ(region1.numNodes(), region2.numNodes())
      << "Region numNodes mismatch: " << region1.numNodes() << " vs " << region2.numNodes();

  // Traverse starting from results - follow backwards through dataflow
  std::unordered_set<const Node *> visited1, visited2;
  std::queue<std::pair<const Node *, const Node *>> nodeQueue;

  if (region1.nresults() > 0)
  {
    // Start from each result and find the node that produces it
    for (size_t i = 0; i < region1.nresults(); ++i)
    {
      auto * origin1 = region1.result(i)->origin();
      auto * origin2 = region2.result(i)->origin();
      ASSERT_TRUE(origin1 && origin2) << "Result origin is null at index " << i << ": "
                                      << "region1=" << !!origin1 << ", region2=" << !!origin2;

      CompareTypes(*origin1->Type(), *origin2->Type());

      if (auto * n1 = TryGetOwnerNode<Node>(*origin1))
      {
        auto * n2 = TryGetOwnerNode<Node>(*origin2);
        ASSERT_NE(n2, nullptr);

        CompareNodes(*n1, *n2);

        visited1.insert(n1);
        visited2.insert(n2);
        nodeQueue.push({ n1, n2 });
      }
      else if (auto * arg1 = dynamic_cast<RegionArgument *>(origin1))
      {
        auto arg2 = assertedCast<RegionArgument>(origin2);
        CompareTypes(*arg1->Type(), *arg2->Type());
      }
      else
      {
        JLM_UNREACHABLE("This should not happen");
      }
    }
  }

  // BFS traversal - follow inputs backwards through the graph
  while (!nodeQueue.empty())
  {
    auto * n1 = nodeQueue.front().first;
    auto * n2 = nodeQueue.front().second;
    nodeQueue.pop();

    for (size_t j = 0; j < n1->ninputs(); ++j)
    {
      auto * origin1 = n1->input(j)->origin();
      auto * origin2 = n2->input(j)->origin();

      ASSERT_TRUE(origin1 && origin2)
          << "Input origin mismatch for node inputs at index " << j << ": "
          << "node1=" << n1 << ", node2=" << n2 << ", origin1=" << !!origin1
          << ", origin2=" << !!origin2;

      if (auto * next1 = TryGetOwnerNode<Node>(*origin1))
      {
        auto * next2 = TryGetOwnerNode<Node>(*origin2);
        ASSERT_NE(next2, nullptr);

        if (!visited1.count(next1))
        {
          CompareNodes(*next1, *next2);

          visited1.insert(next1);
          visited2.insert(next2);
          nodeQueue.push({ next1, next2 });
        }
      }
      else if (auto * arg1 = dynamic_cast<RegionArgument *>(origin1))
      {
        auto arg2 = assertedCast<RegionArgument>(origin2);
        CompareTypes(*arg1->Type(), *arg2->Type());
      }
      else
      {
        JLM_UNREACHABLE("This should not happen");
      }
    }
  }

  // Verify all nodes were visited
  ASSERT_EQ(visited1.size(), region1.numNodes())
      << "Node count mismatch after traversal: visited=" << visited1.size()
      << ", expected=" << region1.numNodes();
  ASSERT_EQ(visited2.size(), region2.numNodes())
      << "Node count mismatch for region2 after traversal: visited=" << visited2.size()
      << ", expected=" << region2.numNodes();
}

/**
 * \brief Compares two LlvmRvsdgModule instances for equality.
 *
 * \param module1 The first module to compare.
 * \param module2 The second module to compare.
 */
void
CompareModules(const LlvmRvsdgModule & module1, const LlvmRvsdgModule & module2)
{
  CompareRegions(module1.Rvsdg().GetRootRegion(), module2.Rvsdg().GetRootRegion());
}

/**
 * \brief Tests that an RVSDG graph roundtrips through MLIR and produces an identical graph.
 *
 * This function performs the following steps:
 * 1. Converts the input RVSDG to MLIR using JlmToMlirConverter
 * 2. Converts the MLIR back to RVSDG using MlirToJlmConverter
 * 3. Compares the roundtrip result with the original graph for deep structural equality
 *
 * \param originalModule The original RVSDG module to test.
 * \param testName A descriptive name for this test (used in failure messages).
 */
void
TestRvsdgRoundtrip(const LlvmRvsdgModule & originalModule, const char * testName)
{
  using namespace jlm::mlir;
  using namespace jlm::rvsdg;

  // Convert original RVSDG to MLIR
  JlmToMlirConverter mlirgen;
  auto omega = mlirgen.ConvertModule(originalModule);

  // Convert MLIR back to RVSDG
  std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
  rootBlock->push_back(omega);
  auto roundTripModule = MlirToJlmConverter::CreateAndConvert(rootBlock);

  // Compare the modules
  CompareModules(originalModule, *roundTripModule);
}

} // namespace

TEST(RvsdgRoundtripTests, TestGamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType = BitType::Create(1);
  auto functionType = FunctionType::Create({ bitType, bitType, bitType }, { bitType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode =
      MatchOperation::CreateNode(*lambda->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);
  auto gamma = GammaNode::create(matchNode.output(0), 2);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto entryVar2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto exitVar = gamma->AddExitVar({ entryVar1.branchArgument[0], entryVar2.branchArgument[1] });

  auto func = lambda->finalize({ exitVar.output });
  GraphExport::Create(*func, "");

  TestRvsdgRoundtrip(rvsdgModule, "TestGamma");
}
