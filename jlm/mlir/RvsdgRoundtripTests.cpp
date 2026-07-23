/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <queue>
#include <unordered_set>

#include <jlm/rvsdg/delta.hpp> // DeltaNode::GetContextVars()
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp> // PhiNode::GetContextVars()

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>
#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/control.hpp>

namespace
{

using namespace jlm::llvm;
using namespace jlm::rvsdg;
using namespace jlm::util;

void
CompareRegions(const Region & region1, const Region & region2);

/**
 * \brief Compares two RVSDG types for structural equality using fail-fast assertions.
 */
void
CompareTypes(const Type & type1, const Type & type2)
{
  // If same type class and equal by operator==, return immediately
  if (type1 == type2)
    return;

  if (auto * bitType1 = dynamic_cast<const BitType *>(&type1))
  {
    auto * bitType2 = assertedCast<const BitType>(&type2);
    ASSERT_TRUE(bitType1->nbits() == bitType2->nbits())
        << "BitType mismatch: expected " << type1.debug_string() << " but got "
        << type2.debug_string();
    return;
  }

  // Handle StructType comparison - compare by element types and properties
  if (auto * structType1 = dynamic_cast<const StructType *>(&type1))
  {
    auto * structType2 = assertedCast<const StructType>(&type2);

    // Compare element count using numElements()
    ASSERT_TRUE(structType1->numElements() == structType2->numElements())
        << "StructType element count mismatch: expected " << type1.debug_string() << " but got "
        << type2.debug_string();
    // Compare each element type recursively using getElementType(index)
    for (size_t i = 0; i < structType1->numElements(); ++i)
    {
      CompareTypes(*structType1->getElementType(i), *structType2->getElementType(i));
    }
    // Compare packed status using IsPacked()
    ASSERT_TRUE(structType1->IsPacked() == structType2->IsPacked())
        << "StructType packed mismatch: expected " << type1.debug_string() << " but got "
        << type2.debug_string();
    return;
  }

  // Handle ArrayType comparison - compare by element type and size
  if (auto * arrayType1 = dynamic_cast<const ArrayType *>(&type1))
  {
    auto * arrayType2 = assertedCast<const ArrayType>(&type2);
    CompareTypes(arrayType1->element_type(), arrayType2->element_type());
    ASSERT_TRUE(arrayType1->nelements() == arrayType2->nelements())
        << "ArrayType element count mismatch: expected " << type1.debug_string() << " but got "
        << type2.debug_string();
    return;
  }

  // Handle FunctionType comparison - compare by argument and result types
  if (auto * fnType1 = dynamic_cast<const FunctionType *>(&type1))
  {
    auto * fnType2 = assertedCast<const FunctionType>(&type2);
    ASSERT_TRUE(
        fnType1->NumArguments() == fnType2->NumArguments()
        && fnType1->NumResults() == fnType2->NumResults())
        << "FunctionType argument/result count mismatch: expected " << type1.debug_string()
        << " but got " << type2.debug_string();
    for (size_t i = 0; i < fnType1->NumArguments(); ++i)
    {
      CompareTypes(fnType1->ArgumentType(i), fnType2->ArgumentType(i));
    }
    for (size_t i = 0; i < fnType1->NumResults(); ++i)
    {
      CompareTypes(fnType1->ResultType(i), fnType2->ResultType(i));
    }
    return;
  }

  // Fallback to regular equality for any remaining types
  ASSERT_TRUE(type1 == type2) << "Type mismatch: expected " << type1.debug_string() << " but got "
                              << type2.debug_string();
}

/**
 * \brief Compares two operations for equality, handling different but equivalent
 * operation types.
 */
void
CompareOperations(const Operation & op1, const Operation & op2)
{
  // Handle comparison for operations that use pointer identity for
  // its operator== which would fail

  if (auto * alloca1 = dynamic_cast<const AllocaOperation *>(&op1))
  {
    auto * alloca2 = assertedCast<const AllocaOperation>(&op2);
    CompareTypes(*alloca1->allocatedType(), *alloca2->allocatedType());
    ASSERT_TRUE(alloca1->alignment() == alloca2->alignment())
        << "Alloca mismatch: " << op1.debug_string() << " vs " << op2.debug_string();
    return;
  }

  if (auto * malloc1 = dynamic_cast<const MallocOperation *>(&op1))
  {
    auto * malloc2 = assertedCast<const MallocOperation>(&op2);
    CompareTypes(malloc1->getSizeType(), malloc2->getSizeType());
    return;
  }

  if (auto * free1 = dynamic_cast<const FreeOperation *>(&op1))
  {
    auto * free2 = assertedCast<const FreeOperation>(&op2);
    ASSERT_TRUE(free1->narguments() == free2->narguments())
        << "Free mismatch: " << op1.debug_string() << " vs " << op2.debug_string();
    return;
  }

  if (auto * constDataArr1 = dynamic_cast<const ConstantDataArrayOperation *>(&op1))
  {
    auto * constDataArr2 = assertedCast<const ConstantDataArrayOperation>(&op2);
    CompareTypes(*constDataArr1->result(0), *constDataArr2->result(0));
    return;
  }

  if (auto * constArr1 = dynamic_cast<const ConstantArrayOperation *>(&op1))
  {
    auto * constArr2 = assertedCast<const ConstantArrayOperation>(&op2);
    CompareTypes(*constArr1->result(0), *constArr2->result(0));
    return;
  }

  if (auto * constAggZero1 = dynamic_cast<const ConstantAggregateZeroOperation *>(&op1))
  {
    auto * constAggZero2 = assertedCast<const ConstantAggregateZeroOperation>(&op2);
    CompareTypes(*constAggZero1->result(0), *constAggZero2->result(0));
    return;
  }

  if (auto * constStruct1 = dynamic_cast<const ConstantStructOperation *>(&op1))
  {
    auto * constStruct2 = assertedCast<const ConstantStructOperation>(&op2);
    CompareTypes(*constStruct1->result(0), *constStruct2->result(0));
    return;
  }

  if (auto * call1 = dynamic_cast<const CallOperation *>(&op1))
  {
    auto * call2 = assertedCast<const CallOperation>(&op2);
    CompareTypes(*call1->GetFunctionType(), *call2->GetFunctionType());
    return;
  }

  if (auto * gep1 = dynamic_cast<const GetElementPtrOperation *>(&op1))
  {
    auto * gep2 = assertedCast<const GetElementPtrOperation>(&op2);
    CompareTypes(*gep1->getPointeeType(), *gep2->getPointeeType());
    return;
  }

  if (auto * memcpy1 = dynamic_cast<const jlm::llvm::MemCpyNonVolatileOperation *>(&op1))
  {
    auto * memcpy2 = assertedCast<const jlm::llvm::MemCpyNonVolatileOperation>(&op2);
    CompareTypes(memcpy1->LengthType(), memcpy2->LengthType());
    ASSERT_TRUE(memcpy1->NumMemoryStates() == memcpy2->NumMemoryStates())
        << "MemCpyNonVolatile mismatch: " << op1.debug_string() << " vs " << op2.debug_string();
    return;
  }

  if (auto * vmemcpy1 = dynamic_cast<const jlm::llvm::MemCpyVolatileOperation *>(&op1))
  {
    auto * vmemcpy2 = assertedCast<const jlm::llvm::MemCpyVolatileOperation>(&op2);
    CompareTypes(vmemcpy1->LengthType(), vmemcpy2->LengthType());
    ASSERT_TRUE(vmemcpy1->NumMemoryStates() == vmemcpy2->NumMemoryStates())
        << "MemCpyVolatile mismatch: " << op1.debug_string() << " vs " << op2.debug_string();
    return;
  }

  if (auto * lambda1 = dynamic_cast<const jlm::llvm::LlvmLambdaOperation *>(&op1))
  {
    auto * lambda2 = assertedCast<const jlm::llvm::LlvmLambdaOperation>(&op2);

    JLM_ASSERT(lambda1->name() == lambda2->name());
    JLM_ASSERT(lambda1->linkage() == lambda2->linkage());
    JLM_ASSERT(lambda1->callingConvention() == lambda2->callingConvention());

    auto type1 = lambda1->type();
    auto type2 = lambda2->type();

    JLM_ASSERT(type1.NumArguments() == type2.NumArguments());
    for (size_t i = 0; i < type1.NumArguments(); ++i)
    {
      CompareTypes(type1.ArgumentType(i), type2.ArgumentType(i));
    }

    for (size_t i = 0; i < type1.NumResults(); ++i)
    {
      CompareTypes(type1.ResultType(i), type2.ResultType(i));
    }
  }

  // If same type, use the regular operator==
  if (typeid(op1) == typeid(op2))
  {
    ASSERT_TRUE(op1 == op2) << "Same-type operation inequality: " << op1.debug_string() << " vs "
                            << op2.debug_string();
    return;
  }

  FAIL() << "Unknown operation comparison: " << op1.debug_string() << " vs " << op2.debug_string();
}

/**
 * \brief Compares two RVSDG nodes for equality.
 */
void
CompareNodes(const Node & node1, const Node & node2)
{
  // Check if structural node
  if (auto * snode1 = dynamic_cast<const StructuralNode *>(&node1))
  {
    auto * snode2 = assertedCast<const StructuralNode>(&node2);

    CompareOperations(snode1->GetOperation(), snode2->GetOperation());
    ASSERT_EQ(snode1->nsubregions(), snode2->nsubregions())
        << "StructuralNode nsubregions mismatch";

    // Compare each region recursively
    for (size_t r = 0; r < snode1->nsubregions(); ++r)
    {
      CompareRegions(*snode1->subregion(r), *snode2->subregion(r));
    }

    // Compare inputs and outputs types
    ASSERT_EQ(snode1->ninputs(), snode2->ninputs()) << "StructuralNode ninputs mismatch";
    for (size_t i = 0; i < snode1->ninputs(); ++i)
    {
      CompareTypes(*snode1->input(i)->Type(), *snode2->input(i)->Type());
    }

    ASSERT_EQ(snode1->noutputs(), snode2->noutputs()) << "StructuralNode noutputs mismatch";
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

    CompareOperations(simp1->GetOperation(), simp2->GetOperation());

    // Compare inputs
    ASSERT_EQ(simp1->ninputs(), simp2->ninputs()) << "SimpleNode ninputs mismatch";
    for (size_t i = 0; i < simp1->ninputs(); ++i)
    {
      CompareTypes(*simp1->input(i)->Type(), *simp2->input(i)->Type());
    }

    // Compare outputs
    ASSERT_EQ(simp1->noutputs(), simp2->noutputs()) << "SimpleNode noutputs mismatch";
    for (size_t i = 0; i < simp1->noutputs(); ++i)
    {
      CompareTypes(*simp1->output(i)->Type(), *simp2->output(i)->Type());
    }
    return;
  }
  ADD_FAILURE() << "Node comparison failed: could not identify node type";
}

/**
 * \brief Collect context variable origin node pairs from a structural node.
 *
 * For LambdaNode, DeltaNode, and PhiNode, this extracts the owner nodes of all
 * context variable inputs. This enables BFS to traverse into subregions via
 * their context variable dependencies that are not reachable through normal
 * input→origin edges.
 *
 * Context variables are realized as:
 *   - An Input to the structural node (supplying the value)
 *   - An Output argument in the subregion (binding the value internally)
 */
static std::vector<std::pair<const Node *, const Node *>>
CollectContextVarOrigins(const Node & node1, const Node & node2)
{
  std::vector<std::pair<const Node *, const Node *>> origins;

  // LambdaNode case
  if (auto * lambda1 = dynamic_cast<const LambdaNode *>(&node1))
  {
    auto * lambda2 = assertedCast<const LambdaNode>(&node2);
    auto cvList1 = lambda1->GetContextVars();
    auto cvList2 = lambda2->GetContextVars();

    auto it1 = cvList1.begin(), it2 = cvList2.begin();
    while (it1 != cvList1.end() && it2 != cvList2.end())
    {
      if (auto * origin1 = TryGetOwnerNode<Node>(*it1->input->origin()))
      {
        auto * origin2 = TryGetOwnerNode<Node>(*it2->input->origin());
        if (origin2)
          origins.push_back({ origin1, origin2 });
      }
      ++it1;
      ++it2;
    }

    JLM_ASSERT(
        std::distance(cvList1.begin(), cvList1.end())
        == std::distance(cvList2.begin(), cvList2.end()));
  }
  // DeltaNode case - same pattern as LambdaNode
  else if (auto * delta1 = dynamic_cast<const DeltaNode *>(&node1))
  {
    auto * delta2 = assertedCast<const DeltaNode>(&node2);
    auto cvList1 = delta1->GetContextVars();
    auto cvList2 = delta2->GetContextVars();

    auto it1 = cvList1.begin(), it2 = cvList2.begin();
    while (it1 != cvList1.end() && it2 != cvList2.end())
    {
      if (auto * origin1 = TryGetOwnerNode<Node>(*it1->input->origin()))
      {
        auto * origin2 = TryGetOwnerNode<Node>(*it2->input->origin());
        if (origin2)
          origins.push_back({ origin1, origin2 });
      }
      ++it1;
      ++it2;
    }

    JLM_ASSERT(
        std::distance(cvList1.begin(), cvList1.end())
        == std::distance(cvList2.begin(), cvList2.end()));
  }
  // PhiNode case - same pattern as LambdaNode
  else if (auto * phi1 = dynamic_cast<const PhiNode *>(&node1))
  {
    auto * phi2 = assertedCast<const PhiNode>(&node2);
    auto cvList1 = phi1->GetContextVars();
    auto cvList2 = phi2->GetContextVars();

    auto it1 = cvList1.begin(), it2 = cvList2.begin();
    while (it1 != cvList1.end() && it2 != cvList2.end())
    {
      if (auto * origin1 = TryGetOwnerNode<Node>(*it1->input->origin()))
      {
        auto * origin2 = TryGetOwnerNode<Node>(*it2->input->origin());
        if (origin2)
          origins.push_back({ origin1, origin2 });
      }
      ++it1;
      ++it2;
    }

    JLM_ASSERT(
        std::distance(cvList1.begin(), cvList1.end())
        == std::distance(cvList2.begin(), cvList2.end()));
  }
  // SimpleNode and other non-structural nodes have no context variables - return empty vector

  return origins;
}

/**
 * \brief Check for context variables and traverse their origin nodes into the BFS queue.
 *
 * When a structural node (LambdaNode, DeltaNode, PhiNode) is visited during BFS,
 * its context variable inputs point to values that may not be reachable through
 * normal input→origin edge traversal from region results. This function:
 *   1. Collects all context variable origin node pairs using CollectContextVarOrigins()
 *   2. For each unvisited origin pair, performs CompareNodes() and adds to BFS queue
 *   3. Recursively expands from each newly discovered origin (for chains of dependencies)
 */
static void
CheckForContextVariablesAndTravers(
    const Node & node1,
    const Node & node2,
    std::unordered_set<const Node *> & visited1,
    std::unordered_set<const Node *> & visited2,
    std::queue<std::pair<const Node *, const Node *>> & nodeQueue)
{
  auto cvOrigins = CollectContextVarOrigins(node1, node2);

  for (auto [origin1, origin2] : cvOrigins)
  {
    if (!visited1.count(origin1))
    {
      visited1.insert(origin1);
      visited2.insert(origin2);

      // Compare the context variable origin nodes structurally
      CompareNodes(*origin1, *origin2);

      // Add to BFS queue for further expansion (traverse their inputs)
      nodeQueue.push({ origin1, origin2 });

      // Recursively expand from this origin's context variables too.
      // This handles cases like: Lambda A has context var from Lambda B,
      // which itself has context vars we need to visit.
      CheckForContextVariablesAndTravers(*origin1, *origin2, visited1, visited2, nodeQueue);
    }
  }
}

/**
 * \brief Compares two RVSDG regions for equality by traversing through results
 * and verifying the same graph structure exists in both regions.
 */
// Debug: Add counter for lambda comparisons
void
CompareRegions(const Region & region1, const Region & region2)
{
  // Check number of arguments and results
  ASSERT_EQ(region1.narguments(), region2.narguments()) << "Region narguments mismatch";
  for (size_t i = 0; i < region1.narguments(); ++i)
  {
    auto * arg1 = region1.argument(i);
    auto * arg2 = region2.argument(i);
    CompareTypes(*arg1->Type(), *arg2->Type());
  }

  ASSERT_EQ(region1.nresults(), region2.nresults());
  for (size_t i = 0; i < region1.nresults(); ++i)
  {
    CompareTypes(*region1.result(i)->Type(), *region2.result(i)->Type());
  }

  size_t count1 = region1.numNodes();
  size_t count2 = region2.numNodes();

  std::unordered_set<const Node *> visited1, visited2;
  std::queue<std::pair<const Node *, const Node *>> nodeQueue;

  // Start from each result and find the node that produces it
  for (size_t i = 0; i < region1.nresults(); ++i)
  {
    auto * origin1 = region1.result(i)->origin();
    auto * origin2 = region2.result(i)->origin();

    if (!origin1 || !origin2)
    {
      ADD_FAILURE() << "Result origin is null at index " << i;
      return;
    }

    CompareTypes(*origin1->Type(), *origin2->Type());

    if (auto * node1 = TryGetOwnerNode<Node>(*origin1))
    {
      auto * node2 = TryGetOwnerNode<Node>(*origin2);
      ASSERT_NE(node2, nullptr);

      visited1.insert(node1);
      visited2.insert(node2);
      nodeQueue.push({ node1, node2 });
      CompareNodes(*node1, *node2);
      CheckForContextVariablesAndTravers(*node1, *node2, visited1, visited2, nodeQueue);
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

  // BFS traversal - follow inputs backwards through the graph
  while (!nodeQueue.empty())
  {
    auto * node1 = nodeQueue.front().first;
    auto * node2 = nodeQueue.front().second;
    nodeQueue.pop();

    for (size_t j = 0; j < node1->ninputs(); ++j)
    {
      auto * origin1 = node1->input(j)->origin();
      auto * origin2 = node2->input(j)->origin();

      if (!origin1 || !origin2)
      {
        ADD_FAILURE() << "Input origin mismatch for node inputs at index " << j;
        return;
      }

      if (auto * next1 = TryGetOwnerNode<Node>(*origin1))
      {
        auto * next2 = TryGetOwnerNode<Node>(*origin2);
        ASSERT_NE(next2, nullptr);

        if (!visited1.count(next1))
        {
          visited1.insert(next1);
          visited2.insert(next2);
          nodeQueue.push({ next1, next2 });
          CompareNodes(*next1, *next2);
          CheckForContextVariablesAndTravers(*next1, *next2, visited1, visited2, nodeQueue);
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

  // Phase 2: Handle any remaining unvisited nodes not reachable from results.
  // These include nodes in nested subregions (gamma arms), structural nodes with
  // outputs not connected to region results, etc. Match by operation type +
  // structural properties instead of fragile debug_string comparison.
  auto SameOpType = [](const Operation & op1, const Operation & op2) noexcept -> bool
  {
    return typeid(op1) == typeid(op2);
  };

  while (visited1.size() < count1)
  {
    bool found = false;

    for (const auto & node : region1.Nodes())
    {
      if (visited1.count(&node))
        continue;

      // Find an unvisited matching node in region2 by operation type + inputs/outputs
      const Node * matchNode = nullptr;
      for (const auto & n : region2.Nodes())
      {
        if (visited2.count(&n))
          continue;

        // Must be same concrete type, and compatible structural properties
        if (SameOpType(node.GetOperation(), n.GetOperation()) && node.ninputs() == n.ninputs()
            && node.noutputs() == n.noutputs())
        {
          matchNode = &n;
          break;
        }
      }

      if (!matchNode)
        continue;

      CompareNodes(node, *matchNode);
      visited1.insert(&node);
      visited2.insert(matchNode);
      found = true;
      break; // restart to find more nodes
    }

    if (!found)
      break;
  }

  // Final assertion: all nodes must have been visited and compared.
  ASSERT_EQ(visited1.size(), count1) << "Not all nodes in region1 were visited during BFS";
  ASSERT_EQ(visited2.size(), count2) << "Not all nodes in region2 were visited during BFS";
}

/**
 * \brief Compares two LlvmRvsdgModule instances for equality.
 */
void
CompareModules(const LlvmRvsdgModule & module1, const LlvmRvsdgModule & module2)
{
  CompareRegions(module1.Rvsdg().GetRootRegion(), module2.Rvsdg().GetRootRegion());
}

/**
 * \brief Tests that an RVSDG graph roundtrips through MLIR.
 */
void
TestRvsdgRoundtrip(const LlvmRvsdgModule & originalModule, const char * testName)
{
  using namespace jlm::mlir;
  using namespace jlm::rvsdg;

  JlmToMlirConverter mlirgen;
  auto omega = mlirgen.ConvertModule(originalModule);

  std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
  rootBlock->push_back(omega);

  auto roundTripModule = MlirToJlmConverter::CreateAndConvert(rootBlock);

  CompareModules(originalModule, *roundTripModule);
}

} // namespace

// Tests for all RVSDG graphs defined in jlm/llvm/TestRvsdgs.cpp

TEST(RvsdgRoundtripTests, TestTheta)
{
  ::jlm::llvm::ThetaTest test;
  TestRvsdgRoundtrip(test.module(), "TestTheta");
}

TEST(RvsdgRoundtripTests, TestStoreTest1)
{
  ::jlm::llvm::StoreTest1 test;
  TestRvsdgRoundtrip(test.module(), "StoreTest1");
}

TEST(RvsdgRoundtripTests, TestStoreTest2)
{
  ::jlm::llvm::StoreTest2 test;
  TestRvsdgRoundtrip(test.module(), "StoreTest2");
}

TEST(RvsdgRoundtripTests, TestLoadTest1)
{
  ::jlm::llvm::LoadTest1 test;
  TestRvsdgRoundtrip(test.module(), "LoadTest1");
}

TEST(RvsdgRoundtripTests, TestLoadTest2)
{
  ::jlm::llvm::LoadTest2 test;
  TestRvsdgRoundtrip(test.module(), "LoadTest2");
}

TEST(RvsdgRoundtripTests, TestLoadFromUndef)
{
  ::jlm::llvm::LoadFromUndefTest test;
  TestRvsdgRoundtrip(test.module(), "LoadFromUndef");
}

TEST(RvsdgRoundtripTests, TestGetElementPtr)
{
  ::jlm::llvm::GetElementPtrTest test;
  TestRvsdgRoundtrip(test.module(), "GetElementPtr");
}

TEST(RvsdgRoundtripTests, TestBitCast)
{
  ::jlm::llvm::BitCastTest test;
  TestRvsdgRoundtrip(test.module(), "BitCast");
}

TEST(RvsdgRoundtripTests, TestBits2Ptr)
{
  ::jlm::llvm::Bits2PtrTest test;
  TestRvsdgRoundtrip(test.module(), "Bits2Ptr");
}

TEST(RvsdgRoundtripTests, TestConstantPointerNull)
{
  ::jlm::llvm::ConstantPointerNullTest test;
  TestRvsdgRoundtrip(test.module(), "ConstantPointerNull");
}

TEST(RvsdgRoundtripTests, TestCallTest1)
{
  ::jlm::llvm::CallTest1 test;
  TestRvsdgRoundtrip(test.module(), "CallTest1");
}

TEST(RvsdgRoundtripTests, TestCallTest2)
{
  ::jlm::llvm::CallTest2 test;
  TestRvsdgRoundtrip(test.module(), "CallTest2");
}

TEST(RvsdgRoundtripTests, TestIndirectCallTest1)
{
  ::jlm::llvm::IndirectCallTest1 test;
  TestRvsdgRoundtrip(test.module(), "IndirectCallTest1");
}

TEST(RvsdgRoundtripTests, TestIndirectCallTest2)
{
  ::jlm::llvm::IndirectCallTest2 test;
  TestRvsdgRoundtrip(test.module(), "IndirectCallTest2");
}

TEST(RvsdgRoundtripTests, TestExternalCallTest1)
{
  ::jlm::llvm::ExternalCallTest1 test;
  TestRvsdgRoundtrip(test.module(), "ExternalCallTest1");
}

TEST(RvsdgRoundtripTests, TestExternalCallTest2)
{
  ::jlm::llvm::ExternalCallTest2 test;
  TestRvsdgRoundtrip(test.module(), "ExternalCallTest2");
}

TEST(RvsdgRoundtripTests, TestDeltaTest1)
{
  ::jlm::llvm::DeltaTest1 test;
  TestRvsdgRoundtrip(test.module(), "DeltaTest1");
}

TEST(RvsdgRoundtripTests, TestDeltaTest2)
{
  ::jlm::llvm::DeltaTest2 test;
  TestRvsdgRoundtrip(test.module(), "DeltaTest2");
}

TEST(RvsdgRoundtripTests, TestDeltaTest3)
{
  ::jlm::llvm::DeltaTest3 test;
  TestRvsdgRoundtrip(test.module(), "DeltaTest3");
}

TEST(RvsdgRoundtripTests, TestImportTest)
{
  ::jlm::llvm::ImportTest test;
  TestRvsdgRoundtrip(test.module(), "ImportTest");
}

TEST(RvsdgRoundtripTests, TestGammaTest2)
{
  ::jlm::llvm::GammaTest2 test;
  TestRvsdgRoundtrip(test.module(), "GammaTest2");
}

TEST(RvsdgRoundtripTests, TestExternalMemory)
{
  ::jlm::llvm::ExternalMemoryTest test;
  TestRvsdgRoundtrip(test.module(), "ExternalMemory");
}

TEST(RvsdgRoundtripTests, TestEscapedMemoryTest1)
{
  ::jlm::llvm::EscapedMemoryTest1 test;
  TestRvsdgRoundtrip(test.module(), "EscapedMemoryTest1");
}

TEST(RvsdgRoundtripTests, TestEscapedMemoryTest2)
{
  ::jlm::llvm::EscapedMemoryTest2 test;
  TestRvsdgRoundtrip(test.module(), "EscapedMemoryTest2");
}

TEST(RvsdgRoundtripTests, TestEscapedMemoryTest3)
{
  ::jlm::llvm::EscapedMemoryTest3 test;
  TestRvsdgRoundtrip(test.module(), "EscapedMemoryTest3");
}

TEST(RvsdgRoundtripTests, TestLinkedList)
{
  ::jlm::llvm::LinkedListTest test;
  TestRvsdgRoundtrip(test.module(), "LinkedList");
}

TEST(RvsdgRoundtripTests, TestAllMemoryNodes)
{
  ::jlm::llvm::AllMemoryNodesTest test;
  TestRvsdgRoundtrip(test.module(), "AllMemoryNodes");
}

TEST(RvsdgRoundtripTests, TestEscapingLocalFunction)
{
  ::jlm::llvm::EscapingLocalFunctionTest test;
  TestRvsdgRoundtrip(test.module(), "EscapingLocalFunction");
}

TEST(RvsdgRoundtripTests, TestFreeNull)
{
  ::jlm::llvm::FreeNullTest test;
  TestRvsdgRoundtrip(test.module(), "FreeNull");
}

TEST(RvsdgRoundtripTests, TestLambdaCallArgumentMismatch)
{
  ::jlm::llvm::LambdaCallArgumentMismatch test;
  TestRvsdgRoundtrip(test.module(), "LambdaCallArgumentMismatch");
}

TEST(RvsdgRoundtripTests, TestVariadicFunctionTest1)
{
  ::jlm::llvm::VariadicFunctionTest1 test;
  TestRvsdgRoundtrip(test.module(), "VariadicFunctionTest1");
}

TEST(RvsdgRoundtripTests, TestVariadicFunctionTest2)
{
  ::jlm::llvm::VariadicFunctionTest2 test;
  TestRvsdgRoundtrip(test.module(), "VariadicFunctionTest2");
}

TEST(RvsdgRoundtripTests, TestPhiTest1)
{
  ::jlm::llvm::PhiTest1 test;
  TestRvsdgRoundtrip(test.module(), "PhiTest1");
}

TEST(RvsdgRoundtripTests, TestPhiTest2)
{
  ::jlm::llvm::PhiTest2 test;
  TestRvsdgRoundtrip(test.module(), "PhiTest2");
}

TEST(RvsdgRoundtripTests, TestPhiWithDelta)
{
  ::jlm::llvm::PhiWithDeltaTest test;
  TestRvsdgRoundtrip(test.module(), "PhiWithDelta");
}

TEST(RvsdgRoundtripTests, TestMemcpy)
{
  ::jlm::llvm::MemcpyTest test;
  TestRvsdgRoundtrip(test.module(), "Memcpy");
}

TEST(RvsdgRoundtripTests, TestMemcpyTest2)
{
  ::jlm::llvm::MemcpyTest2 test;
  TestRvsdgRoundtrip(test.module(), "MemcpyTest2");
}

TEST(RvsdgRoundtripTests, TestMemcpyTest3)
{
  ::jlm::llvm::MemcpyTest3 test;
  TestRvsdgRoundtrip(test.module(), "MemcpyTest3");
}
