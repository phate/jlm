/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>
#include <queue>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>

namespace
{

using namespace jlm::llvm;
using namespace jlm::rvsdg;
using namespace jlm::util;

void
CompareNodes(const Node & node1, const Node & node2);

void
CompareRegions(const Region & region1, const Region & region2);

/**
 * \brief Compares two RVSDG types for structural equality.
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
        << "CompareTypes: BitType mismatch: expected " << type1.debug_string() << " but got "
        << type2.debug_string();
    return;
  }

  // Handle StructType comparison - compare by element types and properties
  if (auto * structType1 = dynamic_cast<const StructType *>(&type1))
  {
    auto * structType2 = assertedCast<const StructType>(&type2);

    // Compare element count using numElements()
    ASSERT_TRUE(structType1->numElements() == structType2->numElements())
        << "CompareTypes: StructType element count mismatch: expected " << type1.debug_string()
        << " but got " << type2.debug_string();
    // Compare each element type recursively using getElementType(index)
    for (size_t i = 0; i < structType1->numElements(); ++i)
    {
      CompareTypes(*structType1->getElementType(i), *structType2->getElementType(i));
    }
    // Compare packed status using IsPacked()
    ASSERT_TRUE(structType1->IsPacked() == structType2->IsPacked())
        << "CompareTypes: StructType packed mismatch: expected " << type1.debug_string()
        << " but got " << type2.debug_string();
    return;
  }

  // Handle ArrayType comparison - compare by element type and size
  if (auto * arrayType1 = dynamic_cast<const ArrayType *>(&type1))
  {
    auto * arrayType2 = assertedCast<const ArrayType>(&type2);
    CompareTypes(arrayType1->element_type(), arrayType2->element_type());
    ASSERT_TRUE(arrayType1->nelements() == arrayType2->nelements())
        << "CompareTypes: ArrayType element count mismatch: expected " << type1.debug_string()
        << " but got " << type2.debug_string();
    return;
  }

  // Handle FunctionType comparison - compare by argument and result types
  if (auto * fnType1 = dynamic_cast<const FunctionType *>(&type1))
  {
    auto * fnType2 = assertedCast<const FunctionType>(&type2);
    ASSERT_TRUE(
        fnType1->NumArguments() == fnType2->NumArguments()
        && fnType1->NumResults() == fnType2->NumResults())
        << "CompareTypes: FunctionType argument/result count mismatch: expected "
        << type1.debug_string() << " but got " << type2.debug_string();
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
  ASSERT_TRUE(type1 == type2) << "CompareTypes: Type mismatch: expected " << type1.debug_string()
                              << " but got " << type2.debug_string();
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
        << "CompareOperations: Alloca mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
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
        << "CompareOperations: Free mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
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
        << "CompareOperations: MemCpyNonVolatile mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
    return;
  }

  if (auto * vmemcpy1 = dynamic_cast<const jlm::llvm::MemCpyVolatileOperation *>(&op1))
  {
    auto * vmemcpy2 = assertedCast<const jlm::llvm::MemCpyVolatileOperation>(&op2);
    CompareTypes(vmemcpy1->LengthType(), vmemcpy2->LengthType());
    ASSERT_TRUE(vmemcpy1->NumMemoryStates() == vmemcpy2->NumMemoryStates())
        << "CompareOperations: MemCpyVolatile mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
    return;
  }

  if (auto * lambda1 = dynamic_cast<const jlm::llvm::LlvmLambdaOperation *>(&op1))
  {
    auto * lambda2 = assertedCast<const jlm::llvm::LlvmLambdaOperation>(&op2);

    ASSERT_TRUE(lambda1->name() == lambda2->name())
        << "CompareOperations: Lambda name mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
    ASSERT_TRUE(lambda1->linkage() == lambda2->linkage())
        << "CompareOperations: Lambda linkage mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
    ASSERT_TRUE(lambda1->callingConvention() == lambda2->callingConvention())
        << "CompareOperations: Lambda calling convention mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();

    auto type1 = lambda1->type();
    auto type2 = lambda2->type();

    // Compare argument types
    ASSERT_TRUE(type1.NumArguments() == type2.NumArguments())
        << "CompareOperations: Lambda arg count mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
    for (size_t i = 0; i < type1.NumArguments(); ++i)
    {
      CompareTypes(type1.ArgumentType(i), type2.ArgumentType(i));
    }

    // Compare result types
    ASSERT_TRUE(type1.NumResults() == type2.NumResults())
        << "CompareOperations: Lambda result count mismatch: " << op1.debug_string() << " vs "
        << op2.debug_string();
    for (size_t i = 0; i < type1.NumResults(); ++i)
    {
      CompareTypes(type1.ResultType(i), type2.ResultType(i));
    }
  }

  // If same type, use the regular operator==
  if (typeid(op1) == typeid(op2))
  {
    ASSERT_TRUE(op1 == op2) << "CompareOperations: Same-type operation inequality: "
                            << op1.debug_string() << " vs " << op2.debug_string();
    return;
  }

  FAIL() << "CompareOperations: Unknown operation comparison: " << op1.debug_string() << " vs "
         << op2.debug_string();
}

/**
 * \brief Compares two ThetaNode loop variables for structural equality.
 *
 * This verifies the LoopVar struct integrity including input, pre, post, and output
 * fields. It also checks that redirect chains (post->divert_to()) are structurally identical.
 */
static void
CompareThetaLoopVars(const ThetaNode::LoopVar & lv1, const ThetaNode::LoopVar & lv2)
{
  // Verify input types match
  CompareTypes(*lv1.input->Type(), *lv2.input->Type());

  // Verify pre (loop variable value before iteration) types match
  ASSERT_NE(lv1.pre, nullptr) << "CompareThetaLoopVars: Theta LoopVar.pre is null in graph 1";
  ASSERT_NE(lv2.pre, nullptr) << "CompareThetaLoopVars: Theta LoopVar.pre is null in graph 2";
  if (lv1.pre && lv2.pre)
  {
    CompareTypes(*lv1.pre->Type(), *lv2.pre->Type());
  }

  // Verify post (loop variable value after iteration) types match
  ASSERT_NE(lv1.post, nullptr) << "CompareThetaLoopVars: Theta LoopVar.post is null in graph 1";
  ASSERT_NE(lv2.post, nullptr) << "CompareThetaLoopVars: Theta LoopVar.post is null in graph 2";
  if (lv1.post && lv2.post)
  {
    CompareTypes(*lv1.post->Type(), *lv2.post->Type());

    // Verify redirect chain integrity: post->origin() should have same structure
    // When post is redirected via divert_to, we need to compare the origin nodes
    auto * origin1 = TryGetOwnerNode<Node>(*lv1.post->origin());
    auto * origin2 = TryGetOwnerNode<Node>(*lv2.post->origin());

    if (origin1 && origin2)
    {
      CompareNodes(*origin1, *origin2);
    }
    else if (origin1 || origin2)
    {
      // One is redirected but the other isn't - structural mismatch
      FAIL() << "CompareThetaLoopVars: Theta LoopVar post redirect mismatch";
    }
  }

  // Verify output (final value at loop exit) types match
  ASSERT_NE(lv1.output, nullptr) << "CompareThetaLoopVars: Theta LoopVar.output is null in graph 1";
  ASSERT_NE(lv2.output, nullptr) << "CompareThetaLoopVars: Theta LoopVar.output is null in graph 2";
  CompareTypes(*lv1.output->Type(), *lv2.output->Type());
}

/**
 * \brief Compares two GammaNode exit variables for structural equality.
 *
 * This verifies that ExitVar::branchResult vectors have matching sizes and types,
 * and that the output linkage is structurally identical.
 */
static void
CompareGammaExitVars(const GammaNode::ExitVar & ev1, const GammaNode::ExitVar & ev2)
{
  ASSERT_EQ(ev1.branchResult.size(), ev2.branchResult.size())
      << "CompareGammaExitVars: Gamma ExitVar branchResult count mismatch";

  for (size_t i = 0; i < ev1.branchResult.size(); ++i)
  {
    CompareTypes(*ev1.branchResult[i]->Type(), *ev2.branchResult[i]->Type());
  }

  // Verify output linkage integrity
  ASSERT_NE(ev1.output, nullptr) << "CompareGammaExitVars: Gamma ExitVar.output is null in graph 1";
  ASSERT_NE(ev2.output, nullptr) << "CompareGammaExitVars: Gamma ExitVar.output is null in graph 2";
  CompareTypes(*ev1.output->Type(), *ev2.output->Type());
}

/**
 * \brief Compares two PhiNode fixpoint variables for structural equality.
 *
 * This verifies the FixVar struct integrity including recref (recursive reference),
 * result (definition input), and output (external reference) fields. These fields
 * are crucial for defining mutually recursive functions in the RVSDG.
 */
static void
ComparePhiFixVars(const PhiNode::FixVar & fv1, const PhiNode::FixVar & fv2)
{
  // Verify recref (recursive reference to self/other fixpoint) types match
  ASSERT_NE(fv1.recref, nullptr) << "ComparePhiFixVars: Phi FixVar.recref is null in graph 1";
  ASSERT_NE(fv2.recref, nullptr) << "ComparePhiFixVars: Phi FixVar.recref is null in graph 2";
  // Recreftype comparison: the recref is a region argument of the phi subregion.
  // It doesn't have an origin() because it IS the value source for recursive calls.
  CompareTypes(*fv1.recref->Type(), *fv2.recref->Type());

  // Verify result (definition from phi region) types match
  ASSERT_NE(fv1.result, nullptr) << "ComparePhiFixVars: Phi FixVar.result is null in graph 1";
  ASSERT_NE(fv2.result, nullptr) << "ComparePhiFixVars: Phi FixVar.result is null in graph 2";
  CompareTypes(*fv1.result->Type(), *fv2.result->Type());

  // The result is an input to the phi region; follow its origin to compare definition nodes
  auto * origin1 = TryGetOwnerNode<Node>(*fv1.result->origin());
  auto * origin2 = TryGetOwnerNode<Node>(*fv2.result->origin());

  if (origin1 && origin2)
  {
    CompareNodes(*origin1, *origin2);
  }
  else if (origin1 || origin2)
  {
    // One has a redirect but the other doesn't - structural mismatch
    FAIL() << "ComparePhiFixVars: Phi FixVar result redirect mismatch";
  }

  // Verify output (external reference to fixpoint value) types match
  ASSERT_NE(fv1.output, nullptr) << "ComparePhiFixVars: Phi FixVar.output is null in graph 1";
  ASSERT_NE(fv2.output, nullptr) << "ComparePhiFixVars: Phi FixVar.output is null in graph 2";
  CompareTypes(*fv1.output->Type(), *fv2.output->Type());
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
        << "CompareNodes: StructuralNode number of subregions mismatch";

    // Compare each region recursively
    for (size_t r = 0; r < snode1->nsubregions(); ++r)
    {
      CompareRegions(*snode1->subregion(r), *snode2->subregion(r));
    }

    // Compare inputs types
    ASSERT_EQ(snode1->ninputs(), snode2->ninputs())
        << "CompareNodes: Structural node with number of inputs mismatch";
    for (size_t i = 0; i < snode1->ninputs(); ++i)
    {
      CompareTypes(*snode1->input(i)->Type(), *snode2->input(i)->Type());
    }

    // Compare output types
    ASSERT_EQ(snode1->noutputs(), snode2->noutputs())
        << "CompareNodes: Structural node with number of outputs mismatch";
    for (size_t i = 0; i < snode1->noutputs(); ++i)
    {
      CompareTypes(*snode1->output(i)->Type(), *snode2->output(i)->Type());
    }

    // Theta-specific: compare loop variable struct fields
    if (auto * theta1 = dynamic_cast<const ThetaNode *>(&node1))
    {
      auto * theta2 = assertedCast<const ThetaNode>(&node2);

      auto lvList1 = theta1->GetLoopVars();
      auto lvList2 = theta2->GetLoopVars();

      ASSERT_EQ(
          std::distance(lvList1.begin(), lvList1.end()),
          std::distance(lvList2.begin(), lvList2.end()))
          << "CompareNodes: Theta node with variable count mismatch";

      auto it1 = lvList1.begin(), it2 = lvList2.begin();
      while (it1 != lvList1.end() && it2 != lvList2.end())
      {
        CompareThetaLoopVars(*it1, *it2);
        ++it1;
        ++it2;
      }
    }

    // Gamma-specific: compare exit variable struct fields
    if (auto * gamma1 = dynamic_cast<const GammaNode *>(&node1))
    {
      auto * gamma2 = assertedCast<const GammaNode>(&node2);

      auto evList1 = gamma1->GetExitVars();
      auto evList2 = gamma2->GetExitVars();

      ASSERT_EQ(
          std::distance(evList1.begin(), evList1.end()),
          std::distance(evList2.begin(), evList2.end()))
          << "CompareNodes: Gamma node with exit variable count mismatch";

      auto it1 = evList1.begin(), it2 = evList2.begin();
      while (it1 != evList1.end() && it2 != evList2.end())
      {
        CompareGammaExitVars(*it1, *it2);
        ++it1;
        ++it2;
      }
    }

    // PhiNode-specific: compare fixpoint variable struct fields
    if (auto * phi1 = dynamic_cast<const PhiNode *>(&node1))
    {
      auto * phi2 = assertedCast<const PhiNode>(&node2);

      auto fvList1 = phi1->GetFixVars();
      auto fvList2 = phi2->GetFixVars();

      ASSERT_EQ(
          std::distance(fvList1.begin(), fvList1.end()),
          std::distance(fvList2.begin(), fvList2.end()))
          << "CompareNodes: Phi node with fixpoint variable count mismatch";

      auto it1 = fvList1.begin(), it2 = fvList2.begin();
      while (it1 != fvList1.end() && it2 != fvList2.end())
      {
        ComparePhiFixVars(*it1, *it2);
        ++it1;
        ++it2;
      }
    }
    return;
  }

  // Check if simple node
  if (auto * simp1 = dynamic_cast<const SimpleNode *>(&node1))
  {
    auto * simp2 = assertedCast<const SimpleNode>(&node2);

    CompareOperations(simp1->GetOperation(), simp2->GetOperation());

    // Compare inputs
    ASSERT_EQ(simp1->ninputs(), simp2->ninputs())
        << "CompareNodes: Simple node with number of inputs mismatch";
    for (size_t i = 0; i < simp1->ninputs(); ++i)
    {
      CompareTypes(*simp1->input(i)->Type(), *simp2->input(i)->Type());
    }

    // Compare outputs
    ASSERT_EQ(simp1->noutputs(), simp2->noutputs())
        << "CompareNodes: Simple node woth number of outputs mismatch";
    for (size_t i = 0; i < simp1->noutputs(); ++i)
    {
      CompareTypes(*simp1->output(i)->Type(), *simp2->output(i)->Type());
    }
    return;
  }
  ADD_FAILURE() << "CompareNodes: Could not identify node type";
}

/**
 * \brief Collect context variable origin node pairs from a structural node.
 *
 * For LambdaNode, DeltaNode, and PhiNode, this extracts the owner nodes of all
 * context variable inputs. This enables BFS to traverse into subregions via
 * their context variable dependencies that are not reachable through normal
 * input→origin edges.
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
  // PhiNode case - collects CV inputs (not recref, handled separately)
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

  return origins;
}

/**
 * \brief Check for context variables and traverse their origin nodes into the BFS queue.
 */
static void
CheckForContextVariables(
    const Node & node1,
    const Node & node2,
    std::unordered_set<const Node *> & visited,
    std::queue<std::pair<const Node *, const Node *>> & nodeQueue)
{
  // Context variable origins (LambdaNode, DeltaNode, PhiNode)
  auto cvOrigins = CollectContextVarOrigins(node1, node2);

  for (auto [origin1, origin2] : cvOrigins)
  {
    if (!visited.count(origin1))
    {
      visited.insert(origin1);
      CompareNodes(*origin1, *origin2);
      nodeQueue.push({ origin1, origin2 });

      // Recursively expand from this origin's context variables too.
      // This handles cases like: Lambda A has context var from Lambda B,
      // which itself has context vars we need to visit.
      CheckForContextVariables(*origin1, *origin2, visited, nodeQueue);
    }
  }
}

/**
 * \brief Compares two RVSDG regions for equality by traversing through results
 * and verifying the same graph structure exists in both regions.
 */
void
CompareRegions(const Region & region1, const Region & region2)
{
  ASSERT_EQ(region1.narguments(), region2.narguments())
      << "CompareRegions: Region number of arguments mismatch";
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

  ASSERT_EQ(region1.numNodes(), region2.numNodes()) << "CompareRegions: Number of nodes mismatch.";

  std::unordered_set<const Node *> visited;
  std::queue<std::pair<const Node *, const Node *>> nodeQueue;

  // Seed from each region result and find the node that produces it
  for (size_t i = 0; i < region1.nresults(); ++i)
  {
    auto * origin1 = region1.result(i)->origin();
    auto * origin2 = region2.result(i)->origin();

    if (!origin1 || !origin2)
    {
      ADD_FAILURE() << "CompareRegions: Result origin is null at index " << i;
      return;
    }

    CompareTypes(*origin1->Type(), *origin2->Type());

    if (auto * node1 = TryGetOwnerNode<Node>(*origin1))
    {
      auto * node2 = TryGetOwnerNode<Node>(*origin2);
      ASSERT_NE(node2, nullptr);
      // Add nodes to queue for BFS traversing
      nodeQueue.push({ node1, node2 });
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
        ADD_FAILURE() << "CompareRegions: Input origin mismatch for node inputs at index " << j;
        return;
      }

      if (auto * next1 = TryGetOwnerNode<Node>(*origin1))
      {
        auto * next2 = TryGetOwnerNode<Node>(*origin2);
        ASSERT_NE(next2, nullptr);

        if (!visited.count(next1))
        {
          visited.insert(next1);
          nodeQueue.push({ next1, next2 });
          CompareNodes(*next1, *next2);
          CheckForContextVariables(*next1, *next2, visited, nodeQueue);
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
}

/**
 * \brief Compares two LlvmRvsdgModule instances for equality.
 */
void
CompareModules(const LlvmRvsdgModule & module1, const LlvmRvsdgModule & module2)
{
  CompareRegions(module1.Rvsdg().GetRootRegion(), module2.Rvsdg().GetRootRegion());

  // Compare root region exports
  ASSERT_EQ(module1.Rvsdg().GetRootRegion().nresults(), module2.Rvsdg().GetRootRegion().nresults())
      << "CompareModules: Root region export count mismatch";
  for (size_t i = 0; i < module1.Rvsdg().GetRootRegion().nresults(); ++i)
  {
    auto * exp1 = assertedCast<const GraphExport>(module1.Rvsdg().GetRootRegion().result(i));
    auto * exp2 = assertedCast<const GraphExport>(module2.Rvsdg().GetRootRegion().result(i));

    ASSERT_STREQ(exp1->Name().c_str(), exp2->Name().c_str())
        << "CompareModules: Export name mismatch at index " << i;
  }
}

/**
 * \brief Tests that an RVSDG graph roundtrips through MLIR.
 */
void
TestRvsdgRoundtrip(const LlvmRvsdgModule & originalModule, const char * testName)
{
  using namespace jlm::mlir;
  (void)testName;

  JlmToMlirConverter mlirgen;
  auto omega = mlirgen.ConvertModule(originalModule);

  std::unique_ptr<mlir::Block> rootBlock = std::make_unique<mlir::Block>();
  rootBlock->push_back(omega);

  auto roundTripModule = MlirToJlmConverter::CreateAndConvert(rootBlock);

  CompareModules(originalModule, *roundTripModule);
}

} // namespace

// Tests from RVSDG graphs defined in jlm/llvm/TestRvsdgs.cpp

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

TEST(RvsdgRoundtripTests, TestExternalCallTest1)
{
  ::jlm::llvm::ExternalCallTest1 test;
  TestRvsdgRoundtrip(test.module(), "ExternalCallTest1");
}

TEST(RvsdgRoundtripTests, TestDeltaTest1)
{
  ::jlm::llvm::DeltaTest1 test;
  TestRvsdgRoundtrip(test.module(), "DeltaTest1");
}

TEST(RvsdgRoundtripTests, TestExternalMemory)
{
  ::jlm::llvm::ExternalMemoryTest test;
  TestRvsdgRoundtrip(test.module(), "ExternalMemory");
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

TEST(RvsdgRoundtripTests, TestFreeNull)
{
  ::jlm::llvm::FreeNullTest test;
  TestRvsdgRoundtrip(test.module(), "FreeNull");
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
