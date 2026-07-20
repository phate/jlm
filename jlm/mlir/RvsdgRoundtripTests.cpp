/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <queue>
#include <unordered_set>

#include <jlm/rvsdg/lambda.hpp>

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

// Now we have the full type definitions - use them
using namespace jlm::llvm;
using namespace jlm::rvsdg;
using namespace jlm::util;

/**
 * \brief Check if any node in the region is a Memcpy operation
 */
bool
ContainsMemcpy(const Region & region)
{
  for (const auto & node : region.Nodes())
  {
    if (auto * snode = dynamic_cast<const SimpleNode *>(&node))
    {
      if (dynamic_cast<const jlm::llvm::MemCpyNonVolatileOperation *>(&snode->GetOperation())
          || dynamic_cast<const jlm::llvm::MemCpyVolatileOperation *>(&snode->GetOperation()))
      {
        return true;
      }
    }
  }
  return false;
}

/**
 * \brief Internal comparison function - returns true if types are structurally equivalent.
 */
bool
DoCompareTypes(const Type & type1, const Type & type2)
{
  // If same type class and equal by operator==, return true immediately
  if (type1 == type2)
    return true;

  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Handle BitType comparison (both should be BitType in JLM's rvsdg)
  auto * bitType1 = dynamic_cast<const BitType *>(&type1);
  auto * bitType2 = dynamic_cast<const BitType *>(&type2);

  if (bitType1 && bitType2)
    return bitType1->nbits() == bitType2->nbits();

  // Handle StructType comparison - compare by element types and properties
  auto * structType1 = dynamic_cast<const StructType *>(&type1);
  auto * structType2 = dynamic_cast<const StructType *>(&type2);

  if (structType1 && structType2)
  {
    // Compare element count using numElements()
    if (structType1->numElements() != structType2->numElements())
      return false;
    // Compare each element type recursively using getElementType(index)
    for (size_t i = 0; i < structType1->numElements(); ++i)
    {
      if (!DoCompareTypes(*structType1->getElementType(i), *structType2->getElementType(i)))
        return false;
    }
    // Compare packed status using IsPacked()
    return structType1->IsPacked() == structType2->IsPacked();
  }

  // Handle ArrayType comparison - compare by element type and size
  auto * arrayType1 = dynamic_cast<const ArrayType *>(&type1);
  auto * arrayType2 = dynamic_cast<const ArrayType *>(&type2);

  if (arrayType1 && arrayType2)
  {
    return DoCompareTypes(arrayType1->element_type(), arrayType2->element_type())
        && arrayType1->nelements() == arrayType2->nelements();
  }

  // Handle FunctionType comparison - compare by argument and result types
  auto * fnType1 = dynamic_cast<const FunctionType *>(&type1);
  auto * fnType2 = dynamic_cast<const FunctionType *>(&type2);

  if (fnType1 && fnType2)
  {
    if (fnType1->NumArguments() != fnType2->NumArguments()
        || fnType1->NumResults() != fnType2->NumResults())
      return false;
    // FunctionType::ArgumentType() and ResultType() return const Type&, not shared_ptr
    for (size_t i = 0; i < fnType1->NumArguments(); ++i)
    {
      if (!DoCompareTypes(fnType1->ArgumentType(i), fnType2->ArgumentType(i)))
        return false;
    }
    for (size_t i = 0; i < fnType1->NumResults(); ++i)
    {
      if (!DoCompareTypes(fnType1->ResultType(i), fnType2->ResultType(i)))
        return false;
    }
    return true;
  }

  // Fallback to regular equality for any remaining types
  return type1 == type2;
}

// Forward declarations - functions are called before their definitions
void
CompareTypes(const Type & type1, const Type & type2);

void
CompareRegions(const Region & region1, const Region & region2);

/**
 * \brief Compares two operations for equality, handling different but equivalent
 * operation types.
 */
bool
CompareOperations(const Operation & op1, const Operation & op2)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  // Handle AllocaOperation comparison first (before typeid check) since
  // AllocaOperation uses pointer identity for its operator== which would fail
  auto * alloca1 = dynamic_cast<const AllocaOperation *>(&op1);
  auto * alloca2 = dynamic_cast<const AllocaOperation *>(&op2);

  if (alloca1 && alloca2)
  {
    bool result = DoCompareTypes(*alloca1->allocatedType(), *alloca2->allocatedType())
               && alloca1->alignment() == alloca2->alignment();
    if (!result)
      std::cout << "Alloca mismatch: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // Handle MallocOperation comparison
  auto * malloc1 = dynamic_cast<const MallocOperation *>(&op1);
  auto * malloc2 = dynamic_cast<const MallocOperation *>(&op2);

  if (malloc1 && malloc2)
  {
    bool result = DoCompareTypes(malloc1->getSizeType(), malloc2->getSizeType());
    if (!result)
      std::cout << "Malloc mismatch: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // Handle FreeOperation comparison
  auto * free1 = dynamic_cast<const FreeOperation *>(&op1);
  auto * free2 = dynamic_cast<const FreeOperation *>(&op2);

  if (free1 && free2)
  {
    bool result = free1->narguments() == free2->narguments();
    if (!result)
      std::cout << "Free mismatch: " << op1.debug_string() << " vs " << op2.debug_string() << "\n";
    return result;
  }

  // Handle ConstantDataArrayOperation and ConstantArrayOperation comparison.
  // Both get converted to/from mlir::jlm::ConstantDataArray, so they should be considered
  // equivalent.
  auto * constDataArr1 = dynamic_cast<const ConstantDataArrayOperation *>(&op1);
  auto * constDataArr2 = dynamic_cast<const ConstantDataArrayOperation *>(&op2);
  auto * constArr1 = dynamic_cast<const ConstantArrayOperation *>(&op1);
  auto * constArr2 = dynamic_cast<const ConstantArrayOperation *>(&op2);

  // Both are ConstantDataArrayOperation - compare by array type
  if (constDataArr1 && constDataArr2)
  {
    bool result = DoCompareTypes(*constDataArr1->result(0), *constDataArr2->result(0));
    if (!result)
      std::cout << "ConstantDataArray mismatch: " << op1.debug_string() << " vs "
                << op2.debug_string() << "\n";
    return result;
  }

  // Both are ConstantArrayOperation - compare by array type
  if (constArr1 && constArr2)
  {
    bool result = DoCompareTypes(*constArr1->result(0), *constArr2->result(0));
    if (!result)
      std::cout << "ConstantArray mismatch: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // One is ConstantDataArrayOperation, the other is ConstantArrayOperation.
  // These are equivalent since both convert to/from mlir::jlm::ConstantDataArray.
  if ((constDataArr1 != nullptr || constDataArr2 != nullptr)
      && (constArr1 != nullptr || constArr2 != nullptr))
  {
    std::shared_ptr<const Type> type1, type2;
    if (constDataArr1)
      type1 = constDataArr1->result(0);
    else
      type1 = constArr1->result(0);
    if (constDataArr2)
      type2 = constDataArr2->result(0);
    else
      type2 = constArr2->result(0);
    bool result = DoCompareTypes(*type1, *type2);
    if (!result)
      std::cout << "ConstantArray/ConstantDataArray mismatch: " << op1.debug_string() << " vs "
                << op2.debug_string() << "\n";
    return result;
  }

  // Handle ConstantStructOperation - compare by struct type only (elements are pointers to nodes
  // in the region, not stored on the operation)
  auto * constStruct1 = dynamic_cast<const ConstantStructOperation *>(&op1);
  auto * constStruct2 = dynamic_cast<const ConstantStructOperation *>(&op2);

  if (constStruct1 && constStruct2)
  {
    bool result = DoCompareTypes(*constStruct1->result(0), *constStruct2->result(0));
    // Also verify element counts match
    // Note: elements are stored in the region, not on the operation itself
    if (!result)
      std::cout << "ConstantStruct mismatch: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // Handle ConstantAggregateZeroOperation - compare by type
  auto * constAggZero1 = dynamic_cast<const ConstantAggregateZeroOperation *>(&op1);
  auto * constAggZero2 = dynamic_cast<const ConstantAggregateZeroOperation *>(&op2);

  if (constAggZero1 && constAggZero2)
  {
    bool result = DoCompareTypes(*constAggZero1->result(0), *constAggZero2->result(0));
    if (!result)
      std::cout << "ConstantAggregateZero mismatch: " << op1.debug_string() << " vs "
                << op2.debug_string() << "\n";
    return result;
  }

  // Handle CallOperation - compare by function type using value comparison
  auto * call1 = dynamic_cast<const CallOperation *>(&op1);
  auto * call2 = dynamic_cast<const CallOperation *>(&op2);

  if (call1 && call2)
  {
    bool typesMatch = DoCompareTypes(*call1->GetFunctionType(), *call2->GetFunctionType());
    if (!typesMatch)
      std::cout << "Call type mismatch: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return typesMatch;
  }

  // Handle GetElementPtrOperation - compare the pointee types before typeid check
  auto * gep1 = dynamic_cast<const GetElementPtrOperation *>(&op1);
  auto * gep2 = dynamic_cast<const GetElementPtrOperation *>(&op2);

  if (gep1 && gep2)
  {
    // Compare pointee types
    bool typesMatch = DoCompareTypes(*gep1->getPointeeType(), *gep2->getPointeeType());

    if (!typesMatch)
      std::cout << "GetElementPtr type mismatch: " << op1.debug_string() << " vs "
                << op2.debug_string() << "\n";

    return typesMatch;
  }

  // Handle MemCpy operations first (before typeid check) since they might have different
  // memory state counts after conversion and need special handling
  auto * memcpy1 = dynamic_cast<const jlm::llvm::MemCpyNonVolatileOperation *>(&op1);
  auto * memcpy2 = dynamic_cast<const jlm::llvm::MemCpyNonVolatileOperation *>(&op2);

  if (memcpy1 && memcpy2)
  {
    bool result = DoCompareTypes(memcpy1->LengthType(), memcpy2->LengthType())
               && memcpy1->NumMemoryStates() == memcpy2->NumMemoryStates();
    if (!result)
      std::cout << "MemCpyNonVolatile mismatch: " << op1.debug_string() << " vs "
                << op2.debug_string() << "\n";
    return result;
  }

  auto * vmemcpy1 = dynamic_cast<const jlm::llvm::MemCpyVolatileOperation *>(&op1);
  auto * vmemcpy2 = dynamic_cast<const jlm::llvm::MemCpyVolatileOperation *>(&op2);

  if (vmemcpy1 && vmemcpy2)
  {
    bool result = DoCompareTypes(vmemcpy1->LengthType(), vmemcpy2->LengthType())
               && vmemcpy1->NumMemoryStates() == vmemcpy2->NumMemoryStates();
    if (!result)
      std::cout << "MemCpyVolatile mismatch: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // If same type, use the regular operator==
  if (typeid(op1) == typeid(op2))
  {
    auto * lambda1 = dynamic_cast<const jlm::llvm::LlvmLambdaOperation *>(&op1);
    auto * lambda2 = dynamic_cast<const jlm::llvm::LlvmLambdaOperation *>(&op2);

    if (lambda1 && lambda2)
    {
      JLM_ASSERT(lambda1->name() == lambda2->name());
      JLM_ASSERT(lambda1->linkage() == lambda2->linkage());
      JLM_ASSERT(lambda1->callingConvention() == lambda2->callingConvention());

      auto type1 = lambda1->type();
      auto type2 = lambda2->type();

      JLM_ASSERT(type1.NumArguments() == type2.NumArguments());
      for (size_t i = 0; i < type1.NumArguments(); ++i)
      {
        JLM_ASSERT(DoCompareTypes(type1.ArgumentType(i), type2.ArgumentType(i)));
      }

      for (size_t i = 0; i < type1.NumResults(); ++i)
      {
        JLM_ASSERT(DoCompareTypes(type1.ResultType(i), type2.ResultType(i)));
      }
    }

    return op1 == op2;
  }

  // Lambda node region argument comparison happens at CompareRegions level

  // If types are different, check if they're equivalent but not identical
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  // Handle BitConstantOperation vs IntegerConstantOperation comparison
  auto * bitConst1 = dynamic_cast<const BitConstantOperation *>(&op1);
  auto * intConst2 = dynamic_cast<const IntegerConstantOperation *>(&op2);

  if (bitConst1 && intConst2)
  {
    bool result = bitConst1->value() == intConst2->Representation();
    if (!result)
      std::cout << "BitIntConst fail: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // Reverse check
  auto * intConst1 = dynamic_cast<const IntegerConstantOperation *>(&op1);
  auto * bitConst2 = dynamic_cast<const BitConstantOperation *>(&op2);

  if (intConst1 && bitConst2)
  {
    bool result = intConst1->Representation() == bitConst2->value();
    if (!result)
      std::cout << "IntBitConst fail: " << op1.debug_string() << " vs " << op2.debug_string()
                << "\n";
    return result;
  }

  // Cross-type comparison for binary ops
  auto * bitBinOp1 = dynamic_cast<const BitBinaryOperation *>(&op1);
  auto * intBinOp2 = dynamic_cast<const IntegerBinaryOperation *>(&op2);

  if (bitBinOp1 && intBinOp2)
  {
    bool result = DoCompareTypes(*bitBinOp1->result(0), *intBinOp2->result(0));
    if (!result)
      std::cout << "BitIntBin fail: " << op1.debug_string() << " vs " << op2.debug_string() << "\n";
    return result;
  }

  // Reverse check for binary ops
  auto * intBinOp1 = dynamic_cast<const IntegerBinaryOperation *>(&op1);
  auto * bitBinOpRev = dynamic_cast<const BitBinaryOperation *>(&op2);

  if (intBinOp1 && bitBinOpRev)
  {
    bool result = DoCompareTypes(*intBinOp1->result(0), *bitBinOpRev->result(0));
    if (!result)
      std::cout << "IntBitBin fail: " << op1.debug_string() << " vs " << op2.debug_string() << "\n";
    return result;
  }

  // Note: GetElementPtr handler is placed at the beginning of CompareOperations to handle
  // same-type comparisons before falling through to typeid check

  // Handle IntegerUltOperation - compare with BitComparisonOperation
  auto * intUlt1 = dynamic_cast<const IntegerUltOperation *>(&op1);
  auto * bitCompOp2 = dynamic_cast<const jlm::rvsdg::BitCompareOperation *>(&op2);

  if (intUlt1 && bitCompOp2)
    return DoCompareTypes(*intUlt1->result(0), *bitCompOp2->result(0));

  // Reverse check for integer comparison
  auto * intUlt2 = dynamic_cast<const IntegerUltOperation *>(&op2);
  auto * bitCompOp1 = dynamic_cast<const jlm::rvsdg::BitCompareOperation *>(&op1);

  if (intUlt2 && bitCompOp1)
    return DoCompareTypes(*bitCompOp1->result(0), *intUlt2->result(0));

  // Handle IntegerEqOperation - compare with BitCompareOperation
  auto * intEq1 = dynamic_cast<const IntegerEqOperation *>(&op1);
  auto * bitCompEq2 = dynamic_cast<const jlm::rvsdg::BitCompareOperation *>(&op2);

  if (intEq1 && bitCompEq2)
    return DoCompareTypes(*intEq1->result(0), *bitCompEq2->result(0));

  // Reverse check for equality comparison
  auto * intEq2 = dynamic_cast<const IntegerEqOperation *>(&op2);
  auto * bitCompEq1 = dynamic_cast<const jlm::rvsdg::BitCompareOperation *>(&op1);

  if (intEq2 && bitCompEq1)
    return DoCompareTypes(*bitCompEq1->result(0), *intEq2->result(0));

  // Handle IntegerAddOperation - compare with BitBinaryOperation
  auto * intAdd1 = dynamic_cast<const IntegerAddOperation *>(&op1);
  auto * bitBinOp2 = dynamic_cast<const jlm::rvsdg::BitBinaryOperation *>(&op2);

  if (intAdd1 && bitBinOp2)
    return DoCompareTypes(*intAdd1->result(0), *bitBinOp2->result(0));

  // Handle MemoryStateMergeOperation - compare with Store operation
  auto * memMerge1 = dynamic_cast<const jlm::llvm::MemoryStateMergeOperation *>(&op1);
  auto * store2 = dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&op2);

  if (memMerge1 && store2)
  {
    // Memory state merge and store both produce memory state
    return DoCompareTypes(*memMerge1->result(0), *store2->result(0));
  }

  auto * store1 = dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&op1);
  auto * memMerge2 = dynamic_cast<const jlm::llvm::MemoryStateMergeOperation *>(&op2);

  if (store1 && memMerge2)
  {
    return DoCompareTypes(*store1->result(0), *memMerge2->result(0));
  }

  // Print what we're trying to compare - especially for Memcpy operations
  std::cout << "Unknown comparison: " << typeid(op1).name() << " vs " << typeid(op2).name() << ": "
            << op1.debug_string() << " vs " << op2.debug_string() << "\n";

  // If same type class, they should be equal via operator== (already checked above)
  if (typeid(op1) == typeid(op2))
  {
    bool result = op1 == op2;
    if (!result)
      std::cout << "Same-type operation inequality: " << op1.debug_string() << " vs "
                << op2.debug_string() << "\n";
    return result;
  }

  return false;
}

/**
 * \brief Compares two RVSDG types for equality.
 */
// Debug for function type comparison failures
void
CompareTypes(const Type & type1, const Type & type2)
{
  if (!DoCompareTypes(type1, type2))
  {
    std::cerr << "DEBUG CompareTypes FAILED: expected=" << type1.debug_string()
              << ", got=" << type2.debug_string() << std::endl;
    // Print more info for function types
    auto * fnType1 = dynamic_cast<const FunctionType *>(&type1);
    auto * fnType2 = dynamic_cast<const FunctionType *>(&type2);
    if (fnType1 && fnType2)
    {
      std::cerr << "DEBUG: FunctionType1 - numArgs=" << fnType1->NumArguments()
                << ", numResults=" << fnType1->NumResults() << std::endl;
      for (size_t i = 0; i < fnType1->NumArguments(); ++i)
        std::cerr << "DEBUG:   Arg " << i << ": " << fnType1->ArgumentType(i).debug_string()
                  << std::endl;
    }
    if (fnType2 && fnType2 != fnType1)
    {
      std::cerr << "DEBUG: FunctionType2 - numArgs=" << fnType2->NumArguments()
                << ", numResults=" << fnType2->NumResults() << std::endl;
      for (size_t i = 0; i < fnType2->NumArguments(); ++i)
        std::cerr << "DEBUG:   Arg " << i << ": " << fnType2->ArgumentType(i).debug_string()
                  << std::endl;
    }
  }

  ASSERT_TRUE(DoCompareTypes(type1, type2))
      << "Type mismatch: expected " << type1.debug_string() << " but got " << type2.debug_string();
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

    ASSERT_TRUE(CompareOperations(snode1->GetOperation(), snode2->GetOperation()))
        << "StructuralNode operation mismatch: node1=" << &node1 << ", node2=" << &node2;
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

    const auto & op1 = simp1->GetOperation();
    const auto & op2 = simp2->GetOperation();

    bool result = CompareOperations(op1, op2);
    if (!result)
    {
      std::cout << "DEBUG: CompareNodes operation mismatch:\n";
      std::cout << "  Original: " << typeid(op1).name() << " - " << op1.debug_string() << "\n";
      std::cout << "  Converted: " << typeid(op2).name() << " - " << op2.debug_string() << "\n";
    }

    ASSERT_TRUE(result) << "SimpleNode operation mismatch";

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
 * \brief Compares two RVSDG regions for equality by traversing through results
 * and verifying the same graph structure exists in both regions.
 */
// Debug: Add counter for lambda comparisons
void
CompareRegions(const Region & region1, const Region & region2)
{
  // Check number of arguments and results
  std::cerr << "DEBUG CompareRegions: region1.nargs=" << region1.narguments()
            << ", region2.nargs=" << region2.narguments() << ", nresults=" << region1.nresults()
            << std::endl;
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

  // Check node count (skip for regions containing Memcpy operations since they create extra
  // ConstantIntOp nodes)
  bool hasMemcpy1 = ContainsMemcpy(region1);
  bool hasMemcpy2 = ContainsMemcpy(region2);

  size_t count1 = region1.numNodes();
  size_t count2 = region2.numNodes();

  if (!hasMemcpy1 && !hasMemcpy2)
  {
    ASSERT_EQ(count1, count2) << "Node count mismatch: " << count1 << " vs " << count2;
  }

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

  // Phase 2: Handle unvisited nodes (context variables in lambdas)
  size_t iteration = 0;
  while (visited1.size() < count1 && iteration++ < 10)
  {
    bool foundContextVar = false;

    // Find an unvisited lambda with context variables
    for (const auto & node : region1.Nodes())
    {
      if (visited1.count(&node))
        continue;

      if (auto * lambda = dynamic_cast<const LambdaNode *>(&node))
      {
        // Get the corresponding node from region2 by matching operation type/name
        const LambdaNode * lambda2 = nullptr;
        for (const auto & n : region2.Nodes())
        {
          if (visited2.count(&n))
            continue;

          if (auto * l2 = dynamic_cast<const LambdaNode *>(&n))
          {
            if (lambda->GetOperation().debug_string() == l2->GetOperation().debug_string())
            {
              lambda2 = l2;
              break;
            }
          }
        }

        if (!lambda2)
          continue;

        // Get context variables and compare their origins
        auto cvList1 = lambda->GetContextVars();
        auto cvList2 = lambda2->GetContextVars();

        size_t cvCount1 = std::distance(cvList1.begin(), cvList1.end());
        size_t cvCount2 = std::distance(cvList2.begin(), cvList2.end());

        ASSERT_EQ(cvCount1, cvCount2)
            << "Lambda context variable count mismatch: " << cvCount1 << " vs " << cvCount2;

        auto it1 = cvList1.begin();
        auto it2 = cvList2.begin();

        while (it1 != cvList1.end() && it2 != cvList2.end())
        {
          // Compare the origin of this context variable
          if (auto * origin1 = TryGetOwnerNode<Node>(*it1->input->origin()))
          {
            if (auto * origin2 = TryGetOwnerNode<Node>(*it2->input->origin()))
            {
              ASSERT_NE(origin2, nullptr);

              CompareNodes(*origin1, *origin2);

              visited1.insert(origin1);
              visited2.insert(origin2);
              foundContextVar = true;
            }
          }
          ++it1;
          ++it2;
        }

        // Also visit the lambda node itself
        visited1.insert(lambda);
        visited2.insert(lambda2);
        break;
      }
    }

    if (!foundContextVar)
      break; // No more context variables to process
  }

  // Phase 3: Visit any remaining unvisited nodes (dangling operations in subregions)
  // These are nodes not reachable from results via normal graph edges
  // For lambdas, also compare their context variable origins
  while (visited1.size() < count1)
  {
    bool foundRemaining = false;

    for (const auto & node : region1.Nodes())
    {
      if (visited1.count(&node))
        continue;

      // Try to find corresponding unvisited node in region2 by operation type/name
      const Node * matchingNode = nullptr;
      for (const auto & n : region2.Nodes())
      {
        if (visited2.count(&n))
          continue;

        // Use CompareOperations which handles type equivalence (e.g., BITS32 vs I32)
        if (CompareOperations(node.GetOperation(), n.GetOperation()))
        {
          matchingNode = &n;
          break;
        }
      }

      if (!matchingNode)
        continue;

      CompareNodes(node, *matchingNode);

      visited1.insert(&node);
      visited2.insert(matchingNode);
      foundRemaining = true;
      break; // Restart loop to find more nodes
    }

    if (!foundRemaining)
      break;
  }

  // Verify all nodes were visited
  ASSERT_EQ(visited1.size(), count1) << "Node count mismatch after traversal";
  ASSERT_EQ(visited2.size(), count2) << "Node count mismatch for region2 after traversal";
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
