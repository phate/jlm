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

bool
CompareTypes(const Type & type1, const Type & type2);

bool
CompareNodes(const Node & node1, const Node & node2);

bool
CompareRegions(const Region & region1, const Region & region2);

/**
 * \brief Compares two RVSDG types for equality.
 *
 * \param type1 The first type to compare.
 * \param type2 The second type to compare.
 * \return True if the types are equal, false otherwise.
 */
bool
CompareTypes(const Type & type1, const Type & type2)
{
  if (is<BitType>(type1))
  {
    auto * bit1 = dynamic_cast<const BitType *>(&type1);
    auto * bit2 = dynamic_cast<const BitType *>(&type2);
    if (!bit1 || !bit2)
    {
      std::cerr << "Type mismatch: expected BitType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    if (bit1->nbits() != bit2->nbits())
    {
      std::cerr << "BitType mismatch: nbits=" << bit1->nbits() << " vs " << bit2->nbits()
                << " for types " << type1.debug_string() << " and " << type2.debug_string()
                << std::endl;
      return false;
    }
    return true;
  }

  if (is<PointerType>(type1))
  {
    if (!is<PointerType>(type2))
    {
      std::cerr << "Type mismatch: expected PointerType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    return true;
  }

  if (is<MemoryStateType>(type1))
  {
    if (!is<MemoryStateType>(type2))
    {
      std::cerr << "Type mismatch: expected MemoryStateType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    return true;
  }

  if (is<IOStateType>(type1))
  {
    if (!is<IOStateType>(type2))
    {
      std::cerr << "Type mismatch: expected IOStateType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    return true;
  }

  if (auto * func1 = dynamic_cast<const FunctionType *>(&type1))
  {
    auto * func2 = dynamic_cast<const FunctionType *>(&type2);
    if (!func2)
    {
      std::cerr << "Type mismatch: expected FunctionType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    if (func1->NumArguments() != func2->NumArguments())
    {
      std::cerr << "FunctionType mismatch: NumArguments=" << func1->NumArguments() << " vs "
                << func2->NumArguments() << " for types " << type1.debug_string() << " and "
                << type2.debug_string() << std::endl;
      return false;
    }
    if (func1->NumResults() != func2->NumResults())
    {
      std::cerr << "FunctionType mismatch: NumResults=" << func1->NumResults() << " vs "
                << func2->NumResults() << " for types " << type1.debug_string() << " and "
                << type2.debug_string() << std::endl;
      return false;
    }
    for (size_t i = 0; i < func1->NumArguments(); ++i)
    {
      if (!CompareTypes(func1->ArgumentType(i), func2->ArgumentType(i)))
      {
        std::cerr << "FunctionType argument type mismatch at index " << i << std::endl;
        return false;
      }
    }
    for (size_t i = 0; i < func1->NumResults(); ++i)
    {
      if (!CompareTypes(func1->ResultType(i), func2->ResultType(i)))
      {
        std::cerr << "FunctionType result type mismatch at index " << i << std::endl;
        return false;
      }
    }
    return true;
  }

  if (auto * array1 = dynamic_cast<const ArrayType *>(&type1))
  {
    auto * array2 = dynamic_cast<const ArrayType *>(&type2);
    if (!array2)
    {
      std::cerr << "Type mismatch: expected ArrayType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    if (array1->nelements() != array2->nelements())
    {
      std::cerr << "ArrayType mismatch: nelements=" << array1->nelements() << " vs "
                << array2->nelements() << " for types " << type1.debug_string() << " and "
                << type2.debug_string() << std::endl;
      return false;
    }
    if (!CompareTypes(array1->element_type(), array2->element_type()))
    {
      std::cerr << "ArrayType element type mismatch" << std::endl;
      return false;
    }
    return true;
  }

  if (auto * struct1 = dynamic_cast<const StructType *>(&type1))
  {
    auto * struct2 = dynamic_cast<const StructType *>(&type2);
    if (!struct2)
    {
      std::cerr << "Type mismatch: expected StructType for both, got " << type1.debug_string()
                << " vs " << type2.debug_string() << std::endl;
      return false;
    }
    if (struct1->numElements() != struct2->numElements())
    {
      std::cerr << "StructType mismatch: numElements=" << struct1->numElements() << " vs "
                << struct2->numElements() << " for types " << type1.debug_string() << " and "
                << type2.debug_string() << std::endl;
      return false;
    }
    for (size_t i = 0; i < struct1->numElements(); ++i)
    {
      if (!CompareTypes(*struct1->getElementType(i), *struct2->getElementType(i)))
      {
        std::cerr << "StructType element type mismatch at index " << i << std::endl;
        return false;
      }
    }
    return true;
  }

  if (type1 == type2)
  {
    // Default: use operator== for comparison
    return true;
  }

  std::cerr << "Type mismatch via operator==: " << type1.debug_string() << " vs "
            << type2.debug_string() << std::endl;
  return false;
}

/**
 * \brief Compares two RVSDG nodes for equality.
 *
 * \param node1 The first node to compare.
 * \param node2 The second node to compare.
 * \return True if the nodes are equal, false otherwise.
 */
bool
CompareNodes(const Node & node1, const Node & node2)
{
  // Check if both are structural nodes (Lambda, Gamma, Theta, Delta)
  auto * snode1 = dynamic_cast<const StructuralNode *>(&node1);
  auto * snode2 = dynamic_cast<const StructuralNode *>(&node2);

  if (snode1 && snode2)
  {
    if (snode1->nsubregions() != snode2->nsubregions())
    {
      std::cerr << "StructuralNode nsubregions mismatch: node1=" << &node1 << ", node2=" << &node2
                << ": " << snode1->nsubregions() << " vs " << snode2->nsubregions() << std::endl;
      return false;
    }
    if (snode1->ninputs() != snode2->ninputs())
    {
      std::cerr << "StructuralNode ninputs mismatch: node1=" << &node1 << ", node2=" << &node2
                << ": " << snode1->ninputs() << " vs " << snode2->ninputs() << std::endl;
      return false;
    }
    if (snode1->noutputs() != snode2->noutputs())
    {
      std::cerr << "StructuralNode noutputs mismatch: node1=" << &node1 << ", node2=" << &node2
                << ": " << snode1->noutputs() << " vs " << snode2->noutputs() << std::endl;
      return false;
    }

    // Compare each region recursively
    for (size_t r = 0; r < snode1->nsubregions(); ++r)
    {
      if (!CompareRegions(*snode1->subregion(r), *snode2->subregion(r)))
      {
        std::cerr << "StructuralNode subregion " << r << " mismatch" << std::endl;
        return false;
      }
    }

    // Compare input types
    for (size_t i = 0; i < snode1->ninputs(); ++i)
    {
      if (!CompareTypes(*snode1->input(i)->Type(), *snode2->input(i)->Type()))
      {
        std::cerr << "StructuralNode input type mismatch at index " << i << std::endl;
        return false;
      }
    }

    // Compare output types
    for (size_t i = 0; i < snode1->noutputs(); ++i)
    {
      if (!CompareTypes(*snode1->output(i)->Type(), *snode2->output(i)->Type()))
      {
        std::cerr << "StructuralNode output type mismatch at index " << i << std::endl;
        return false;
      }
    }

    return true;
  }

  // Check if both are simple nodes
  auto * simp1 = dynamic_cast<const SimpleNode *>(&node1);
  auto * simp2 = dynamic_cast<const SimpleNode *>(&node2);

  if (simp1 && simp2)
  {
    const auto & op1 = simp1->GetOperation();
    const auto & op2 = simp2->GetOperation();

    if (typeid(op1) != typeid(op2))
    {
      std::cerr << "SimpleNode operation type mismatch: " << typeid(op1).name() << " vs "
                << typeid(op2).name() << std::endl;
      return false;
    }

    // Compare input counts and types
    if (simp1->ninputs() != simp2->ninputs())
    {
      std::cerr << "SimpleNode ninputs mismatch: " << simp1->ninputs() << " vs " << simp2->ninputs()
                << std::endl;
      return false;
    }
    for (size_t i = 0; i < simp1->ninputs(); ++i)
    {
      if (!CompareTypes(*simp1->input(i)->Type(), *simp2->input(i)->Type()))
      {
        std::cerr << "SimpleNode input type mismatch at index " << i << std::endl;
        return false;
      }
    }

    // Compare output counts and types
    if (simp1->noutputs() != simp2->noutputs())
    {
      std::cerr << "SimpleNode noutputs mismatch: " << simp1->noutputs() << " vs "
                << simp2->noutputs() << std::endl;
      return false;
    }
    for (size_t i = 0; i < simp1->noutputs(); ++i)
    {
      if (!CompareTypes(*simp1->output(i)->Type(), *simp2->output(i)->Type()))
      {
        std::cerr << "SimpleNode output type mismatch at index " << i << std::endl;
        return false;
      }
    }

    // Specialized comparisons based on operation type
    if (auto intAdd1 = dynamic_cast<const IntegerAddOperation *>(&op1))
    {
      if (!dynamic_cast<const IntegerAddOperation *>(&op2))
      {
        std::cerr << "IntegerAddOperation mismatch" << std::endl;
        return false;
      }
    }

    else if (auto intSub1 = dynamic_cast<const IntegerSubOperation *>(&op1))
    {
      if (!dynamic_cast<const IntegerSubOperation *>(&op2))
      {
        std::cerr << "IntegerSubOperation mismatch" << std::endl;
        return false;
      }
    }

    else if (auto intMul1 = dynamic_cast<const IntegerMulOperation *>(&op1))
    {
      if (!dynamic_cast<const IntegerMulOperation *>(&op2))
      {
        std::cerr << "IntegerMulOperation mismatch" << std::endl;
        return false;
      }
    }

    else if (auto fpBin1 = dynamic_cast<const FBinaryOperation *>(&op1))
    {
      auto * fpBin2 = dynamic_cast<const FBinaryOperation *>(&op2);
      if (!fpBin2)
      {
        std::cerr << "FBinaryOperation mismatch" << std::endl;
        return false;
      }
      if (fpBin1->fpop() != fpBin2->fpop())
      {
        fprintf(
            stderr,
            "FBinaryOperation fpop mismatch: %d vs %d\n",
            static_cast<int>(fpBin1->fpop()),
            static_cast<int>(fpBin2->fpop()));
        return false;
      }
    }

    else if (auto load1 = dynamic_cast<const LoadNonVolatileOperation *>(&op1))
    {
      auto * load2 = dynamic_cast<const LoadNonVolatileOperation *>(&op2);
      if (!load2)
      {
        std::cerr << "LoadOperation mismatch" << std::endl;
        return false;
      }
      if (load1->GetAlignment() != load2->GetAlignment())
      {
        std::cerr << "LoadOperation alignment mismatch: " << load1->GetAlignment() << " vs "
                  << load2->GetAlignment() << std::endl;
        return false;
      }
      if (!CompareTypes(*load1->GetLoadedType(), *load2->GetLoadedType()))
      {
        std::cerr << "LoadOperation loaded type mismatch" << std::endl;
        return false;
      }
    }

    else if (auto store1 = dynamic_cast<const StoreNonVolatileOperation *>(&op1))
    {
      auto * store2 = dynamic_cast<const StoreNonVolatileOperation *>(&op2);
      if (!store2)
      {
        std::cerr << "StoreOperation mismatch" << std::endl;
        return false;
      }
      if (store1->GetAlignment() != store2->GetAlignment())
      {
        std::cerr << "StoreOperation alignment mismatch: " << store1->GetAlignment() << " vs "
                  << store2->GetAlignment() << std::endl;
        return false;
      }
      if (!CompareTypes(store1->GetStoredType(), store2->GetStoredType()))
      {
        std::cerr << "StoreOperation stored type mismatch" << std::endl;
        return false;
      }
    }

    else if (auto alloca1 = dynamic_cast<const AllocaOperation *>(&op1))
    {
      auto * alloca2 = dynamic_cast<const AllocaOperation *>(&op2);
      if (!alloca2)
      {
        std::cerr << "AllocaOperation mismatch" << std::endl;
        return false;
      }
      if (alloca1->alignment() != alloca2->alignment())
      {
        std::cerr << "AllocaOperation alignment mismatch: " << alloca1->alignment() << " vs "
                  << alloca2->alignment() << std::endl;
        return false;
      }
      if (!CompareTypes(*alloca1->allocatedType(), *alloca2->allocatedType()))
      {
        std::cerr << "AllocaOperation allocated type mismatch" << std::endl;
        return false;
      }
    }

    else if (auto constOp1 = dynamic_cast<const BitConstantOperation *>(&op1))
    {
      auto * constOp2 = dynamic_cast<const BitConstantOperation *>(&op2);
      if (!constOp2)
      {
        std::cerr << "BitConstantOperation mismatch" << std::endl;
        return false;
      }
      if (constOp1->value() != constOp2->value())
      {
        std::cerr << "BitConstantOperation value mismatch: " << constOp1->value().to_uint()
                  << " vs " << constOp2->value().to_uint() << std::endl;
        return false;
      }
      if (!is<BitType>(*simp1->output(0)->Type()) || !is<BitType>(*simp2->output(0)->Type()))
      {
        std::cerr << "BitConstantOperation output type mismatch" << std::endl;
        return false;
      }
      auto * bt1 = static_cast<const BitType *>(simp1->output(0)->Type().get());
      auto * bt2 = static_cast<const BitType *>(simp2->output(0)->Type().get());
      if (bt1->nbits() != bt2->nbits())
      {
        std::cerr << "BitConstantOperation output bit width mismatch: " << bt1->nbits() << " vs "
                  << bt2->nbits() << std::endl;
        return false;
      }
    }

    else if (auto intConst1 = dynamic_cast<const IntegerConstantOperation *>(&op1))
    {
      auto * intConst2 = dynamic_cast<const IntegerConstantOperation *>(&op2);
      if (!intConst2)
      {
        std::cerr << "IntegerConstantOperation mismatch" << std::endl;
        return false;
      }
      if (intConst1->Representation() != intConst2->Representation())
      {
        std::cerr << "IntegerConstantOperation representation mismatch" << std::endl;
        return false;
      }
    }

    else if (auto fpConst1 = dynamic_cast<const ConstantFP *>(&op1))
    {
      auto * fpConst2 = dynamic_cast<const ConstantFP *>(&op2);
      if (!fpConst2)
      {
        std::cerr << "ConstantFP mismatch" << std::endl;
        return false;
      }
      if (!fpConst1->constant().bitwiseIsEqual(fpConst2->constant()))
      {
        std::cerr << "ConstantFP value mismatch" << std::endl;
        return false;
      }
    }

    else if (auto sext1 = dynamic_cast<const SExtOperation *>(&op1))
    {
      auto * sext2 = dynamic_cast<const SExtOperation *>(&op2);
      if (!sext2)
      {
        std::cerr << "SExtOperation mismatch" << std::endl;
        return false;
      }
      if (sext1->ndstbits() != sext2->ndstbits())
      {
        std::cerr << "SExtOperation ndstbits mismatch: " << sext1->ndstbits() << " vs "
                  << sext2->ndstbits() << std::endl;
        return false;
      }
    }

    else if (auto zext1 = dynamic_cast<const ZExtOperation *>(&op1))
    {
      auto * zext2 = dynamic_cast<const ZExtOperation *>(&op2);
      if (!zext2)
      {
        std::cerr << "ZExtOperation mismatch" << std::endl;
        return false;
      }
      if (zext1->ndstbits() != zext2->ndstbits())
      {
        std::cerr << "ZExtOperation ndstbits mismatch: " << zext1->ndstbits() << " vs "
                  << zext2->ndstbits() << std::endl;
        return false;
      }
    }

    else if (auto trunc1 = dynamic_cast<const TruncOperation *>(&op1))
    {
      auto * trunc2 = dynamic_cast<const TruncOperation *>(&op2);
      if (!trunc2)
      {
        std::cerr << "TruncOperation mismatch" << std::endl;
        return false;
      }
      if (trunc1->nsrcbits() != trunc2->nsrcbits())
      {
        std::cerr << "TruncOperation nsrcbits mismatch: " << trunc1->nsrcbits() << " vs "
                  << trunc2->nsrcbits() << std::endl;
        return false;
      }
      if (trunc1->ndstbits() != trunc2->ndstbits())
      {
        std::cerr << "TruncOperation ndstbits mismatch: " << trunc1->ndstbits() << " vs "
                  << trunc2->ndstbits() << std::endl;
        return false;
      }
    }

    else if (auto undef1 = dynamic_cast<const UndefValueOperation *>(&op1))
    {
      auto * undef2 = dynamic_cast<const UndefValueOperation *>(&op2);
      if (!undef2)
      {
        std::cerr << "UndefValueOperation mismatch" << std::endl;
        return false;
      }
      if (!CompareTypes(*simp1->output(0)->Type(), *simp2->output(0)->Type()))
      {
        std::cerr << "UndefValueOperation output type mismatch" << std::endl;
        return false;
      }
    }

    else if (auto ptrCmp1 = dynamic_cast<const PtrCmpOperation *>(&op1))
    {
      if (!dynamic_cast<const PtrCmpOperation *>(&op2))
      {
        std::cerr << "PtrCmpOperation mismatch" << std::endl;
        return false;
      }
    }

    else if (auto gep1 = dynamic_cast<const GetElementPtrOperation *>(&op1))
    {
      auto * gep2 = dynamic_cast<const GetElementPtrOperation *>(&op2);
      if (!gep2)
      {
        std::cerr << "GetElementPtrOperation mismatch" << std::endl;
        return false;
      }
      if (!CompareTypes(*gep1->getPointeeType(), *gep2->getPointeeType()))
      {
        std::cerr << "GetElementPtrOperation pointee type mismatch" << std::endl;
        return false;
      }
    }

    else if (auto fneg1 = dynamic_cast<const FNegOperation *>(&op1))
    {
      auto * fneg2 = dynamic_cast<const FNegOperation *>(&op2);
      if (!fneg2)
      {
        std::cerr << "FNegOperation mismatch" << std::endl;
        return false;
      }
      if (fneg1->size() != fneg2->size())
      {
        fprintf(
            stderr,
            "FNegOperation size mismatch: %d vs %d\n",
            static_cast<int>(fneg1->size()),
            static_cast<int>(fneg2->size()));
        return false;
      }
    }

    else if (auto cda1 = dynamic_cast<const ConstantDataArrayOperation *>(&op1))
    {
      auto * cda2 = dynamic_cast<const ConstantDataArrayOperation *>(&op2);
      if (!cda2)
      {
        std::cerr << "ConstantDataArrayOperation mismatch" << std::endl;
        return false;
      }
      if (simp1->ninputs() != simp2->ninputs())
      {
        std::cerr << "ConstantDataArrayOperation ninputs mismatch: " << simp1->ninputs() << " vs "
                  << simp2->ninputs() << std::endl;
        return false;
      }
      if (!CompareTypes(*cda1->result(0), *cda2->result(0)))
      {
        std::cerr << "ConstantDataArrayOperation result type mismatch" << std::endl;
        return false;
      }
    }

    else if (auto match1 = dynamic_cast<const MatchOperation *>(&op1))
    {
      auto * match2 = dynamic_cast<const MatchOperation *>(&op2);
      if (!match2)
      {
        std::cerr << "MatchOperation mismatch" << std::endl;
        return false;
      }
      // Compare default_alternative
      if (match1->default_alternative() != match2->default_alternative())
      {
        std::cerr << "MatchOperation default_alternative mismatch: "
                  << match1->default_alternative() << " vs " << match2->default_alternative()
                  << std::endl;
        return false;
      }
      // Compare nalternatives
      if (match1->nalternatives() != match2->nalternatives())
      {
        std::cerr << "MatchOperation nalternatives mismatch: " << match1->nalternatives() << " vs "
                  << match2->nalternatives() << std::endl;
        return false;
      }
      // Compare mapping - both need to have same key-value pairs
      // Build maps from iterators and compare
      std::unordered_map<uint64_t, uint64_t> map1, map2;
      for (auto it = match1->begin(); it != match1->end(); ++it)
        map1[it->first] = it->second;
      for (auto it = match2->begin(); it != match2->end(); ++it)
        map2[it->first] = it->second;

      if (map1.size() != map2.size())
      {
        std::cerr << "MatchOperation mapping size mismatch: " << map1.size() << " vs "
                  << map2.size() << std::endl;
        return false;
      }
      for (const auto & [key, val] : map1)
      {
        auto it = map2.find(key);
        if (it == map2.end() || it->second != val)
        {
          std::cerr << "MatchOperation mapping mismatch at key=" << key << ": value=" << val
                    << " vs " << (it != map2.end() ? std::to_string(it->second) : "not found")
                    << std::endl;
          return false;
        }
      }
    }

    else
    {
      std::cerr << "Unknown SimpleNode operation: " << typeid(op1).name() << std::endl;
      return false;
    }

    return true;
  }

  std::cerr << "Node comparison failed: node1 is SimpleNode=" << !!simp1
            << ", StructuralNode=" << !!snode1 << "; node2 is SimpleNode=" << !!simp2
            << ", StructuralNode=" << !!snode2 << std::endl;
  return false;
}

/**
 * \brief Compares two RVSDG regions for equality by traversing through results
 * and verifying the same graph structure exists in both regions.
 *
 * \param region1 The first region to compare.
 * \param region2 The second region to compare.
 * \return True if the regions are equal, false otherwise.
 */
bool
CompareRegions(const Region & region1, const Region & region2)
{
  // Check number of arguments
  if (region1.narguments() != region2.narguments())
  {
    std::cerr << "Region narguments mismatch: " << region1.narguments() << " vs "
              << region2.narguments() << std::endl;
    return false;
  }
  for (size_t i = 0; i < region1.narguments(); ++i)
  {
    if (!CompareTypes(*region1.argument(i)->Type(), *region2.argument(i)->Type()))
    {
      std::cerr << "Region argument type mismatch at index " << i << std::endl;
      return false;
    }
  }

  // Check number of results
  if (region1.nresults() != region2.nresults())
  {
    std::cerr << "Region nresults mismatch: " << region1.nresults() << " vs " << region2.nresults()
              << std::endl;
    return false;
  }
  for (size_t i = 0; i < region1.nresults(); ++i)
  {
    if (!CompareTypes(*region1.result(i)->Type(), *region2.result(i)->Type()))
    {
      std::cerr << "Region result type mismatch at index " << i << std::endl;
      return false;
    }
  }

  // Check node count
  if (region1.numNodes() != region2.numNodes())
  {
    std::cerr << "Region numNodes mismatch: " << region1.numNodes() << " vs " << region2.numNodes()
              << std::endl;
    return false;
  }

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

      if (!origin1 || !origin2)
      {
        std::cerr << "Result origin is null at index " << i << ": "
                  << "region1=" << !!origin1 << ", region2=" << !!origin2 << std::endl;
        return false;
      }

      if (!CompareTypes(*origin1->Type(), *origin2->Type()))
      {
        std::cerr << "Result origin type mismatch at index " << i << std::endl;
        return false;
      }

      auto * n1 = TryGetOwnerNode<Node>(*origin1);
      auto * n2 = TryGetOwnerNode<Node>(*origin2);
      if (n1 && n2)
      {
        if (!CompareNodes(*n1, *n2))
        {
          std::cerr << "CompareNodes failed for nodes connected to result " << i << " : " << n1
                    << " and " << n2 << std::endl;
          return false;
        }

        visited1.insert(n1);
        visited2.insert(n2);
        nodeQueue.push({ n1, n2 });
      }
      else if (!n1 && !n2)
      {
        // Result is an argument - no traversal needed
        continue;
      }
      else
      {
        std::cerr << "Result origin ownership mismatch at index " << i << ": "
                  << "node1=" << !!n1 << ", node2=" << !!n2 << std::endl;
        return false;
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

      if (!origin1 || !origin2)
      {
        std::cerr << "Input origin mismatch for node inputs at index " << j << ": "
                  << "node1=" << n1 << ", node2=" << n2 << ", origin1=" << !!origin1
                  << ", origin2=" << !!origin2 << std::endl;
        return false;
      }

      // If origin is a node output (not an argument), find matching nodes
      auto * next1 = TryGetOwnerNode<Node>(*origin1);
      auto * next2 = TryGetOwnerNode<Node>(*origin2);
      if (next1 && next2)
      {
        if (!visited1.count(next1))
        {
          if (!CompareNodes(*next1, *next2))
          {
            std::cerr << "BFS traversal: CompareNodes failed for follow-up nodes " << next1
                      << " and " << next2 << std::endl;
            return false;
          }

          visited1.insert(next1);
          visited2.insert(next2);
          nodeQueue.push({ next1, next2 });
        }
      }
    }
  }

  // Verify all nodes were visited
  if (visited1.size() != region1.numNodes())
  {
    std::cerr << "Node count mismatch after traversal: visited=" << visited1.size()
              << ", expected=" << region1.numNodes() << std::endl;
    return false;
  }
  if (visited2.size() != region2.numNodes())
  {
    std::cerr << "Node count mismatch for region2 after traversal: visited=" << visited2.size()
              << ", expected=" << region2.numNodes() << std::endl;
    return false;
  }

  return true;
}

/**
 * \brief Compares two LlvmRvsdgModule instances for equality.
 *
 * \param module1 The first module to compare.
 * \param module2 The second module to compare.
 * \return True if the modules are equal, false otherwise.
 */
bool
CompareModules(const LlvmRvsdgModule & module1, const LlvmRvsdgModule & module2)
{
  return CompareRegions(module1.Rvsdg().GetRootRegion(), module2.Rvsdg().GetRootRegion());
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
  ASSERT_TRUE(CompareModules(originalModule, *roundTripModule))
      << "Roundtrip failed for test: " << testName;
}

} // namespace

TEST(RvsdgRoundtripTests, TestGamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType = BitType::Create(1);
  auto functionType = FunctionType::Create({ bitType, bitType, bitType }, { bitType });

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");

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
