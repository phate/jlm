/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/mlir/backend/JlmToMlirConverter.hpp>
#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>

namespace
{

using namespace jlm::llvm;
using namespace jlm::rvsdg;

bool
CompareTypes(const Type & type1, const Type & type2);

bool
CompareOutputs(const Output & output1, const Output & output2);

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
    return bit1 && bit2 && bit1->nbits() == bit2->nbits();
  }
  else if (is<PointerType>(type1))
  {
    return is<PointerType>(type2);
  }
  else if (is<MemoryStateType>(type1))
  {
    return is<MemoryStateType>(type2);
  }
  else if (is<IOStateType>(type1))
  {
    return is<IOStateType>(type2);
  }
  else if (auto * func1 = dynamic_cast<const FunctionType *>(&type1))
  {
    auto * func2 = dynamic_cast<const FunctionType *>(&type2);
    if (!func2 || func1->NumArguments() != func2->NumArguments()
        || func1->NumResults() != func2->NumResults())
    {
      return false;
    }

    for (size_t i = 0; i < func1->NumArguments(); ++i)
    {
      if (!CompareTypes(func1->ArgumentType(i), func2->ArgumentType(i)))
      {
        return false;
      }
    }

    for (size_t i = 0; i < func1->NumResults(); ++i)
    {
      if (!CompareTypes(func1->ResultType(i), func2->ResultType(i)))
      {
        return false;
      }
    }

    return true;
  }
  else if (auto * array1 = dynamic_cast<const ArrayType *>(&type1))
  {
    auto * array2 = dynamic_cast<const ArrayType *>(&type2);
    if (!array2 || array1->nelements() != array2->nelements())
    {
      return false;
    }
    return CompareTypes(array1->element_type(), array2->element_type());
  }
  else if (auto * struct1 = dynamic_cast<const StructType *>(&type1))
  {
    auto * struct2 = dynamic_cast<const StructType *>(&type2);
    if (!struct2 || struct1->numElements() != struct2->numElements())
    {
      return false;
    }

    for (size_t i = 0; i < struct1->numElements(); ++i)
    {
      if (!CompareTypes(*struct1->getElementType(i), *struct2->getElementType(i)))
      {
        return false;
      }
    }

    return true;
  }

  // Default: use operator== for comparison
  return type1 == type2;
}

/**
 * \brief Compares two RVSDG outputs for equality by checking their origins.
 *
 * \param output1 The first output to compare.
 * \param output2 The second output to compare.
 * \return True if the outputs are equal, false otherwise.
 */
bool
CompareOutputs(const Output & output1, const Output & output2)
{
  // Check if both are arguments (imports/inputs to the graph)
  auto * arg1 = dynamic_cast<const RegionArgument *>(&output1);
  auto * arg2 = dynamic_cast<const RegionArgument *>(&output2);
  if (arg1 && arg2)
  {
    return CompareTypes(*arg1->Type(), *arg2->Type());
  }

  // Check if both are results
  auto * res1 = dynamic_cast<const RegionResult *>(&output1);
  auto * res2 = dynamic_cast<const RegionResult *>(&output2);
  if (res1 && res2)
  {
    return CompareTypes(*res1->Type(), *res2->Type());
  }

  // Check if both are node outputs
  auto * node1 = TryGetOwnerNode<Node>(output1);
  auto * node2 = TryGetOwnerNode<Node>(output2);
  if (node1 && node2)
  {
    return CompareNodes(*node1, *node2);
  }

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
    if (snode1->nsubregions() != snode2->nsubregions() || snode1->ninputs() != snode2->ninputs()
        || snode1->noutputs() != snode2->noutputs())
    {
      return false;
    }

    // Compare each region recursively
    for (size_t r = 0; r < snode1->nsubregions(); ++r)
    {
      if (!CompareRegions(*snode1->subregion(r), *snode2->subregion(r)))
      {
        return false;
      }
    }

    // Compare input types
    for (size_t i = 0; i < snode1->ninputs(); ++i)
    {
      if (!CompareTypes(*snode1->input(i)->Type(), *snode2->input(i)->Type()))
      {
        return false;
      }
    }

    // Compare output types
    for (size_t i = 0; i < snode1->noutputs(); ++i)
    {
      if (!CompareTypes(*snode1->output(i)->Type(), *snode2->output(i)->Type()))
      {
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
      return false;
    }

    // Compare input counts and types
    if (simp1->ninputs() != simp2->ninputs())
    {
      return false;
    }
    for (size_t i = 0; i < simp1->ninputs(); ++i)
    {
      if (!CompareTypes(*simp1->input(i)->Type(), *simp2->input(i)->Type()))
      {
        return false;
      }
    }

    // Compare output counts and types
    if (simp1->noutputs() != simp2->noutputs())
    {
      return false;
    }
    for (size_t i = 0; i < simp1->noutputs(); ++i)
    {
      if (!CompareTypes(*simp1->output(i)->Type(), *simp2->output(i)->Type()))
      {
        return false;
      }
    }

    if (auto intAdd1 = dynamic_cast<const IntegerAddOperation *>(&op1))
    {
      return dynamic_cast<const IntegerAddOperation *>(&op2) != nullptr;
    }
    else if (auto intSub1 = dynamic_cast<const IntegerSubOperation *>(&op1))
    {
      return dynamic_cast<const IntegerSubOperation *>(&op2) != nullptr;
    }
    else if (auto intMul1 = dynamic_cast<const IntegerMulOperation *>(&op1))
    {
      return dynamic_cast<const IntegerMulOperation *>(&op2) != nullptr;
    }
    else if (auto fpBin1 = dynamic_cast<const FBinaryOperation *>(&op1))
    {
      auto * fpBin2 = dynamic_cast<const FBinaryOperation *>(&op2);
      return fpBin2 && fpBin1->fpop() == fpBin2->fpop();
    }
    else if (auto load1 = dynamic_cast<const LoadNonVolatileOperation *>(&op1))
    {
      auto * load2 = dynamic_cast<const LoadNonVolatileOperation *>(&op2);
      return load2 && load1->GetAlignment() == load2->GetAlignment()
          && CompareTypes(*load1->GetLoadedType(), *load2->GetLoadedType());
    }
    else if (auto store1 = dynamic_cast<const StoreNonVolatileOperation *>(&op1))
    {
      auto * store2 = dynamic_cast<const StoreNonVolatileOperation *>(&op2);
      return store2 && store1->GetAlignment() == store2->GetAlignment()
          && CompareTypes(store1->GetStoredType(), store2->GetStoredType());
    }
    else if (auto alloca1 = dynamic_cast<const AllocaOperation *>(&op1))
    {
      auto * alloca2 = dynamic_cast<const AllocaOperation *>(&op2);
      return alloca2 && alloca1->alignment() == alloca2->alignment()
          && CompareTypes(*alloca1->allocatedType(), *alloca2->allocatedType());
    }
    else if (auto constOp1 = dynamic_cast<const BitConstantOperation *>(&op1))
    {
      auto * constOp2 = dynamic_cast<const BitConstantOperation *>(&op2);
      return constOp2 && constOp1->value() == constOp2->value()
          && is<BitType>(*simp1->output(0)->Type()) && is<BitType>(*simp2->output(0)->Type())
          && static_cast<const BitType *>(simp1->output(0)->Type().get())->nbits()
                 == static_cast<const BitType *>(simp2->output(0)->Type().get())->nbits();
    }
    else if (auto intConst1 = dynamic_cast<const IntegerConstantOperation *>(&op1))
    {
      auto * intConst2 = dynamic_cast<const IntegerConstantOperation *>(&op2);
      return intConst2 && intConst1->Representation() == intConst2->Representation();
    }
    else if (auto fpConst1 = dynamic_cast<const ConstantFP *>(&op1))
    {
      auto * fpConst2 = dynamic_cast<const ConstantFP *>(&op2);
      return fpConst2 && fpConst1->constant().bitwiseIsEqual(fpConst2->constant());
    }
    else if (auto sext1 = dynamic_cast<const SExtOperation *>(&op1))
    {
      auto * sext2 = dynamic_cast<const SExtOperation *>(&op2);
      return sext2 && sext1->ndstbits() == sext2->ndstbits();
    }
    else if (auto zext1 = dynamic_cast<const ZExtOperation *>(&op1))
    {
      auto * zext2 = dynamic_cast<const ZExtOperation *>(&op2);
      return zext2 && zext1->ndstbits() == zext2->ndstbits();
    }
    else if (auto trunc1 = dynamic_cast<const TruncOperation *>(&op1))
    {
      auto * trunc2 = dynamic_cast<const TruncOperation *>(&op2);
      return trunc2 && trunc1->nsrcbits() == trunc2->nsrcbits()
          && trunc1->ndstbits() == trunc2->ndstbits();
    }
    else if (auto undef1 = dynamic_cast<const UndefValueOperation *>(&op1))
    {
      // UndefValueOperation has no Type() method, compare based on output type
      auto * undef2 = dynamic_cast<const UndefValueOperation *>(&op2);
      return undef2 && CompareTypes(*simp1->output(0)->Type(), *simp2->output(0)->Type());
    }
    else if (auto ptrCmp1 = dynamic_cast<const PtrCmpOperation *>(&op1))
    {
      return dynamic_cast<const PtrCmpOperation *>(&op2) != nullptr;
    }
    else if (auto gep1 = dynamic_cast<const GetElementPtrOperation *>(&op1))
    {
      auto * gep2 = dynamic_cast<const GetElementPtrOperation *>(&op2);
      return gep2 && CompareTypes(*gep1->getPointeeType(), *gep2->getPointeeType());
    }
    else if (auto fneg1 = dynamic_cast<const FNegOperation *>(&op1))
    {
      auto * fneg2 = dynamic_cast<const FNegOperation *>(&op2);
      return fneg2 && fneg1->size() == fneg2->size();
    }
    else if (auto cda1 = dynamic_cast<const ConstantDataArrayOperation *>(&op1))
    {
      auto * cda2 = dynamic_cast<const ConstantDataArrayOperation *>(&op2);
      return cda2 && simp1->ninputs() == simp2->ninputs()
          && CompareTypes(*cda1->result(0), *cda2->result(0));
    }

    return false;
  }

  return false;
}

/**
 * \brief Compares two RVSDG regions for equality by checking all nodes and connections.
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
    return false;
  }
  // Compare argument types
  for (size_t i = 0; i < region1.narguments(); ++i)
  {
    if (!CompareTypes(*region1.argument(i)->Type(), *region2.argument(i)->Type()))
    {
      return false;
    }
  }

  // Check number of results
  if (region1.nresults() != region2.nresults())
  {
    return false;
  }
  // Compare result types
  for (size_t i = 0; i < region1.nresults(); ++i)
  {
    if (!CompareTypes(*region1.result(i)->Type(), *region2.result(i)->Type()))
    {
      return false;
    }
  }

  // Check node count
  if (region1.numNodes() != region2.numNodes())
  {
    return false;
  }

  // Compare nodes in order
  auto it1 = region1.Nodes().begin();
  auto it2 = region2.Nodes().begin();

  for (; it1 != region1.Nodes().end() && it2 != region2.Nodes().end(); ++it1, ++it2)
  {
    if (!CompareNodes(*it1, *it2))
    {
      return false;
    }

    // Check input connections
    auto & node1 = *it1;
    auto & node2 = *it2;

    if (node1.ninputs() != node2.ninputs())
    {
      return false;
    }

    for (size_t j = 0; j < node1.ninputs(); ++j)
    {
      auto * origin1 = node1.input(j)->origin();
      auto * origin2 = node2.input(j)->origin();

      if (!origin1 && !origin2)
      {
        continue;
      }

      if (!origin1 || !origin2 || !CompareOutputs(*origin1, *origin2))
      {
        return false;
      }
    }
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

// Test that CompareTypes works correctly
TEST(RvsdgRoundtripTests, TestBitTypeComparison)
{
  using namespace jlm::rvsdg;

  auto bit32 = BitType::Create(32);
  auto bit64 = BitType::Create(64);

  EXPECT_TRUE(CompareTypes(*bit32, *BitType::Create(32)));
  EXPECT_FALSE(CompareTypes(*bit32, *bit64));
}

TEST(RvsdgRoundtripTests, TestPointerTypeComparison)
{
  using namespace jlm::llvm;

  auto ptr1 = PointerType::Create();
  auto ptr2 = PointerType::Create();

  EXPECT_TRUE(CompareTypes(*ptr1, *ptr2));
}

TEST(RvsdgRoundtripTests, TestFunctionTypeComparison)
{
  using namespace jlm::rvsdg;

  auto bit32 = BitType::Create(32);
  auto bit64 = BitType::Create(64);

  auto func1Args = std::vector<std::shared_ptr<const Type>>{ bit32, bit64 };
  auto func1Results = std::vector<std::shared_ptr<const Type>>{ bit64 };
  auto func1 = FunctionType::Create(func1Args, func1Results);

  auto func2Args = std::vector<std::shared_ptr<const Type>>{ bit32, bit64 };
  auto func2Results = std::vector<std::shared_ptr<const Type>>{ bit64 };
  auto func2 = FunctionType::Create(func2Args, func2Results);

  EXPECT_TRUE(CompareTypes(*func1, *func2));

  auto func3Args = std::vector<std::shared_ptr<const Type>>{ bit64 };
  auto func3Results = std::vector<std::shared_ptr<const Type>>{ bit32 };
  auto func3 = FunctionType::Create(func3Args, func3Results);

  EXPECT_FALSE(CompareTypes(*func1, *func3));
}

TEST(RvsdgRoundtripTests, TestArrayTypeComparison)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto element = BitType::Create(32);
  auto arr1 = ArrayType::Create(element, 5);
  auto arr2 = ArrayType::Create(element, 5);
  auto arr3 = ArrayType::Create(element, 10);

  EXPECT_TRUE(CompareTypes(*arr1, *arr2));
  EXPECT_FALSE(CompareTypes(*arr1, *arr3));
}

TEST(RvsdgRoundtripTests, TestLambdaWithAdd)
{
  using namespace jlm::llvm;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule->Rvsdg();

  auto bitType = BitType::Create(64);
  auto functionType = FunctionType::Create({ bitType, bitType }, { bitType });

  auto lambda = LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::internalLinkage));

  auto arg1 = lambda->GetFunctionArguments().at(0);
  auto arg2 = lambda->GetFunctionArguments().at(1);

  IntegerAddOperation addOp(64);
  SimpleNode::Create(*lambda->subregion(), addOp.copy(), { arg1, arg2 });

  auto & subregion = *lambda->subregion();
  lambda->finalize({ subregion.Nodes().begin().ptr()->output(0) });

  TestRvsdgRoundtrip(*rvsdgModule, "TestLambdaWithAdd");
}
