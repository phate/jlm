/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-prints.hpp>
#include <jlm/hls/backend/rvsdg2rhls/instrument-ref.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <cmath>

namespace jlm::hls
{

rvsdg::LambdaNode *
change_function_name(rvsdg::LambdaNode * ln, const std::string & name)
{
  const auto & op = dynamic_cast<llvm::LlvmLambdaOperation &>(ln->GetOperation());
  auto lambda = rvsdg::LambdaNode::Create(
      *ln->region(),
      llvm::LlvmLambdaOperation::Create(op.Type(), name, op.linkage(), op.attributes()));

  /* add context variables */
  rvsdg::SubstitutionMap subregionmap;
  for (const auto & cv : ln->GetContextVars())
  {
    auto origin = cv.input->origin();
    auto newcv = lambda->AddContextVar(*origin);
    subregionmap.insert(cv.inner, newcv.inner);
  }
  /* collect function arguments */
  auto args = ln->GetFunctionArguments();
  auto newArgs = lambda->GetFunctionArguments();
  JLM_ASSERT(args.size() == newArgs.size());
  for (std::size_t n = 0; n < args.size(); ++n)
  {
    subregionmap.insert(args[n], newArgs[n]);
  }

  /* copy subregion */
  ln->subregion()->copy(lambda->subregion(), subregionmap, false, false);

  /* collect function results */
  std::vector<jlm::rvsdg::output *> results;
  for (auto result : ln->GetFunctionResults())
    results.push_back(subregionmap.lookup(result->origin()));

  /* finalize lambda */
  lambda->finalize(results);

  divert_users(ln, outputs(lambda));
  jlm::rvsdg::remove(ln);

  return lambda;
}

void
instrument_ref(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  auto lambda = dynamic_cast<rvsdg::LambdaNode *>(root->Nodes().begin().ptr());

  auto newLambda = change_function_name(lambda, "instrumented_ref");

  auto functionType = newLambda->GetOperation().type();
  auto numArguments = functionType.NumArguments();
  if (numArguments == 0)
  {
    // The lambda has no arguments so it shouldn't have any memory operations
    return;
  }

  auto memStateArgumentIndex = numArguments - 1;
  if (!rvsdg::is<llvm::MemoryStateType>(functionType.ArgumentType(memStateArgumentIndex)))
  {
    // The lambda has no memory state so it shouldn't have any memory operations
    return;
  }
  // The function should always have an IO state if it has a memory state
  auto ioStateArgumentIndex = numArguments - 2;
  JLM_ASSERT(rvsdg::is<llvm::IOStateType>(functionType.ArgumentType(ioStateArgumentIndex)));

  //  addr, width, memstate
  auto loadFunctionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(),
        jlm::rvsdg::bittype::Create(64),
        llvm::IOStateType::Create(),
        llvm::MemoryStateType::Create() },
      { llvm::IOStateType::Create(), llvm::MemoryStateType::Create() });
  auto & reference_load = llvm::GraphImport::Create(
      graph,
      loadFunctionType,
      loadFunctionType,
      "reference_load",
      llvm::linkage::external_linkage);
  auto & reference_store = llvm::GraphImport::Create(
      graph,
      loadFunctionType,
      loadFunctionType,
      "reference_store",
      llvm::linkage::external_linkage);
  // addr, size, memstate
  auto allocaFunctionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(),
        jlm::rvsdg::bittype::Create(64),
        llvm::IOStateType::Create(),
        jlm::llvm::MemoryStateType::Create() },
      { llvm::IOStateType::Create(), jlm::llvm::MemoryStateType::Create() });
  auto & reference_alloca = llvm::GraphImport::Create(
      graph,
      allocaFunctionType,
      allocaFunctionType,
      "reference_alloca",
      llvm::linkage::external_linkage);

  instrument_ref(
      root,
      newLambda->subregion()->argument(ioStateArgumentIndex),
      &reference_load,
      loadFunctionType,
      &reference_store,
      loadFunctionType,
      &reference_alloca,
      allocaFunctionType);
}

void
instrument_ref(
    rvsdg::Region * region,
    jlm::rvsdg::output * ioState,
    jlm::rvsdg::output * load_func,
    const std::shared_ptr<const jlm::rvsdg::FunctionType> & loadFunctionType,
    jlm::rvsdg::output * store_func,
    const std::shared_ptr<const jlm::rvsdg::FunctionType> & storeFunctionType,
    jlm::rvsdg::output * alloca_func,
    const std::shared_ptr<const jlm::rvsdg::FunctionType> & allocaFunctionType)
{
  load_func = route_to_region_rvsdg(load_func, region);
  store_func = route_to_region_rvsdg(store_func, region);
  alloca_func = route_to_region_rvsdg(alloca_func, region);
  auto void_ptr = jlm::llvm::PointerType::Create();
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        auto subregion = structnode->subregion(n);
        auto ioStateRouted = route_to_region_rvsdg(ioState, subregion);
        instrument_ref(
            subregion,
            ioStateRouted,
            load_func,
            loadFunctionType,
            store_func,
            storeFunctionType,
            alloca_func,
            allocaFunctionType);
      }
    }
    else if (
        auto loadOp =
            dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(&(node->GetOperation())))
    {
      auto addr = node->input(0)->origin();
      JLM_ASSERT(dynamic_cast<const jlm::llvm::PointerType *>(&addr->type()));
      size_t bitWidth = BaseHLS::JlmSize(&*loadOp->GetLoadedType());
      int log2Bytes = log2(bitWidth / 8);
      auto & widthNode = llvm::IntegerConstantOperation::Create(*region, 64, log2Bytes);

      // Does this IF make sense now when the void_ptr doesn't have a type?
      if (*addr->Type() != *void_ptr)
      {
        addr = jlm::llvm::bitcast_op::create(addr, void_ptr);
      }
      auto memstate = node->input(1)->origin();
      auto callOp = jlm::llvm::CallOperation::Create(
          load_func,
          loadFunctionType,
          { addr, widthNode.output(0), ioState, memstate });
      // Divert the memory state of the load to the new memstate from the call operation
      node->input(1)->divert_to(callOp[1]);
    }
    else if (auto ao = dynamic_cast<const jlm::llvm::alloca_op *>(&(node->GetOperation())))
    {
      // ensure that the size is one
      JLM_ASSERT(node->ninputs() == 1);
      auto constant_output = dynamic_cast<jlm::rvsdg::node_output *>(node->input(0)->origin());
      JLM_ASSERT(constant_output);
      auto constant_operation = dynamic_cast<const llvm::IntegerConstantOperation *>(
          &constant_output->node()->GetOperation());
      JLM_ASSERT(constant_operation);
      JLM_ASSERT(constant_operation->Representation().to_uint() == 1);
      jlm::rvsdg::output * addr = node->output(0);
      // ensure that the alloca is an array type
      JLM_ASSERT(jlm::rvsdg::is<const llvm::PointerType>(addr->Type()));
      auto at = dynamic_cast<const llvm::ArrayType *>(&ao->value_type());
      JLM_ASSERT(at);
      auto & sizeNode =
          llvm::IntegerConstantOperation::Create(*region, 64, BaseHLS::JlmSize(at) / 8);

      // Does this IF make sense now when the void_ptr doesn't have a type?
      if (*addr->Type() != *void_ptr)
      {
        addr = jlm::llvm::bitcast_op::create(addr, void_ptr);
      }
      std::vector<jlm::rvsdg::input *> old_users(node->output(1)->begin(), node->output(1)->end());
      auto memstate = node->output(1);
      auto callOp = jlm::llvm::CallOperation::Create(
          alloca_func,
          allocaFunctionType,
          { addr, sizeNode.output(0), ioState, memstate });
      for (auto ou : old_users)
      {
        // Divert the memory state of the load to the new memstate from the call operation
        ou->divert_to(callOp[1]);
      }
    }
    else if (
        auto so =
            dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&(node->GetOperation())))
    {
      auto addr = node->input(0)->origin();
      JLM_ASSERT(dynamic_cast<const jlm::llvm::PointerType *>(&addr->type()));
      auto bitWidth = JlmSize(&so->GetStoredType());
      int log2Bytes = log2(bitWidth / 8);
      auto & widthNode = llvm::IntegerConstantOperation::Create(*region, 64, log2Bytes);

      // Does this IF make sense now when the void_ptr doesn't have a type?
      if (*addr->Type() != *void_ptr)
      {
        addr = jlm::llvm::bitcast_op::create(addr, void_ptr);
      }
      auto memstate = node->output(0);
      std::vector<jlm::rvsdg::input *> oldUsers(memstate->begin(), memstate->end());
      auto callOp = jlm::llvm::CallOperation::Create(
          store_func,
          storeFunctionType,
          { addr, widthNode.output(0), ioState, memstate });
      // Divert the memory state after the store to the new memstate from the call operation
      for (auto user : oldUsers)
      {
        user->divert_to(callOp[1]);
      }
    }
  }
}

} // namespace jlm::hls
