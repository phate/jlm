/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <math.h>

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-prints.hpp>
#include <jlm/hls/backend/rvsdg2rhls/instrument-ref.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/load.hpp>
#include <jlm/llvm/ir/operators/store.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

llvm::lambda::node *
change_function_name(llvm::lambda::node * ln, const std::string & name)
{
  auto lambda =
      llvm::lambda::node::create(ln->region(), ln->type(), name, ln->linkage(), ln->attributes());

  /* add context variables */
  jlm::rvsdg::substitution_map subregionmap;
  for (auto & cv : ln->ctxvars())
  {
    auto origin = cv.origin();
    auto newcv = lambda->add_ctxvar(origin);
    subregionmap.insert(cv.argument(), newcv);
  }

  /* collect function arguments */
  for (size_t n = 0; n < ln->nfctarguments(); n++)
  {
    lambda->fctargument(n)->set_attributes(ln->fctargument(n)->attributes());
    subregionmap.insert(ln->fctargument(n), lambda->fctargument(n));
  }

  /* copy subregion */
  ln->subregion()->copy(lambda->subregion(), subregionmap, false, false);

  /* collect function results */
  std::vector<jlm::rvsdg::output *> results;
  for (auto & result : ln->fctresults())
    results.push_back(subregionmap.lookup(result.origin()));

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
  auto root = graph.root();
  auto lambda = dynamic_cast<llvm::lambda::node *>(root->nodes.begin().ptr());

  auto newLambda = change_function_name(lambda, "instrumented_ref");

//  auto lambdaRegion = newLambda->subregion();
  auto functionType = newLambda->type();
  auto numArguments = functionType.NumArguments();
//  auto numArguments = lambdaRegion->narguments();
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
  JLM_ASSERT(rvsdg::is<llvm::iostatetype>(functionType.ArgumentType(ioStateArgumentIndex)));

  // TODO: make this less hacky by using the correct state types
  //  addr, width, memstate
  jlm::llvm::FunctionType loadFunctionType(
      { jlm::llvm::PointerType::Create().get(),
        &jlm::rvsdg::bit64,
        llvm::iostatetype::create().get(),
        llvm::MemoryStateType::Create().get() },
      { llvm::iostatetype::create().get(), llvm::MemoryStateType::Create().get() });
  jlm::llvm::impport load_imp(
      loadFunctionType,
      "reference_load",
      jlm::llvm::linkage::external_linkage);
  auto reference_load = graph.add_import(load_imp);
  // addr, data, width, memstate
  jlm::llvm::FunctionType storeFunctionType(
      { jlm::llvm::PointerType::Create().get(),
        &jlm::rvsdg::bit64,
        &jlm::rvsdg::bit64,
        llvm::iostatetype::create().get(),
        jlm::llvm::MemoryStateType::Create().get() },
      { llvm::iostatetype::create().get(), jlm::llvm::MemoryStateType::Create().get() });
  jlm::llvm::impport store_imp(
      storeFunctionType,
      "reference_store",
      jlm::llvm::linkage::external_linkage);
  auto reference_store = graph.add_import(store_imp);
  // addr, size, memstate
  jlm::llvm::FunctionType allocaFunctionType(
      { jlm::llvm::PointerType::Create().get(),
        &jlm::rvsdg::bit64,
        llvm::iostatetype::create().get(),
        jlm::llvm::MemoryStateType::Create().get() },
      { llvm::iostatetype::create().get(), jlm::llvm::MemoryStateType::Create().get() });
  jlm::llvm::impport alloca_imp(
      allocaFunctionType,
      "reference_alloca",
      jlm::llvm::linkage::external_linkage);
  auto reference_alloca = graph.add_import(alloca_imp);

  instrument_ref(
      root,
      newLambda->subregion()->argument(ioStateArgumentIndex),
      reference_load,
      loadFunctionType,
      reference_store,
      storeFunctionType,
      reference_alloca,
      allocaFunctionType);
}

void
instrument_ref(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * ioState,
    jlm::rvsdg::output * load_func,
    jlm::llvm::FunctionType & loadFunctionType,
    jlm::rvsdg::output * store_func,
    jlm::llvm::FunctionType & storeFunctionType,
    jlm::rvsdg::output * alloca_func,
    jlm::llvm::FunctionType & allocaFunctionType)
{
  load_func = route_to_region(load_func, region);
  store_func = route_to_region(store_func, region);
  alloca_func = route_to_region(alloca_func, region);
  jlm::llvm::PointerType void_ptr;
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        auto subregion = structnode->subregion(n);
        auto ioStateRouted = route_to_region(ioState, subregion);
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
    else if (auto loadOp = dynamic_cast<const jlm::llvm::LoadOperation *>(&(node->operation())))
    {
      auto addr = node->input(0)->origin();
      JLM_ASSERT(dynamic_cast<const jlm::llvm::PointerType *>(&addr->type()));
      size_t bitWidth = BaseHLS::JlmSize(&loadOp->GetLoadedType());
      int log2Bytes = log2(bitWidth / 8);
      auto width = jlm::rvsdg::create_bitconstant(region, 64, log2Bytes);

      // Does this IF make sense now when the void_ptr doesn't have a type?
      if (addr->type() != void_ptr)
      {
        addr = jlm::llvm::bitcast_op::create(addr, void_ptr);
      }
      auto memstate = node->input(1)->origin();
      auto callOp = jlm::llvm::CallNode::Create(
          load_func,
          loadFunctionType,
          { addr, width, ioState, memstate });
      // Divert the memory state of the load to the new memstate from the call operation
      node->input(1)->divert_to(callOp[1]);
    }
    else if (auto ao = dynamic_cast<const jlm::llvm::alloca_op *>(&(node->operation())))
    {
      // ensure that the size is one
      JLM_ASSERT(node->ninputs() == 1);
      auto constant_output = dynamic_cast<jlm::rvsdg::node_output *>(node->input(0)->origin());
      JLM_ASSERT(constant_output);
      auto constant_operation =
          dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&constant_output->node()->operation());
      JLM_ASSERT(constant_operation);
      JLM_ASSERT(constant_operation->value().to_uint() == 1);
      jlm::rvsdg::output * addr = node->output(0);
      // ensure that the alloca is an array type
      auto pt = dynamic_cast<const jlm::llvm::PointerType *>(&addr->type());
      JLM_ASSERT(pt);
      auto at = dynamic_cast<const jlm::llvm::arraytype *>(&ao->value_type());
      JLM_ASSERT(at);
      auto size = jlm::rvsdg::create_bitconstant(region, 64, BaseHLS::JlmSize(at) / 8);

      // Does this IF make sense now when the void_ptr doesn't have a type?
      if (addr->type() != void_ptr)
      {
        addr = jlm::llvm::bitcast_op::create(addr, void_ptr);
      }
      std::vector<jlm::rvsdg::input *> old_users(node->output(1)->begin(), node->output(1)->end());
      auto memstate = node->output(1);
      auto callOp = jlm::llvm::CallNode::Create(
          alloca_func,
          allocaFunctionType,
          { addr, size, ioState, memstate });
      for (auto ou : old_users)
      {
        // Divert the memory state of the load to the new memstate from the call operation
        ou->divert_to(callOp[1]);
      }
    }
    else if (auto so = dynamic_cast<const jlm::llvm::StoreOperation *>(&(node->operation())))
    {
      auto addr = node->input(0)->origin();
      JLM_ASSERT(dynamic_cast<const jlm::llvm::PointerType *>(&addr->type()));
      auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&so->GetStoredType());
      JLM_ASSERT(bt);
      auto bitWidth = bt->nbits();
      int log2Bytes = log2(bitWidth / 8);
      auto width = jlm::rvsdg::create_bitconstant(region, 64, log2Bytes);

      // Does this IF make sense now when the void_ptr doesn't have a type?
      if (addr->type() != void_ptr)
      {
        addr = jlm::llvm::bitcast_op::create(addr, void_ptr);
      }
      auto data = node->input(1)->origin();
      auto dbt = dynamic_cast<const jlm::rvsdg::bittype *>(&data->type());
      if (*dbt != jlm::rvsdg::bit64)
      {
        jlm::llvm::zext_op op(dbt->nbits(), 64);
        data = jlm::rvsdg::simple_node::create_normalized(data->region(), op, { data })[0];
      }
      auto memstate = node->input(2)->origin();
      auto callOp = jlm::llvm::CallNode::Create(
          store_func,
          storeFunctionType,
          { addr, data, width, ioState, memstate });
      // Divert the memory state of the load to the new memstate from the call operation
      node->input(2)->divert_to(callOp[1]);
    }
  }
}

} // namespace jlm::hls
