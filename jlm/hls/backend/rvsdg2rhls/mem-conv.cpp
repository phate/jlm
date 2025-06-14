/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * and Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/UnusedStateRemoval.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::hls
{
rvsdg::SimpleNode *
find_decouple_response(
    const rvsdg::LambdaNode * lambda,
    const llvm::IntegerConstantOperation * request_constant)
{
  auto response_functions = find_function_arguments(lambda, "decouple_res");
  for (auto & func : response_functions)
  {
    std::unordered_set<rvsdg::Output *> visited;
    std::vector<rvsdg::SimpleNode *> reponse_calls;
    trace_function_calls(func.inner, reponse_calls, visited);
    for (auto & rc : reponse_calls)
    {
      auto response_constant = trace_constant(rc->input(1)->origin());
      if (*response_constant == *request_constant)
      {
        return rc;
      }
    }
  }
  JLM_UNREACHABLE("No response found");
}

std::pair<rvsdg::SimpleInput *, std::vector<rvsdg::SimpleInput *>>
TraceEdgeToMerge(rvsdg::Input * state_edge)
{
  std::vector<rvsdg::SimpleInput *> encountered_muxes;
  // should encounter no new loops, or gammas, only exit them
  rvsdg::Input * previous_state_edge = nullptr;
  while (true)
  {
    // make sure we make progress
    JLM_ASSERT(previous_state_edge != state_edge);
    if (dynamic_cast<jlm::rvsdg::RegionResult *>(state_edge))
    {
      JLM_UNREACHABLE("this should be handled by branch");
    }
    else if (rvsdg::TryGetOwnerNode<loop_node>(*state_edge))
    {
      JLM_UNREACHABLE("there should be no new loops");
    }
    auto si = util::AssertedCast<rvsdg::SimpleInput>(state_edge);
    auto sn = si->node();
    auto [branchNode, branchOperation] = rvsdg::TryGetSimpleNodeAndOp<BranchOperation>(*state_edge);
    auto [muxNode, muxOperation] = rvsdg::TryGetSimpleNodeAndOp<MuxOperation>(*state_edge);
    if (branchOperation)
    {
      // end of loop
      JLM_ASSERT(branchOperation->loop);
      state_edge = get_mem_state_user(
          util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output());
    }
    else if (muxOperation && !muxOperation->loop)
    {
      // end of gamma
      encountered_muxes.push_back(si);
      state_edge = get_mem_state_user(sn->output(0));
    }
    else if (
        std::get<1>(rvsdg::TryGetSimpleNodeAndOp<llvm::MemoryStateMergeOperation>(*state_edge))
        || std::get<1>(
            rvsdg::TryGetSimpleNodeAndOp<llvm::LambdaExitMemoryStateMergeOperation>(*state_edge)))
    {
      return { util::AssertedCast<rvsdg::SimpleInput>(state_edge), encountered_muxes };
    }
    else
    {
      JLM_UNREACHABLE("whoops");
    }
  }
}

void
OptimizeResMemState(rvsdg::Output * res_mem_state)
{
  // replace other branches with undefs, so the stateedge before the res can be killed.
  auto [merge_in, encountered_muxes] = TraceEdgeToMerge(get_mem_state_user(res_mem_state));
  JLM_ASSERT(merge_in);
  for (auto si : encountered_muxes)
  {
    auto sn = si->node();
    for (size_t i = 1; i < sn->ninputs(); ++i)
    {
      if (i != si->index())
      {
        auto state_dummy = llvm::UndefValueOperation::Create(*si->region(), si->Type());
        sn->input(i)->divert_to(state_dummy);
      }
    }
  }
}

void
OptimizeReqMemState(rvsdg::Output * req_mem_state)
{
  // there is no reason to wait for requests, if we already wait for responses, so we kill the rest
  // of this state edge
  auto [merge_in, _] = TraceEdgeToMerge(get_mem_state_user(req_mem_state));
  JLM_ASSERT(merge_in);
  auto merge_node = merge_in->node();
  std::vector<rvsdg::Output *> merge_origins;
  for (size_t i = 0; i < merge_in->node()->ninputs(); ++i)
  {
    if (i != merge_in->index())
    {
      merge_origins.push_back(merge_in->node()->input(i)->origin());
    }
  }
  auto new_merge_output = llvm::MemoryStateMergeOperation::Create(merge_origins);
  merge_node->output(0)->divert_users(new_merge_output);
  JLM_ASSERT(merge_node->IsDead());
  remove(merge_node);
}

rvsdg::SimpleNode *
ReplaceDecouple(
    const rvsdg::LambdaNode * lambda,
    rvsdg::SimpleNode * decouple_request,
    rvsdg::Output * resp)
{
  JLM_ASSERT(dynamic_cast<const llvm::CallOperation *>(&decouple_request->GetOperation()));
  auto channel = decouple_request->input(1)->origin();
  auto channel_constant = trace_constant(channel);

  auto decouple_response = find_decouple_response(lambda, channel_constant);

  // handle request
  auto addr = decouple_request->input(2)->origin();
  auto req_mem_state = decouple_request->input(decouple_request->ninputs() - 1)->origin();
  // state gate for req
  auto sg_out = state_gate_op::create(*addr, { req_mem_state });
  addr = sg_out[0];
  req_mem_state = sg_out[1];
  // redirect memstate - iostate output has already been removed by mem-sep pass
  decouple_request->output(decouple_request->noutputs() - 1)->divert_users(req_mem_state);

  // handle response
  int load_capacity = 10;
  if (rvsdg::is<const rvsdg::bittype>(decouple_response->input(2)->Type()))
  {
    auto constant = trace_constant(decouple_response->input(2)->origin());
    load_capacity = constant->Representation().to_int();
    assert(load_capacity >= 0);
  }
  auto routed_resp = route_response_rhls(decouple_request->region(), resp);
  auto dload_out = decoupled_load_op::create(*addr, *routed_resp, load_capacity);
  auto dload_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*dload_out[0]);

  auto routed_data = route_to_region_rhls(decouple_response->region(), dload_out[0]);
  decouple_response->output(0)->divert_users(routed_data);
  auto response_state_origin = decouple_response->input(decouple_response->ninputs() - 1)->origin();

  if (decouple_request->region() != decouple_response->region())
  {
    // they are in different regions, so we handle state edge at response
    auto state_dummy = llvm::UndefValueOperation::Create(
        *response_state_origin->region(),
        response_state_origin->Type());
    auto sg_resp = state_gate_op::create(*routed_data, { state_dummy });
    decouple_response->output(decouple_response->noutputs() - 1)->divert_users(sg_resp[1]);
    JLM_ASSERT(decouple_response->IsDead());
    remove(decouple_response);
    JLM_ASSERT(decouple_request->IsDead());
    remove(decouple_request);

    OptimizeResMemState(sg_resp[1]);
    OptimizeReqMemState(req_mem_state);
  }
  else
  {
    // they are in the same region, handle at request
    // remove mem state from response call
    decouple_response->output(decouple_response->noutputs() - 1)
        ->divert_users(response_state_origin);

    auto state_dummy = llvm::UndefValueOperation::Create(
        *response_state_origin->region(),
        response_state_origin->Type());
    // put state gate on load response
    auto sg_resp = state_gate_op::create(*dload_node->input(1)->origin(), { state_dummy });
    dload_node->input(1)->divert_to(sg_resp[0]);
    auto state_user = get_mem_state_user(req_mem_state);
    state_user->divert_to(sg_resp[1]);

    JLM_ASSERT(decouple_response->IsDead());
    remove(decouple_response);
    JLM_ASSERT(decouple_request->IsDead());
    remove(decouple_request);

    // these are swapped in this scenario, since we keep the one from request
    OptimizeReqMemState(response_state_origin);
    OptimizeResMemState(sg_resp[1]);
  }

  auto nn = dynamic_cast<rvsdg::node_output *>(dload_out[0])->node();
  return dynamic_cast<rvsdg::SimpleNode *>(nn);
}

void
gather_mem_nodes(
    rvsdg::Region * region,
    std::vector<rvsdg::SimpleNode *> & loadNodes,
    std::vector<rvsdg::SimpleNode *> & storeNodes,
    std::vector<rvsdg::SimpleNode *> & decoupleNodes,
    std::unordered_set<rvsdg::SimpleNode *> exclude)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        gather_mem_nodes(structnode->subregion(n), loadNodes, storeNodes, decoupleNodes, exclude);
    }
    else if (auto simplenode = dynamic_cast<rvsdg::SimpleNode *>(node))
    {
      if (exclude.find(simplenode) != exclude.end())
      {
        continue;
      }
      if (dynamic_cast<const llvm::StoreNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        storeNodes.push_back(simplenode);
      }
      else if (dynamic_cast<const llvm::LoadNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        loadNodes.push_back(simplenode);
      }
      else if (dynamic_cast<const llvm::CallOperation *>(&simplenode->GetOperation()))
      {
        // we only want to collect requests
        if (is_dec_req(simplenode))
          decoupleNodes.push_back(simplenode);
      }
    }
  }
}

/**
 * If the output is a pointer, it traces it to all memory operations it reaches.
 * Pointers read from memory is not traced, i.e., the output of load operations is not traced.
 * @param output The output to trace
 * @param loadNodes A vector containing all load nodes that are reached
 * @param storeNodes A vector containing all store nodes that are reached
 * @param decoupleNodes A vector containing all decoupled load nodes that are reached
 * @param visited A set of already visited outputs
 */
void
TracePointer(
    rvsdg::Output * output,
    std::vector<rvsdg::SimpleNode *> & loadNodes,
    std::vector<rvsdg::SimpleNode *> & storeNodes,
    std::vector<rvsdg::SimpleNode *> & decoupleNodes,
    std::unordered_set<rvsdg::Output *> & visited)
{
  if (!rvsdg::is<llvm::PointerType>(output->Type()))
  {
    // Only process pointer outputs
    return;
  }
  if (visited.count(output))
  {
    // Skip already processed outputs
    return;
  }
  visited.insert(output);
  for (auto user : *output)
  {
    if (auto si = dynamic_cast<rvsdg::SimpleInput *>(user))
    {
      auto simplenode = si->node();
      if (dynamic_cast<const llvm::StoreNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        storeNodes.push_back(simplenode);
      }
      else if (dynamic_cast<const llvm::LoadNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        loadNodes.push_back(simplenode);
      }
      else if (dynamic_cast<const llvm::CallOperation *>(&simplenode->GetOperation()))
      {
        // request
        JLM_ASSERT(is_dec_req(simplenode));
        decoupleNodes.push_back(simplenode);
      }
      else
      {
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          TracePointer(simplenode->output(i), loadNodes, storeNodes, decoupleNodes, visited);
        }
      }
    }
    else if (auto sti = dynamic_cast<rvsdg::StructuralInput *>(user))
    {
      for (auto & arg : sti->arguments)
      {
        TracePointer(&arg, loadNodes, storeNodes, decoupleNodes, visited);
      }
    }
    else if (auto r = dynamic_cast<rvsdg::RegionResult *>(user))
    {
      if (auto ber = dynamic_cast<backedge_result *>(r))
      {
        TracePointer(ber->argument(), loadNodes, storeNodes, decoupleNodes, visited);
      }
      else
      {
        TracePointer(r->output(), loadNodes, storeNodes, decoupleNodes, visited);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }
}

void
TracePointerArguments(const rvsdg::LambdaNode * lambda, port_load_store_decouple & portNodes)
{
  for (auto arg : lambda->GetFunctionArguments())
  {
    if (rvsdg::is<llvm::PointerType>(arg->Type()))
    {
      std::unordered_set<rvsdg::Output *> visited;
      portNodes.emplace_back();
      TracePointer(
          arg,
          std::get<0>(portNodes.back()),
          std::get<1>(portNodes.back()),
          std::get<2>(portNodes.back()),
          visited);
    }
  }
  for (auto cv : lambda->GetContextVars())
  {
    if (rvsdg::is<llvm::PointerType>(cv.inner->Type()) && !is_function_argument(cv))
    {
      std::unordered_set<rvsdg::Output *> visited;
      portNodes.emplace_back();
      TracePointer(
          cv.inner,
          std::get<0>(portNodes.back()),
          std::get<1>(portNodes.back()),
          std::get<2>(portNodes.back()),
          visited);
    }
  }
}

rvsdg::LambdaNode *
find_containing_lambda(rvsdg::Region * region)
{
  if (auto l = dynamic_cast<rvsdg::LambdaNode *>(region->node()))
  {
    return l;
  }
  return find_containing_lambda(region->node()->region());
}

size_t
CalcualtePortWidth(const std::tuple<
                   std::vector<rvsdg::SimpleNode *>,
                   std::vector<rvsdg::SimpleNode *>,
                   std::vector<rvsdg::SimpleNode *>> & loadStoreDecouple)
{
  int max_width = 0;
  for (auto node : std::get<0>(loadStoreDecouple))
  {
    auto loadOp = util::AssertedCast<const llvm::LoadNonVolatileOperation>(&node->GetOperation());
    auto sz = JlmSize(loadOp->GetLoadedType().get());
    max_width = sz > max_width ? sz : max_width;
  }
  for (auto node : std::get<1>(loadStoreDecouple))
  {
    auto storeOp = util::AssertedCast<const llvm::StoreNonVolatileOperation>(&node->GetOperation());
    auto sz = JlmSize(&storeOp->GetStoredType());
    max_width = sz > max_width ? sz : max_width;
  }
  for (auto decoupleRequest : std::get<2>(loadStoreDecouple))
  {
    auto lambda = find_containing_lambda(decoupleRequest->region());
    auto channel = decoupleRequest->input(1)->origin();
    auto channelConstant = trace_constant(channel);
    auto reponse = find_decouple_response(lambda, channelConstant);
    auto sz = JlmSize(reponse->output(0)->Type().get());
    max_width = sz > max_width ? sz : max_width;
  }
  JLM_ASSERT(max_width != 0);
  return max_width;
}

void
MemoryConverter(llvm::RvsdgModule & rm)
{
  //
  // Replacing memory nodes with nodes that have explicit memory ports requires arguments and
  // results to be added to the lambda. The arguments must be added before the memory nodes are
  // replaced, else the input of the new memory node will be left dangling, which is not allowed. We
  // therefore need to first replace the lambda node with a new lambda node that has the new
  // arguments and results. We can then replace the memory nodes and connect them to the new
  // arguments.
  //

  auto root = &rm.Rvsdg().GetRootRegion();
  auto lambda = dynamic_cast<rvsdg::LambdaNode *>(root->Nodes().begin().ptr());

  //
  // Converting loads and stores to explicitly use memory ports
  // This modifies the function signature so we create a new lambda node to replace the old one
  //
  const auto & op = dynamic_cast<llvm::LlvmLambdaOperation &>(lambda->GetOperation());
  auto oldFunctionType = op.type();
  std::vector<std::shared_ptr<const rvsdg::Type>> newArgumentTypes;
  for (size_t i = 0; i < oldFunctionType.NumArguments(); ++i)
  {
    newArgumentTypes.push_back(oldFunctionType.Arguments()[i]);
  }
  std::vector<std::shared_ptr<const rvsdg::Type>> newResultTypes;
  for (size_t i = 0; i < oldFunctionType.NumResults(); ++i)
  {
    newResultTypes.push_back(oldFunctionType.Results()[i]);
  }

  //
  // Get the load and store nodes and add an argument and result for each to represent the memory
  // response and request ports
  //
  port_load_store_decouple portNodes;
  TracePointerArguments(lambda, portNodes);

  std::unordered_set<rvsdg::SimpleNode *> accountedNodes;
  for (auto & portNode : portNodes)
  {
    auto portWidth = CalcualtePortWidth(portNode);
    auto responseTypePtr = get_mem_res_type(rvsdg::bittype::Create(portWidth));
    auto requestTypePtr = get_mem_req_type(rvsdg::bittype::Create(portWidth), false);
    auto requestTypePtrWrite = get_mem_req_type(rvsdg::bittype::Create(portWidth), true);
    newArgumentTypes.push_back(responseTypePtr);
    if (std::get<1>(portNode).empty())
    {
      newResultTypes.push_back(requestTypePtr);
    }
    else
    {
      newResultTypes.push_back(requestTypePtrWrite);
    }
    accountedNodes.insert(std::get<0>(portNode).begin(), std::get<0>(portNode).end());
    accountedNodes.insert(std::get<1>(portNode).begin(), std::get<1>(portNode).end());
    accountedNodes.insert(std::get<2>(portNode).begin(), std::get<2>(portNode).end());
  }
  std::vector<rvsdg::SimpleNode *> unknownLoadNodes;
  std::vector<rvsdg::SimpleNode *> unknownStoreNodes;
  std::vector<rvsdg::SimpleNode *> unknownDecoupledNodes;
  gather_mem_nodes(
      root,
      unknownLoadNodes,
      unknownStoreNodes,
      unknownDecoupledNodes,
      accountedNodes);
  if (!unknownLoadNodes.empty() || !unknownStoreNodes.empty() || !unknownDecoupledNodes.empty())
  {
    auto portWidth = CalcualtePortWidth(
        std::make_tuple(unknownLoadNodes, unknownStoreNodes, unknownDecoupledNodes));
    auto responseTypePtr = get_mem_res_type(rvsdg::bittype::Create(portWidth));
    auto requestTypePtr = get_mem_req_type(rvsdg::bittype::Create(portWidth), false);
    auto requestTypePtrWrite = get_mem_req_type(rvsdg::bittype::Create(portWidth), true);
    // Extra port for loads/stores not associated to a port yet (i.e., unknown base pointer)
    newArgumentTypes.push_back(responseTypePtr);
    if (unknownStoreNodes.empty())
    {
      newResultTypes.push_back(requestTypePtr);
    }
    else
    {
      newResultTypes.push_back(requestTypePtrWrite);
    }
  }

  //
  // Create new lambda and copy the region from the old lambda
  //
  auto newFunctionType = rvsdg::FunctionType::Create(newArgumentTypes, newResultTypes);
  auto newLambda = rvsdg::LambdaNode::Create(
      *lambda->region(),
      llvm::LlvmLambdaOperation::Create(newFunctionType, op.name(), op.linkage(), op.attributes()));

  rvsdg::SubstitutionMap smap;
  for (const auto & ctxvar : lambda->GetContextVars())
  {
    smap.insert(ctxvar.inner, newLambda->AddContextVar(*ctxvar.input->origin()).inner);
  }

  auto args = lambda->GetFunctionArguments();
  auto newArgs = newLambda->GetFunctionArguments();
  // The new function has more arguments than the old function.
  // Substitution of existing arguments is safe, but note
  // that this is not an isomorphism.
  JLM_ASSERT(args.size() <= newArgs.size());
  for (size_t i = 0; i < args.size(); ++i)
  {
    smap.insert(args[i], newArgs[i]);
  }
  lambda->subregion()->copy(newLambda->subregion(), smap, false, false);

  //
  // All memory nodes need to be replaced with new nodes that have explicit memory ports.
  // This needs to happen first and the smap needs to be updated with the new nodes,
  // before we can use the original lambda results and look them up in the updated smap.
  //

  std::vector<rvsdg::Output *> newResults;
  // The new arguments are placed directly after the original arguments so we create an index that
  // points to the first new argument
  auto newArgumentsIndex = args.size();
  for (auto & portNode : portNodes)
  {
    auto loadNodes = std::get<0>(portNode);
    auto storeNodes = std::get<1>(portNode);
    auto decoupledNodes = std::get<2>(portNode);
    newResults.push_back(ConnectRequestResponseMemPorts(
        newLambda,
        newArgumentsIndex++,
        smap,
        loadNodes,
        storeNodes,
        decoupledNodes));
  }
  if (!unknownLoadNodes.empty() || !unknownStoreNodes.empty() || !unknownDecoupledNodes.empty())
  {
    newResults.push_back(ConnectRequestResponseMemPorts(
        newLambda,
        newArgumentsIndex++,
        smap,
        unknownLoadNodes,
        unknownStoreNodes,
        unknownDecoupledNodes));
  }

  std::vector<rvsdg::Output *> originalResults;
  for (auto result : lambda->GetFunctionResults())
  {
    originalResults.push_back(smap.lookup(result->origin()));
  }
  originalResults.insert(originalResults.end(), newResults.begin(), newResults.end());
  auto newOut = newLambda->finalize(originalResults);
  auto oldExport = llvm::ComputeCallSummary(*lambda).GetRvsdgExport();
  llvm::GraphExport::Create(*newOut, oldExport ? oldExport->Name() : "");

  JLM_ASSERT(lambda->output()->nusers() == 1);
  lambda->region()->RemoveResult((*lambda->output()->begin())->index());
  remove(lambda);

  // Remove imports for decouple_ function pointers
  dne(newLambda->subregion());

  //
  // TODO
  // RemoveUnusedStates also creates a new lambda, which we have already done above.
  // It might be better to apply this functionality above such that we only create a new lambda
  // once.
  //
  RemoveUnusedStates(rm);

  // Need to get the lambda from the root since remote_unused_state replaces the lambda
  JLM_ASSERT(root->nnodes() == 1);
  newLambda = util::AssertedCast<rvsdg::LambdaNode>(root->Nodes().begin().ptr());
  auto decouple_funcs = find_function_arguments(newLambda, "decoupled");
  // make sure context vars are actually dead
  for (auto cv : decouple_funcs)
  {
    JLM_ASSERT(cv.inner->nusers() == 0);
  }
  // remove dead cvargs
  newLambda->PruneLambdaInputs();
}

rvsdg::Output *
ConnectRequestResponseMemPorts(
    const rvsdg::LambdaNode * lambda,
    size_t argumentIndex,
    rvsdg::SubstitutionMap & smap,
    const std::vector<rvsdg::SimpleNode *> & originalLoadNodes,
    const std::vector<rvsdg::SimpleNode *> & originalStoreNodes,
    const std::vector<rvsdg::SimpleNode *> & originalDecoupledNodes)
{
  //
  // We have the memory operations from the original lambda and need to lookup the corresponding
  // nodes in the new lambda
  //
  std::vector<rvsdg::SimpleNode *> loadNodes;
  std::vector<std::shared_ptr<const rvsdg::Type>> responseTypes;
  for (auto loadNode : originalLoadNodes)
  {
    JLM_ASSERT(smap.contains(*loadNode->output(0)));
    auto loadOutput = dynamic_cast<rvsdg::SimpleOutput *>(smap.lookup(loadNode->output(0)));
    loadNodes.push_back(loadOutput->node());
    auto loadOp = util::AssertedCast<const llvm::LoadNonVolatileOperation>(
        &loadOutput->node()->GetOperation());
    responseTypes.push_back(loadOp->GetLoadedType());
  }
  std::vector<rvsdg::SimpleNode *> decoupledNodes;
  for (auto decoupleRequest : originalDecoupledNodes)
  {
    JLM_ASSERT(smap.contains(*decoupleRequest->output(0)));
    auto decoupledOutput =
        dynamic_cast<rvsdg::SimpleOutput *>(smap.lookup(decoupleRequest->output(0)));
    decoupledNodes.push_back(decoupledOutput->node());
    // get load type from response output
    auto channel = decoupleRequest->input(1)->origin();
    auto channelConstant = trace_constant(channel);
    auto reponse = find_decouple_response(lambda, channelConstant);
    auto vt = std::dynamic_pointer_cast<const rvsdg::ValueType>(reponse->output(0)->Type());
    responseTypes.push_back(vt);
  }
  std::vector<rvsdg::SimpleNode *> storeNodes;
  for (auto storeNode : originalStoreNodes)
  {
    JLM_ASSERT(smap.contains(*storeNode->output(0)));
    auto storeOutput = dynamic_cast<rvsdg::SimpleOutput *>(smap.lookup(storeNode->output(0)));
    storeNodes.push_back(storeOutput->node());
    // use memory state type as response for stores
    auto vt = std::make_shared<llvm::MemoryStateType>();
    responseTypes.push_back(vt);
  }

  auto lambdaRegion = lambda->subregion();
  auto portWidth = CalcualtePortWidth(
      std::make_tuple(originalLoadNodes, originalStoreNodes, originalDecoupledNodes));
  auto responses =
      mem_resp_op::create(*lambdaRegion->argument(argumentIndex), responseTypes, portWidth);
  // The (decoupled) load nodes are replaced so the pointer to the types will become invalid
  std::vector<std::shared_ptr<const rvsdg::ValueType>> loadTypes;
  std::vector<rvsdg::Output *> loadAddresses;
  for (size_t i = 0; i < loadNodes.size(); ++i)
  {
    auto routed = route_response_rhls(loadNodes[i]->region(), responses[i]);
    // The smap contains the nodes from the original lambda so we need to use the original load node
    // when replacing the load since the smap must be updated
    auto replacement = ReplaceLoad(smap, originalLoadNodes[i], routed);
    auto address =
        route_request_rhls(lambdaRegion, replacement->output(replacement->noutputs() - 1));
    loadAddresses.push_back(address);
    std::shared_ptr<const rvsdg::ValueType> type;
    if (auto loadOperation = dynamic_cast<const LoadOperation *>(&replacement->GetOperation()))
    {
      type = loadOperation->GetLoadedType();
    }
    else if (
        auto loadOperation = dynamic_cast<const decoupled_load_op *>(&replacement->GetOperation()))
    {
      type = loadOperation->GetLoadedType();
    }
    else
    {
      JLM_UNREACHABLE("Unknown load GetOperation");
    }
    JLM_ASSERT(type);
    loadTypes.push_back(type);
  }
  for (size_t i = 0; i < decoupledNodes.size(); ++i)
  {
    auto response = responses[loadNodes.size() + i];
    auto node = decoupledNodes[i];

    // TODO: this beahvior is not completly correct - if a function returns a top-level result from
    // a decouple it fails and smap translation would be required
    auto replacement = ReplaceDecouple(lambda, node, response);
    auto addr = route_request_rhls(lambdaRegion, replacement->output(1));
    loadAddresses.push_back(addr);
    loadTypes.push_back(
        dynamic_cast<const decoupled_load_op *>(&replacement->GetOperation())->GetLoadedType());
  }
  std::vector<rvsdg::Output *> storeOperands;
  for (size_t i = 0; i < storeNodes.size(); ++i)
  {
    auto response = responses[loadNodes.size() + decoupledNodes.size() + i];
    auto routed = route_response_rhls(storeNodes[i]->region(), response);
    // The smap contains the nodes from the original lambda so we need to use the original store
    // node when replacing the store since the smap must be updated
    auto replacement = ReplaceStore(smap, originalStoreNodes[i], routed);
    auto addr = route_request_rhls(lambdaRegion, replacement->output(replacement->noutputs() - 2));
    auto data = route_request_rhls(lambdaRegion, replacement->output(replacement->noutputs() - 1));
    storeOperands.push_back(addr);
    storeOperands.push_back(data);
  }

  return mem_req_op::create(loadAddresses, loadTypes, storeOperands, lambdaRegion)[0];
}

rvsdg::SimpleNode *
ReplaceLoad(
    rvsdg::SubstitutionMap & smap,
    const rvsdg::SimpleNode * originalLoad,
    rvsdg::Output * response)
{
  // We have the load from the original lambda since it is needed to update the smap
  // We need the load in the new lambda such that we can replace it with a load node with explicit
  // memory ports
  auto replacedLoad =
      static_cast<rvsdg::SimpleOutput *>(smap.lookup(originalLoad->output(0)))->node();

  auto loadAddress = replacedLoad->input(0)->origin();
  std::vector<rvsdg::Output *> states;
  for (size_t i = 1; i < replacedLoad->ninputs(); ++i)
  {
    states.push_back(replacedLoad->input(i)->origin());
  }

  rvsdg::Node * newLoad;
  if (states.empty())
  {
    size_t load_capacity = 10;
    auto outputs = decoupled_load_op::create(*loadAddress, *response, load_capacity);
    newLoad = dynamic_cast<rvsdg::node_output *>(outputs[0])->node();
  }
  else
  {
    // TODO: switch this to a decoupled load?
    auto outputs = LoadOperation::create(*loadAddress, states, *response);
    newLoad = dynamic_cast<rvsdg::node_output *>(outputs[0])->node();
  }

  for (size_t i = 0; i < replacedLoad->noutputs(); ++i)
  {
    smap.insert(originalLoad->output(i), newLoad->output(i));
    replacedLoad->output(i)->divert_users(newLoad->output(i));
  }
  remove(replacedLoad);
  return dynamic_cast<rvsdg::SimpleNode *>(newLoad);
}

rvsdg::SimpleNode *
ReplaceStore(
    rvsdg::SubstitutionMap & smap,
    const rvsdg::SimpleNode * originalStore,
    rvsdg::Output * response)
{
  // We have the store from the original lambda since it is needed to update the smap
  // We need the store in the new lambda such that we can replace it with a store node with explicit
  // memory ports
  auto replacedStore =
      static_cast<rvsdg::SimpleOutput *>(smap.lookup(originalStore->output(0)))->node();

  auto addr = replacedStore->input(0)->origin();
  JLM_ASSERT(rvsdg::is<llvm::PointerType>(addr->Type()));
  auto data = replacedStore->input(1)->origin();
  std::vector<rvsdg::Output *> states;
  for (size_t i = 2; i < replacedStore->ninputs(); ++i)
  {
    states.push_back(replacedStore->input(i)->origin());
  }
  auto storeOuts = store_op::create(*addr, *data, states, *response);
  auto newStore = dynamic_cast<rvsdg::node_output *>(storeOuts[0])->node();
  // iterate over output states
  for (size_t i = 0; i < replacedStore->noutputs(); ++i)
  {
    // create a buffer to avoid a scenario where the reponse port is blocked because a merge waits
    // for the store
    // TODO: It might be better to have memstate merges consume individual tokens instead,, and fire
    // the output once all inputs have consumed
    const auto bo = BufferOperation::create(*storeOuts[i], 1, true)[0];
    smap.insert(originalStore->output(i), bo);
    replacedStore->output(i)->divert_users(bo);
  }
  remove(replacedStore);
  return dynamic_cast<rvsdg::SimpleNode *>(newStore);
}
}
