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

static std::pair<rvsdg::Input *, std::vector<rvsdg::Input *>>
TraceEdgeToMerge(rvsdg::Input * state_edge)
{
  std::vector<rvsdg::Input *> encountered_muxes;
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
    else if (rvsdg::TryGetOwnerNode<LoopNode>(*state_edge))
    {
      JLM_UNREACHABLE("there should be no new loops");
    }
    auto si = state_edge;
    auto sn = &rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(*si);
    auto [branchNode, branchOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<BranchOperation>(*state_edge);
    auto [muxNode, muxOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*state_edge);
    if (branchOperation)
    {
      // end of loop
      JLM_ASSERT(branchOperation->loop);
      state_edge = get_mem_state_user(
          util::assertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output());
    }
    else if (muxOperation && !muxOperation->loop)
    {
      // end of gamma
      encountered_muxes.push_back(si);
      state_edge = get_mem_state_user(sn->output(0));
    }
    else if (
        rvsdg::IsOwnerNodeOperation<llvm::MemoryStateMergeOperation>(*state_edge)
        || rvsdg::IsOwnerNodeOperation<llvm::LambdaExitMemoryStateMergeOperation>(*state_edge))
    {
      return { state_edge, encountered_muxes };
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
    auto & sn = rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(*si);
    for (size_t i = 1; i < sn.ninputs(); ++i)
    {
      if (i != si->index())
      {
        auto state_dummy = llvm::UndefValueOperation::Create(*si->region(), si->Type());
        sn.input(i)->divert_to(state_dummy);
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
  auto & merge_node = rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(*merge_in);
  std::vector<rvsdg::Output *> merge_origins;
  for (size_t i = 0; i < merge_node.ninputs(); ++i)
  {
    if (i != merge_in->index())
    {
      merge_origins.push_back(merge_node.input(i)->origin());
    }
  }
  auto new_merge_output = llvm::MemoryStateMergeOperation::Create(merge_origins);
  merge_node.output(0)->divert_users(new_merge_output);
  JLM_ASSERT(merge_node.IsDead());
  remove(&merge_node);
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
  auto sg_out = StateGateOperation::create(*addr, { req_mem_state });
  addr = sg_out[0];
  req_mem_state = sg_out[1];
  // redirect memstate - iostate output has already been removed by mem-sep pass
  decouple_request->output(decouple_request->noutputs() - 1)->divert_users(req_mem_state);

  // handle response
  int load_capacity = 10;
  if (rvsdg::is<const rvsdg::BitType>(decouple_response->input(2)->Type()))
  {
    auto constant = trace_constant(decouple_response->input(2)->origin());
    load_capacity = constant->Representation().to_int();
    assert(load_capacity >= 0);
  }
  auto routed_resp = route_response_rhls(decouple_request->region(), resp);
  auto dload_out = DecoupledLoadOperation::create(*addr, *routed_resp, load_capacity);
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
    auto sg_resp = StateGateOperation::create(*routed_data, { state_dummy });
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
    auto sg_resp = StateGateOperation::create(*dload_node->input(1)->origin(), { state_dummy });
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

  auto nn = dynamic_cast<rvsdg::NodeOutput *>(dload_out[0])->node();
  return dynamic_cast<rvsdg::SimpleNode *>(nn);
}

void
gather_mem_nodes(
    rvsdg::Region * region,
    std::vector<rvsdg::Node *> & loadNodes,
    std::vector<rvsdg::Node *> & storeNodes,
    std::vector<rvsdg::Node *> & decoupleNodes,
    std::unordered_set<rvsdg::Node *> exclude)
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
 * @param visited A set of already visited outputs
 * @param tracedPointerNodes All nodes that are reached
 */
static void
TracePointer(
    rvsdg::Output * output,
    std::unordered_set<rvsdg::Output *> & visited,
    TracedPointerNodes & tracedPointerNodes)
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
  for (auto & user : output->Users())
  {
    if (auto simplenode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user))
    {
      if (dynamic_cast<const llvm::StoreNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        tracedPointerNodes.storeNodes.push_back(simplenode);
      }
      else if (dynamic_cast<const llvm::LoadNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        tracedPointerNodes.loadNodes.push_back(simplenode);
      }
      else if (dynamic_cast<const llvm::CallOperation *>(&simplenode->GetOperation()))
      {
        // request
        JLM_ASSERT(is_dec_req(simplenode));
        tracedPointerNodes.decoupleNodes.push_back(simplenode);
      }
      else
      {
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          TracePointer(simplenode->output(i), visited, tracedPointerNodes);
        }
      }
    }
    else if (auto sti = dynamic_cast<rvsdg::StructuralInput *>(&user))
    {
      for (auto & arg : sti->arguments)
      {
        TracePointer(&arg, visited, tracedPointerNodes);
      }
    }
    else if (auto r = dynamic_cast<rvsdg::RegionResult *>(&user))
    {
      if (auto ber = dynamic_cast<BackEdgeResult *>(r))
      {
        TracePointer(ber->argument(), visited, tracedPointerNodes);
      }
      else
      {
        TracePointer(r->output(), visited, tracedPointerNodes);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }
}

std::vector<TracedPointerNodes>
TracePointerArguments(const rvsdg::LambdaNode * lambda)
{
  std::vector<TracedPointerNodes> tracedPointerNodes;
  for (const auto argument : lambda->GetFunctionArguments())
  {
    if (rvsdg::is<llvm::PointerType>(argument->Type()))
    {
      std::unordered_set<rvsdg::Output *> visited;
      tracedPointerNodes.emplace_back();
      TracePointer(argument, visited, tracedPointerNodes.back());
    }
  }

  for (auto cv : lambda->GetContextVars())
  {
    if (rvsdg::is<llvm::PointerType>(cv.inner->Type()) && !is_function_argument(cv))
    {
      std::unordered_set<rvsdg::Output *> visited;
      tracedPointerNodes.emplace_back();
      TracePointer(cv.inner, visited, tracedPointerNodes.back());
    }
  }

  return tracedPointerNodes;
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

static size_t
CalculatePortWidth(const TracedPointerNodes & tracedPointerNodes)
{
  int max_width = 0;
  for (auto node : tracedPointerNodes.loadNodes)
  {
    auto loadOp = util::assertedCast<const llvm::LoadNonVolatileOperation>(&node->GetOperation());
    auto sz = JlmSize(loadOp->GetLoadedType().get());
    max_width = sz > max_width ? sz : max_width;
  }
  for (auto node : tracedPointerNodes.storeNodes)
  {
    auto storeOp = util::assertedCast<const llvm::StoreNonVolatileOperation>(&node->GetOperation());
    auto sz = JlmSize(&storeOp->GetStoredType());
    max_width = sz > max_width ? sz : max_width;
  }
  for (auto decoupleRequest : tracedPointerNodes.decoupleNodes)
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

static rvsdg::SimpleNode *
ReplaceLoad(
    rvsdg::SubstitutionMap & smap,
    const rvsdg::Node * originalLoad,
    rvsdg::Output * response)
{
  // We have the load from the original lambda since it is needed to update the smap
  // We need the load in the new lambda such that we can replace it with a load node with explicit
  // memory ports
  auto replacedLoad =
      &rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(smap.lookup(*originalLoad->output(0)));

  auto loadAddress = replacedLoad->input(0)->origin();
  std::vector<rvsdg::Output *> states;
  for (size_t i = 1; i < replacedLoad->ninputs(); ++i)
  {
    states.push_back(replacedLoad->input(i)->origin());
  }

  rvsdg::Node * newLoad = nullptr;
  if (states.empty())
  {
    size_t load_capacity = 10;
    auto outputs = DecoupledLoadOperation::create(*loadAddress, *response, load_capacity);
    newLoad = dynamic_cast<rvsdg::NodeOutput *>(outputs[0])->node();
  }
  else
  {
    // TODO: switch this to a decoupled load?
    auto outputs = LoadOperation::create(*loadAddress, states, *response);
    newLoad = dynamic_cast<rvsdg::NodeOutput *>(outputs[0])->node();
  }

  for (size_t i = 0; i < replacedLoad->noutputs(); ++i)
  {
    smap.insert(originalLoad->output(i), newLoad->output(i));
    replacedLoad->output(i)->divert_users(newLoad->output(i));
  }
  remove(replacedLoad);
  return dynamic_cast<rvsdg::SimpleNode *>(newLoad);
}

static rvsdg::SimpleNode *
ReplaceStore(
    rvsdg::SubstitutionMap & smap,
    const rvsdg::Node * originalStore,
    rvsdg::Output * response)
{
  // We have the store from the original lambda since it is needed to update the smap
  // We need the store in the new lambda such that we can replace it with a store node with explicit
  // memory ports
  auto replacedStore =
      &rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(smap.lookup(*originalStore->output(0)));

  auto addr = replacedStore->input(0)->origin();
  JLM_ASSERT(rvsdg::is<llvm::PointerType>(addr->Type()));
  auto data = replacedStore->input(1)->origin();
  std::vector<rvsdg::Output *> states;
  for (size_t i = 2; i < replacedStore->ninputs(); ++i)
  {
    states.push_back(replacedStore->input(i)->origin());
  }
  auto storeOuts = StoreOperation::create(*addr, *data, states, *response);
  auto newStore = dynamic_cast<rvsdg::NodeOutput *>(storeOuts[0])->node();
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

static rvsdg::Output *
ConnectRequestResponseMemPorts(
    const rvsdg::LambdaNode * lambda,
    size_t argumentIndex,
    rvsdg::SubstitutionMap & smap,
    const std::vector<rvsdg::Node *> & originalLoadNodes,
    const std::vector<rvsdg::Node *> & originalStoreNodes,
    const std::vector<rvsdg::Node *> & originalDecoupledNodes)
{
  //
  // We have the memory operations from the original lambda and need to lookup the corresponding
  // nodes in the new lambda
  //
  std::vector<rvsdg::SimpleNode *> loadNodes;
  std::vector<std::shared_ptr<const rvsdg::Type>> responseTypes;
  for (auto loadNode : originalLoadNodes)
  {
    auto oldLoadedValue = loadNode->output(0);
    JLM_ASSERT(smap.contains(*oldLoadedValue));
    auto & newLoadNode = rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(smap.lookup(*oldLoadedValue));
    loadNodes.push_back(&newLoadNode);
    auto loadOp =
        util::assertedCast<const llvm::LoadNonVolatileOperation>(&newLoadNode.GetOperation());
    responseTypes.push_back(loadOp->GetLoadedType());
  }
  std::vector<rvsdg::SimpleNode *> decoupledNodes;
  for (auto decoupleRequest : originalDecoupledNodes)
  {
    auto oldOutput = decoupleRequest->output(0);
    JLM_ASSERT(smap.contains(*oldOutput));
    auto & decoupledRequestNode =
        rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(smap.lookup(*oldOutput));
    decoupledNodes.push_back(&decoupledRequestNode);
    // get load type from response output
    auto channel = decoupleRequest->input(1)->origin();
    auto channelConstant = trace_constant(channel);
    auto reponse = find_decouple_response(lambda, channelConstant);
    auto vt = reponse->output(0)->Type();
    responseTypes.push_back(vt);
  }
  std::vector<rvsdg::SimpleNode *> storeNodes;
  for (auto storeNode : originalStoreNodes)
  {
    auto oldOutput = storeNode->output(0);
    JLM_ASSERT(smap.contains(*oldOutput));
    auto & newStoreNode = rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(smap.lookup(*oldOutput));
    storeNodes.push_back(&newStoreNode);
    // use memory state type as response for stores
    auto vt = std::make_shared<llvm::MemoryStateType>();
    responseTypes.push_back(vt);
  }

  auto lambdaRegion = lambda->subregion();
  auto portWidth =
      CalculatePortWidth({ originalLoadNodes, originalStoreNodes, originalDecoupledNodes });
  auto responses = MemoryResponseOperation::create(
      *lambdaRegion->argument(argumentIndex),
      responseTypes,
      portWidth);
  // The (decoupled) load nodes are replaced so the pointer to the types will become invalid
  std::vector<std::shared_ptr<const rvsdg::Type>> loadTypes;
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
    std::shared_ptr<const rvsdg::Type> type;
    if (auto loadOperation = dynamic_cast<const LoadOperation *>(&replacement->GetOperation()))
    {
      type = loadOperation->GetLoadedType();
    }
    else if (
        auto loadOperation =
            dynamic_cast<const DecoupledLoadOperation *>(&replacement->GetOperation()))
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
    loadTypes.push_back(dynamic_cast<const DecoupledLoadOperation *>(&replacement->GetOperation())
                            ->GetLoadedType());
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

  return MemoryRequestOperation::create(loadAddresses, loadTypes, storeOperands, lambdaRegion)[0];
}

static void
ConvertMemory(rvsdg::RvsdgModule & rvsdgModule)
{
  //
  // Replacing memory nodes with nodes that have explicit memory ports requires arguments and
  // results to be added to the lambda. The arguments must be added before the memory nodes are
  // replaced, else the input of the new memory node will be left dangling, which is not allowed. We
  // therefore need to first replace the lambda node with a new lambda node that has the new
  // arguments and results. We can then replace the memory nodes and connect them to the new
  // arguments.
  //

  const auto & graph = rvsdgModule.Rvsdg();
  const auto rootRegion = &graph.GetRootRegion();
  if (rootRegion->numNodes() != 1)
  {
    throw std::logic_error("Root should have only one node now");
  }

  const auto lambda = dynamic_cast<rvsdg::LambdaNode *>(rootRegion->Nodes().begin().ptr());
  if (!lambda)
  {
    throw std::logic_error("Node needs to be a lambda");
  }

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
  auto tracedPointerNodesVector = TracePointerArguments(lambda);

  std::unordered_set<rvsdg::Node *> accountedNodes;
  for (auto & portNode : tracedPointerNodesVector)
  {
    auto portWidth = CalculatePortWidth(portNode);
    auto responseTypePtr = get_mem_res_type(rvsdg::BitType::Create(portWidth));
    auto requestTypePtr = get_mem_req_type(rvsdg::BitType::Create(portWidth), false);
    auto requestTypePtrWrite = get_mem_req_type(rvsdg::BitType::Create(portWidth), true);
    newArgumentTypes.push_back(responseTypePtr);
    if (portNode.storeNodes.empty())
    {
      newResultTypes.push_back(requestTypePtr);
    }
    else
    {
      newResultTypes.push_back(requestTypePtrWrite);
    }
    accountedNodes.insert(portNode.loadNodes.begin(), portNode.loadNodes.end());
    accountedNodes.insert(portNode.storeNodes.begin(), portNode.storeNodes.end());
    accountedNodes.insert(portNode.decoupleNodes.begin(), portNode.decoupleNodes.end());
  }
  std::vector<rvsdg::Node *> unknownLoadNodes;
  std::vector<rvsdg::Node *> unknownStoreNodes;
  std::vector<rvsdg::Node *> unknownDecoupledNodes;
  gather_mem_nodes(
      rootRegion,
      unknownLoadNodes,
      unknownStoreNodes,
      unknownDecoupledNodes,
      accountedNodes);
  if (!unknownLoadNodes.empty() || !unknownStoreNodes.empty() || !unknownDecoupledNodes.empty())
  {
    auto portWidth =
        CalculatePortWidth({ unknownLoadNodes, unknownStoreNodes, unknownDecoupledNodes });
    auto responseTypePtr = get_mem_res_type(rvsdg::BitType::Create(portWidth));
    auto requestTypePtr = get_mem_req_type(rvsdg::BitType::Create(portWidth), false);
    auto requestTypePtrWrite = get_mem_req_type(rvsdg::BitType::Create(portWidth), true);
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
  lambda->subregion()->copy(newLambda->subregion(), smap);

  //
  // All memory nodes need to be replaced with new nodes that have explicit memory ports.
  // This needs to happen first and the smap needs to be updated with the new nodes,
  // before we can use the original lambda results and look them up in the updated smap.
  //

  std::vector<rvsdg::Output *> newResults;
  // The new arguments are placed directly after the original arguments so we create an index that
  // points to the first new argument
  auto newArgumentsIndex = args.size();
  for (auto & portNode : tracedPointerNodesVector)
  {
    newResults.push_back(ConnectRequestResponseMemPorts(
        newLambda,
        newArgumentsIndex++,
        smap,
        portNode.loadNodes,
        portNode.storeNodes,
        portNode.decoupleNodes));
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
    originalResults.push_back(&smap.lookup(*result->origin()));
  }
  originalResults.insert(originalResults.end(), newResults.begin(), newResults.end());
  auto newOut = newLambda->finalize(originalResults);
  auto oldExport = llvm::ComputeCallSummary(*lambda).GetRvsdgExport();
  rvsdg::GraphExport::Create(*newOut, oldExport ? oldExport->Name() : "");

  JLM_ASSERT(lambda->output()->nusers() == 1);
  lambda->region()->RemoveResults({ (*lambda->output()->Users().begin()).index() });
  remove(lambda);

  // Remove imports for decouple_ function pointers
  RhlsDeadNodeElimination dne;
  util::StatisticsCollector statisticsCollector;
  dne.Run(*newLambda->subregion(), statisticsCollector);

  //
  // TODO
  // RemoveUnusedStates also creates a new lambda, which we have already done above.
  // It might be better to apply this functionality above such that we only create a new lambda
  // once.
  //
  UnusedStateRemoval::CreateAndRun(rvsdgModule, statisticsCollector);

  // Need to get the lambda from the root since remote_unused_state replaces the lambda
  JLM_ASSERT(rootRegion->numNodes() == 1);
  newLambda = util::assertedCast<rvsdg::LambdaNode>(rootRegion->Nodes().begin().ptr());
  auto decouple_funcs = find_function_arguments(newLambda, "decoupled");
  // make sure context vars are actually dead
  for (auto cv : decouple_funcs)
  {
    JLM_ASSERT(cv.inner->nusers() == 0);
  }
  // remove dead cvargs
  newLambda->PruneLambdaInputs();
}

MemoryConverter::~MemoryConverter() noexcept = default;

MemoryConverter::MemoryConverter()
    : Transformation("MemoryConverter")
{}

void
MemoryConverter::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  ConvertMemory(rvsdgModule);
}

}
