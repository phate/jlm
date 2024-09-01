/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * and Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

jlm::rvsdg::output *
jlm::hls::route_response(jlm::rvsdg::region * target, jlm::rvsdg::output * response)
{
  if (response->region() == target)
  {
    return response;
  }
  else
  {
    auto parent_response = route_response(target->node()->region(), response);
    auto ln = dynamic_cast<jlm::hls::loop_node *>(target->node());
    JLM_ASSERT(ln);
    auto input = jlm::rvsdg::structural_input::create(ln, parent_response, parent_response->Type());
    auto & argument = EntryArgument::Create(*target, *input, response->Type());
    return &argument;
  }
}

jlm::rvsdg::output *
jlm::hls::route_request(jlm::rvsdg::region * target, jlm::rvsdg::output * request)
{
  if (request->region() == target)
  {
    return request;
  }
  else
  {
    auto ln = dynamic_cast<jlm::hls::loop_node *>(request->region()->node());
    JLM_ASSERT(ln);
    auto output = jlm::rvsdg::structural_output::create(ln, request->Type());
    ExitResult::Create(*request, *output);
    return route_request(target, output);
  }
}

jlm::rvsdg::simple_node *
replace_load(jlm::rvsdg::simple_node * orig, jlm::rvsdg::output * resp)
{
  auto addr = orig->input(0)->origin();
  std::vector<jlm::rvsdg::output *> states;
  for (size_t i = 1; i < orig->ninputs(); ++i)
  {
    states.push_back(orig->input(i)->origin());
  }
  jlm::rvsdg::node * nn;
  if (states.empty())
  {
    auto outputs = jlm::hls::decoupled_load_op::create(*addr, *resp);
    nn = dynamic_cast<jlm::rvsdg::node_output *>(outputs[0])->node();
  }
  else
  {
    auto outputs = jlm::hls::load_op::create(*addr, states, *resp);
    nn = dynamic_cast<jlm::rvsdg::node_output *>(outputs[0])->node();
  }
  for (size_t i = 0; i < orig->noutputs(); ++i)
  {
    orig->output(i)->divert_users(nn->output(i));
  }
  remove(orig);
  return dynamic_cast<jlm::rvsdg::simple_node *>(nn);
}

const jlm::rvsdg::bitconstant_op *
trace_channel(const jlm::rvsdg::output * dst)
{
  if (auto arg = dynamic_cast<const jlm::rvsdg::argument *>(dst))
  {
    return trace_channel(arg->input()->origin());
  }
  else if (auto so = dynamic_cast<const jlm::rvsdg::simple_output *>(dst))
  {
    if (auto co = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&so->node()->operation()))
    {
      return co;
    }
    for (size_t i = 0; i < so->node()->ninputs(); ++i)
    {
      // TODO: fix, this is a hack
      if (so->node()->input(i)->type() == dst->type())
      {
        return trace_channel(so->node()->input(i)->origin());
      }
    }
  }
  JLM_UNREACHABLE("Channel not found");
}

const jlm::rvsdg::output *
trace_call(const jlm::rvsdg::output * output);

const jlm::rvsdg::output *
trace_call(const jlm::rvsdg::input * input)
{
  // version of trace call for rhls
  return trace_call(input->origin());
}

const jlm::rvsdg::output *
trace_call(const jlm::rvsdg::output * output)
{
  // version of trace call for rhls
  if (auto argument = dynamic_cast<const jlm::rvsdg::argument *>(output))
  {
    auto graph = output->region()->graph();
    if (argument->region() == graph->root())
    {
      return argument;
    }
    else if (dynamic_cast<const jlm::hls::backedge_argument *>(argument))
    {
      // don't follow backedges to avoid cycles
      return nullptr;
    }
    return trace_call(argument->input());
  }
  else if (auto so = dynamic_cast<const jlm::rvsdg::structural_output *>(output))
  {
    for (auto & r : so->results)
    {
      if (auto result = trace_call(&r))
      {
        return result;
      }
    }
  }
  else if (auto so = dynamic_cast<const jlm::rvsdg::simple_output *>(output))
  {
    for (size_t i = 0; i < so->node()->ninputs(); ++i)
    {
      auto ip = so->node()->input(i);
      if (ip->type() == output->type())
      {
        if (auto result = trace_call(ip))
        {
          return result;
        }
      }
    }
  }
  else
  {
    JLM_UNREACHABLE("");
  }
  return nullptr;
}

std::string
get_impport_function_name(jlm::rvsdg::input * input)
{
  auto traced = trace_call(input);
  JLM_ASSERT(traced);
  auto arg = jlm::util::AssertedCast<const jlm::llvm::GraphImport>(traced);
  return arg->Name();
}

// trace function ptr to its call
void
trace_function_calls(
    jlm::rvsdg::output * op,
    std::vector<jlm::rvsdg::simple_node *> & calls,
    std::unordered_set<jlm::rvsdg::output *> & visited)
{
  if (visited.count(op))
  {
    // skip already processed outputs
    return;
  }
  visited.insert(op);
  for (auto user : *op)
  {
    if (auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user))
    {
      auto simplenode = si->node();
      if (dynamic_cast<const jlm::llvm::CallOperation *>(&simplenode->operation()))
      {
        // TODO: verify this is the right type of function call
        calls.push_back(simplenode);
      }
      else
      {
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          trace_function_calls(simplenode->output(i), calls, visited);
        }
      }
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::structural_input *>(user))
    {
      for (auto & arg : sti->arguments)
      {
        trace_function_calls(&arg, calls, visited);
      }
    }
    else if (auto r = dynamic_cast<jlm::rvsdg::result *>(user))
    {
      if (auto ber = dynamic_cast<jlm::hls::backedge_result *>(r))
      {
        trace_function_calls(ber->argument(), calls, visited);
      }
      else
      {
        trace_function_calls(r->output(), calls, visited);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }
}

jlm::rvsdg::simple_node *
find_decouple_response(
    const jlm::llvm::lambda::node * lambda,
    const jlm::rvsdg::bitconstant_op * request_constant)
{
  jlm::rvsdg::argument * response_function = nullptr;
  for (size_t i = 0; i < lambda->ncvarguments(); ++i)
  {
    auto ip = lambda->cvargument(i)->input();
    if (dynamic_cast<const jlm::llvm::PointerType *>(ip)
        && get_impport_function_name(ip) == "decouple_response")
    {
      response_function = lambda->cvargument(i);
    }
  }
  JLM_ASSERT(response_function == nullptr);
  std::vector<jlm::rvsdg::simple_node *> reponse_calls;
  std::unordered_set<jlm::rvsdg::output *> visited;
  trace_function_calls(response_function, reponse_calls, visited);
  JLM_ASSERT(!reponse_calls.empty());
  for (auto & rc : reponse_calls)
  {
    auto response_constant = trace_channel(rc->input(1)->origin());
    if (*response_constant == *request_constant)
    {
      return rc;
    }
  }
  JLM_UNREACHABLE("No response found");
}

jlm::rvsdg::simple_node *
replace_decouple(
    const jlm::llvm::lambda::node * lambda,
    jlm::rvsdg::simple_node * decouple_request,
    jlm::rvsdg::output * resp)
{
  JLM_ASSERT(dynamic_cast<const jlm::llvm::CallOperation *>(&decouple_request->operation()));
  auto channel = decouple_request->input(1)->origin();
  auto channel_constant = trace_channel(channel);

  auto decouple_response = find_decouple_response(lambda, channel_constant);

  auto addr = decouple_request->input(2)->origin();
  // connect out states to in states
  // TODO: use stategate?
  // one function pointer + 2 params
  for (size_t i = 3; i < decouple_request->ninputs(); ++i)
  {
    decouple_request->output(i - 3)->divert_users(decouple_request->input(i)->origin());
  }
  for (size_t i = 2; i < decouple_response->ninputs(); ++i)
  {
    decouple_response->output(i - 2 + 1)->divert_users(decouple_response->input(i)->origin());
  }
  // create this outside loop - need to tunnel outward from request and inward to response
  auto routed_addr = jlm::hls::route_request(lambda->subregion(), addr);
  // response is not routed inward for this case
  auto dload_out = jlm::hls::decoupled_load_op::create(*routed_addr, *resp);
  // use a buffer here to make ready logic for response easy and consistent
  auto buf = jlm::hls::buffer_op::create(*dload_out[0], 2, true)[0];

  auto routed_data = jlm::hls::route_response(decouple_response->region(), buf);
  decouple_response->output(0)->divert_users(routed_data);
  // TODO: handle state edges?
  remove(decouple_request);
  remove(decouple_response);

  auto nn = dynamic_cast<jlm::rvsdg::node_output *>(dload_out[0])->node();
  return dynamic_cast<jlm::rvsdg::simple_node *>(nn);
}

jlm::rvsdg::simple_node *
replace_store(jlm::rvsdg::simple_node * orig)
{
  auto addr = orig->input(0)->origin();
  auto data = orig->input(1)->origin();
  std::vector<jlm::rvsdg::output *> states;
  for (size_t i = 2; i < orig->ninputs(); ++i)
  {
    states.push_back(orig->input(i)->origin());
  }
  auto outputs = jlm::hls::store_op::create(*addr, *data, states);
  auto nn = dynamic_cast<jlm::rvsdg::node_output *>(outputs[0])->node();
  for (size_t i = 0; i < orig->noutputs(); ++i)
  {
    orig->output(i)->divert_users(nn->output(i));
  }
  remove(orig);
  return dynamic_cast<jlm::rvsdg::simple_node *>(nn);
}

void
gather_mem_nodes(
    jlm::rvsdg::region * region,
    std::vector<jlm::rvsdg::simple_node *> & load_nodes,
    std::vector<jlm::rvsdg::simple_node *> & store_nodes,
    std::vector<jlm::rvsdg::simple_node *> & decouple_nodes,
    std::unordered_set<jlm::rvsdg::simple_node *> exclude)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        gather_mem_nodes(
            structnode->subregion(n),
            load_nodes,
            store_nodes,
            decouple_nodes,
            exclude);
    }
    else if (auto simplenode = dynamic_cast<jlm::rvsdg::simple_node *>(node))
    {
      if (exclude.find(simplenode) != exclude.end())
      {
        continue;
      }
      if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&simplenode->operation()))
      {
        store_nodes.push_back(simplenode);
      }
      else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(&simplenode->operation()))
      {
        load_nodes.push_back(simplenode);
      }
      else if (dynamic_cast<const jlm::llvm::CallOperation *>(&simplenode->operation()))
      {
        // TODO: verify this is the right type of function call
        decouple_nodes.push_back(simplenode);
      }
    }
  }
}

// trace each input pointer to loads and stores
void
trace_pointer_argument(
    jlm::rvsdg::output * op,
    std::vector<jlm::rvsdg::simple_node *> & load_nodes,
    std::vector<jlm::rvsdg::simple_node *> & store_nodes,
    std::vector<jlm::rvsdg::simple_node *> & decouple_nodes,
    std::unordered_set<jlm::rvsdg::output *> & visited)
{
  if (!dynamic_cast<const jlm::llvm::PointerType *>(&op->type()))
  {
    // only process pointer outputs
    return;
  }
  if (visited.count(op))
  {
    // skip already processed outputs
    return;
  }
  visited.insert(op);
  for (auto user : *op)
  {
    if (auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user))
    {
      auto simplenode = si->node();
      if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&simplenode->operation()))
      {
        store_nodes.push_back(simplenode);
      }
      else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(&simplenode->operation()))
      {
        load_nodes.push_back(simplenode);
      }
      else if (dynamic_cast<const jlm::llvm::CallOperation *>(&simplenode->operation()))
      {
        // TODO: verify this is the right type of function call
        decouple_nodes.push_back(simplenode);
      }
      else
      {
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          trace_pointer_argument(
              simplenode->output(i),
              load_nodes,
              store_nodes,
              decouple_nodes,
              visited);
        }
      }
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::structural_input *>(user))
    {
      for (auto & arg : sti->arguments)
      {
        trace_pointer_argument(&arg, load_nodes, store_nodes, decouple_nodes, visited);
      }
    }
    else if (auto r = dynamic_cast<jlm::rvsdg::result *>(user))
    {
      if (auto ber = dynamic_cast<jlm::hls::backedge_result *>(r))
      {
        trace_pointer_argument(ber->argument(), load_nodes, store_nodes, decouple_nodes, visited);
      }
      else
      {
        trace_pointer_argument(r->output(), load_nodes, store_nodes, decouple_nodes, visited);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }
}

/**
 * Decoupled loads are user specified and encoded as function calls that need special treatment.
 * This function traces the output to all nodes and checks if it is the first argument to a call
 * operation.
 */
bool
IsDecoupledFunctionPointer(
    jlm::rvsdg::output * output,
    std::unordered_set<jlm::rvsdg::output *> & visited)
{
  if (!dynamic_cast<const jlm::llvm::PointerType *>(&output->type()))
  {
    // Only process pointer outputs
    return false;
  }
  if (visited.count(output))
  {
    // Skip already processed outputs
    return false;
  }
  visited.insert(output);

  bool isDecoupled = false;
  for (auto user : *output)
  {
    if (auto simpleInput = dynamic_cast<jlm::rvsdg::simple_input *>(user))
    {
      auto simpleNode = simpleInput->node();
      if (dynamic_cast<const jlm::llvm::CallOperation *>(&simpleNode->operation()))
      {
        if (simpleNode->input(0)->origin() == output)
        {
          // The output is the first argument to a call operation so this is a function pointer.
          // TODO
          // Currently, we only support decoupled load functions, so all other functions should
          // have bene inlined by now. Maybe a check that this is truly a decoupled load function
          // should be added.
          return true;
        }
      }
      else
      {
        for (size_t i = 0; i < simpleNode->noutputs(); ++i)
        {
          isDecoupled |= IsDecoupledFunctionPointer(simpleNode->output(i), visited);
        }
      }
    }
    else if (auto structuralInput = dynamic_cast<jlm::rvsdg::structural_input *>(user))
    {
      for (auto & arg : structuralInput->arguments)
      {
        isDecoupled |= IsDecoupledFunctionPointer(&arg, visited);
      }
    }
    else if (auto result = dynamic_cast<jlm::rvsdg::result *>(user))
    {
      if (auto backedgeResult = dynamic_cast<jlm::hls::backedge_result *>(result))
      {
        isDecoupled |= IsDecoupledFunctionPointer(backedgeResult->argument(), visited);
      }
      else
      {
        isDecoupled |= IsDecoupledFunctionPointer(result->output(), visited);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }

  return isDecoupled;
}

void
jlm::hls::trace_pointer_arguments(
    const jlm::llvm::lambda::node * ln,
    port_load_store_decouple & port_nodes)
{
  for (size_t i = 0; i < ln->subregion()->narguments(); ++i)
  {
    auto arg = ln->subregion()->argument(i);
    if (dynamic_cast<const jlm::llvm::PointerType *>(&arg->type()))
    {
      // Decoupled loads are user specified and encoded as function calls that need special
      // treatment
      std::unordered_set<jlm::rvsdg::output *> visited;
      if (IsDecoupledFunctionPointer((jlm::rvsdg::output *)(arg), visited))
      {
        // We are only interested in the address of the load and not the function pointer itself
        continue;
      }
      visited.clear();
      port_nodes.emplace_back();
      trace_pointer_argument(
          arg,
          std::get<0>(port_nodes.back()),
          std::get<1>(port_nodes.back()),
          std::get<2>(port_nodes.back()),
          visited);
    }
  }
}

void
jlm::hls::MemoryConverter(jlm::llvm::RvsdgModule & rm)
{
  //
  // Replacing memory nodes with nodes that have explicit memory ports requires arguments and
  // results to be added to the lambda. The arguments must be added before the memory nodes are
  // replaced, else the input of the new memory node will be left dangling, which is not allowed. We
  // therefore need to first replace the lambda node with a new lambda node that has the new
  // arguments and results. We can then replace the memory nodes and connect them to the new
  // arguments.
  //

  auto root = rm.Rvsdg().root();
  auto lambda = dynamic_cast<jlm::llvm::lambda::node *>(root->nodes.begin().ptr());

  //
  // Converting loads and stores to explicitly use memory ports
  // This modifies the function signature so we create a new lambda node to replace the old one
  //
  auto oldFunctionType = lambda->type();
  std::vector<std::shared_ptr<const jlm::rvsdg::type>> newArgumentTypes;
  for (size_t i = 0; i < oldFunctionType.NumArguments(); ++i)
  {
    newArgumentTypes.push_back(oldFunctionType.Arguments()[i]);
  }
  std::vector<std::shared_ptr<const jlm::rvsdg::type>> newResultTypes;
  for (size_t i = 0; i < oldFunctionType.NumResults(); ++i)
  {
    newResultTypes.push_back(oldFunctionType.Results()[i]);
  }

  //
  // Get the load and store nodes and add an argument and result for each to represent the memory
  // response and request ports
  //
  port_load_store_decouple portNodes;
  trace_pointer_arguments(lambda, portNodes);

  auto responseTypePtr = get_mem_res_type(jlm::rvsdg::bittype::Create(64));
  auto requestTypePtr = get_mem_req_type(jlm::rvsdg::bittype::Create(64), false);
  auto requestTypePtrWrite = get_mem_req_type(jlm::rvsdg::bittype::Create(64), true);

  std::unordered_set<jlm::rvsdg::simple_node *> accountedNodes;
  for (auto & portNode : portNodes)
  {
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
  std::vector<jlm::rvsdg::simple_node *> unknownLoadNodes;
  std::vector<jlm::rvsdg::simple_node *> unknownStoreNodes;
  std::vector<jlm::rvsdg::simple_node *> unknownDecoupledNodes;
  gather_mem_nodes(
      root,
      unknownLoadNodes,
      unknownStoreNodes,
      unknownDecoupledNodes,
      accountedNodes);
  if (!unknownLoadNodes.empty() || !unknownStoreNodes.empty() || !unknownDecoupledNodes.empty())
  {
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
  auto newFunctionType = jlm::llvm::FunctionType::Create(newArgumentTypes, newResultTypes);
  auto newLambda = jlm::llvm::lambda::node::create(
      lambda->region(),
      newFunctionType,
      lambda->name(),
      lambda->linkage(),
      lambda->attributes());

  jlm::rvsdg::substitution_map smap;
  for (size_t i = 0; i < lambda->ncvarguments(); ++i)
  {
    smap.insert(
        lambda->cvargument(i),
        newLambda->add_ctxvar(lambda->cvargument(i)->input()->origin()));
  }

  for (size_t i = 0; i < lambda->nfctarguments(); ++i)
  {
    smap.insert(lambda->fctargument(i), newLambda->fctargument(i));
  }
  lambda->subregion()->copy(newLambda->subregion(), smap, false, false);

  //
  // All memory nodes need to be replaced with new nodes that have explicit memory ports.
  // This needs to happen first and the smap needs to be updated with the new nodes,
  // before we can use the original lambda results and look them up in the updated smap.
  //

  std::vector<jlm::rvsdg::output *> newResults;
  // The new arguments are placed directly after the original arguments so we create an index that
  // points to the first new argument
  auto newArgumentsIndex = lambda->nfctarguments();
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

  std::vector<jlm::rvsdg::output *> originalResults;
  for (auto & result : lambda->fctresults())
  {
    originalResults.push_back(smap.lookup(result.origin()));
  }
  originalResults.insert(originalResults.end(), newResults.begin(), newResults.end());
  auto newOut = newLambda->finalize(originalResults);
  auto oldExport = lambda->ComputeCallSummary()->GetRvsdgExport();
  llvm::GraphExport::Create(*newOut, oldExport ? oldExport->Name() : "");

  JLM_ASSERT(lambda->output()->nusers() == 1);
  lambda->region()->RemoveResult((*lambda->output()->begin())->index());
  remove(lambda);

  // Remove imports for decouple_ function pointers
  dne(newLambda->subregion());

  //
  // TODO
  // Remove unused state also creates a new lambda, which we have already done above.
  // It would be better to apply this functionality above such that we only create a new lambda
  // once.
  //
  remove_unused_state(root);
  // Go through in reverse since we are removing things
  for (int i = newLambda->ncvarguments() - 1; i >= 0; --i)
  {
    auto cvarg = newLambda->cvargument(i);
    if (dynamic_cast<const jlm::llvm::PointerType *>(&cvarg->type()))
    {
      // The only functions at this time is decoupled loads that are encoded as functions by the
      // user
      auto visited = std::unordered_set<jlm::rvsdg::output *>();
      if (IsDecoupledFunctionPointer(cvarg->input()->origin(), visited))
      {
        JLM_ASSERT(cvarg->nusers() == 0);
        auto cvip = cvarg->input();
        newLambda->subregion()->RemoveArgument(cvarg->index());
        // TODO: work around const
        newLambda->RemoveInput(cvip->index());
        auto graphImport = util::AssertedCast<const llvm::GraphImport>(cvip->origin());
        root->RemoveArgument(graphImport->index());
      }
    }
  }
}

jlm::rvsdg::output *
jlm::hls::ConnectRequestResponseMemPorts(
    const jlm::llvm::lambda::node * lambda,
    size_t argumentIndex,
    jlm::rvsdg::substitution_map & smap,
    const std::vector<jlm::rvsdg::simple_node *> & originalLoadNodes,
    const std::vector<jlm::rvsdg::simple_node *> & originalStoreNodes,
    const std::vector<jlm::rvsdg::simple_node *> & originalDecoupledNodes)
{
  //
  // We have the memory operations from the original lambda and need to lookup the corresponding
  // nodes in the new lambda
  //
  std::vector<jlm::rvsdg::simple_node *> loadNodes;
  std::vector<std::shared_ptr<const jlm::rvsdg::valuetype>> loadTypes;
  for (auto loadNode : originalLoadNodes)
  {
    JLM_ASSERT(smap.contains(*loadNode->output(0)));
    auto loadOutput = dynamic_cast<jlm::rvsdg::simple_output *>(smap.lookup(loadNode->output(0)));
    loadNodes.push_back(loadOutput->node());
    auto loadOp = jlm::util::AssertedCast<const jlm::llvm::LoadNonVolatileOperation>(
        &loadOutput->node()->operation());
    loadTypes.push_back(loadOp->GetLoadedType());
  }
  std::vector<jlm::rvsdg::simple_node *> storeNodes;
  for (auto storeNode : originalStoreNodes)
  {
    JLM_ASSERT(smap.contains(*storeNode->output(0)));
    auto storeOutput = dynamic_cast<jlm::rvsdg::simple_output *>(smap.lookup(storeNode->output(0)));
    storeNodes.push_back(storeOutput->node());
  }
  std::vector<jlm::rvsdg::simple_node *> decoupledNodes;
  for (auto decoupledNode : originalDecoupledNodes)
  {
    JLM_ASSERT(smap.contains(*decoupledNode->output(0)));
    auto decoupledOutput =
        dynamic_cast<jlm::rvsdg::simple_output *>(smap.lookup(decoupledNode->output(0)));
    decoupledNodes.push_back(decoupledOutput->node());
    loadTypes.push_back(jlm::rvsdg::bittype::Create(32));
  }

  auto lambdaRegion = lambda->subregion();

  auto loadResponses = mem_resp_op::create(*lambdaRegion->argument(argumentIndex), loadTypes);
  // The (decoupled) load nodes are replaced so the pointer to the types will become invalid
  loadTypes.clear();
  std::vector<jlm::rvsdg::output *> loadAddresses;
  for (size_t i = 0; i < loadNodes.size(); ++i)
  {
    auto routed = route_response(loadNodes[i]->region(), loadResponses[i]);
    // The smap contains the nodes from the original lambda so we need to use the original load node
    // when replacing the load since the smap must be updated
    auto replacement = ReplaceLoad(smap, originalLoadNodes[i], routed);
    auto address = route_request(lambdaRegion, replacement->output(replacement->noutputs() - 1));
    loadAddresses.push_back(address);
    std::shared_ptr<const jlm::rvsdg::valuetype> type;
    if (auto loadOperation = dynamic_cast<const jlm::hls::load_op *>(&replacement->operation()))
    {
      type = loadOperation->GetLoadedType();
    }
    else if (
        auto loadOperation =
            dynamic_cast<const jlm::hls::decoupled_load_op *>(&replacement->operation()))
    {
      type = loadOperation->GetLoadedType();
    }
    else
    {
      JLM_UNREACHABLE("Unknown load operation");
    }
    JLM_ASSERT(type);
    loadTypes.push_back(type);
  }
  for (size_t i = 0; i < decoupledNodes.size(); ++i)
  {
    JLM_UNREACHABLE("Handling of decoupled loads has not been updated after changing to the "
                    "version were a new lambda is created.");

    auto reponse = loadResponses[+loadNodes.size() + i];
    auto node = decoupledNodes[i];

    auto replacement = replace_decouple(lambda, node, reponse);
    // TODO: routing is probably not necessary
    auto addr = route_request(lambdaRegion, replacement->output(1));
    loadAddresses.push_back(addr);
    loadTypes.push_back(dynamic_cast<const jlm::hls::decoupled_load_op *>(&replacement->operation())
                            ->GetLoadedType());
  }
  std::vector<jlm::rvsdg::output *> storeOperands;
  for (size_t i = 0; i < storeNodes.size(); ++i)
  {
    // The smap contains the nodes from the original lambda so we need to use the oringal store node
    // when replacing the store since the smap must be updated
    auto replacement = ReplaceStore(smap, originalStoreNodes[i]);
    auto addr = route_request(lambdaRegion, replacement->output(replacement->noutputs() - 2));
    auto data = route_request(lambdaRegion, replacement->output(replacement->noutputs() - 1));
    storeOperands.push_back(addr);
    storeOperands.push_back(data);
  }

  return mem_req_op::create(loadAddresses, loadTypes, storeOperands, lambdaRegion)[0];
}

jlm::rvsdg::simple_node *
jlm::hls::ReplaceLoad(
    jlm::rvsdg::substitution_map & smap,
    const jlm::rvsdg::simple_node * originalLoad,
    jlm::rvsdg::output * response)
{
  // We have the load from the original lambda since it is needed to update the smap
  // We need the load in the new lambda such that we can replace it with a load node with explicit
  // memory ports
  auto replacedLoad = ((jlm::rvsdg::simple_output *)smap.lookup(originalLoad->output(0)))->node();

  auto loadAddress = replacedLoad->input(0)->origin();
  std::vector<jlm::rvsdg::output *> states;
  for (size_t i = 1; i < replacedLoad->ninputs(); ++i)
  {
    states.push_back(replacedLoad->input(i)->origin());
  }

  jlm::rvsdg::node * newLoad;
  if (states.empty())
  {
    auto outputs = jlm::hls::decoupled_load_op::create(*loadAddress, *response);
    newLoad = dynamic_cast<jlm::rvsdg::node_output *>(outputs[0])->node();
  }
  else
  {
    auto outputs = jlm::hls::load_op::create(*loadAddress, states, *response);
    newLoad = dynamic_cast<jlm::rvsdg::node_output *>(outputs[0])->node();
  }

  for (size_t i = 0; i < replacedLoad->noutputs(); ++i)
  {
    smap.insert(originalLoad->output(i), newLoad->output(i));
    replacedLoad->output(i)->divert_users(newLoad->output(i));
  }
  remove(replacedLoad);
  return dynamic_cast<jlm::rvsdg::simple_node *>(newLoad);
}

jlm::rvsdg::simple_node *
jlm::hls::ReplaceStore(
    jlm::rvsdg::substitution_map & smap,
    const jlm::rvsdg::simple_node * originalStore)
{
  // We have the store from the original lambda since it is needed to update the smap
  // We need the store in the new lambda such that we can replace it with a store node with explicit
  // memory ports
  auto replacedStore = ((jlm::rvsdg::simple_output *)smap.lookup(originalStore->output(0)))->node();

  auto addr = replacedStore->input(0)->origin();
  JLM_ASSERT(dynamic_cast<const jlm::llvm::PointerType *>(&addr->type()));
  auto data = replacedStore->input(1)->origin();
  std::vector<jlm::rvsdg::output *> states;
  for (size_t i = 2; i < replacedStore->ninputs(); ++i)
  {
    states.push_back(replacedStore->input(i)->origin());
  }
  auto outputs = jlm::hls::store_op::create(*addr, *data, states);
  auto newStore = dynamic_cast<jlm::rvsdg::node_output *>(outputs[0])->node();
  for (size_t i = 0; i < replacedStore->noutputs(); ++i)
  {
    smap.insert(originalStore->output(i), newStore->output(i));
    replacedStore->output(i)->divert_users(newStore->output(i));
  }
  remove(replacedStore);
  return dynamic_cast<jlm::rvsdg::simple_node *>(newStore);
}

jlm::rvsdg::simple_node *
ReplaceDecouple(
    jlm::rvsdg::substitution_map & smap,
    const jlm::llvm::lambda::node * lambda,
    jlm::rvsdg::simple_node * originalDecoupleRequest,
    jlm::rvsdg::output * response)
{
  // We have the load from the original lambda since it is needed to update the smap
  // We need the store in the new lambda such that we can replace it with a store node with explicit
  // memory ports
  auto decoupleRequest =
      ((jlm::rvsdg::simple_output *)smap.lookup(originalDecoupleRequest->output(0)))->node();

  JLM_ASSERT(dynamic_cast<const jlm::llvm::CallOperation *>(&decoupleRequest->operation()));
  auto channel = decoupleRequest->input(1)->origin();
  auto channelConstant = trace_channel(channel);

  auto decoupledResponse = find_decouple_response(lambda, channelConstant);

  auto addr = decoupleRequest->input(2)->origin();
  // connect out states to in states
  // TODO: use stategate?
  // one function pointer + 2 params
  for (size_t i = 3; i < decoupleRequest->ninputs(); ++i)
  {
    decoupleRequest->output(i - 3)->divert_users(decoupleRequest->input(i)->origin());
  }
  for (size_t i = 2; i < decoupledResponse->ninputs(); ++i)
  {
    decoupledResponse->output(i - 2 + 1)->divert_users(decoupledResponse->input(i)->origin());
  }
  // create this outside loop - need to tunnel outward from request and inward to response
  auto routedAddress = jlm::hls::route_request(lambda->subregion(), addr);
  // response is not routed inward for this case
  auto decoupledLoadOutput = jlm::hls::decoupled_load_op::create(*routedAddress, *response);
  // use a buffer here to make ready logic for response easy and consistent
  auto buf = jlm::hls::buffer_op::create(*decoupledLoadOutput[0], 2, true)[0];

  auto routedData = jlm::hls::route_response(decoupledResponse->region(), buf);
  decoupledResponse->output(0)->divert_users(routedData);
  // TODO: handle state edges?
  remove(decoupleRequest);
  remove(decoupledResponse);

  auto nodeOutput = dynamic_cast<jlm::rvsdg::node_output *>(decoupledLoadOutput[0])->node();
  return dynamic_cast<jlm::rvsdg::simple_node *>(nodeOutput);
}
