/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/common.hpp>

namespace jlm::hls
{

void
remove_unused_state(rvsdg::Region * region, bool can_remove_arguments)
{
  // process children first so that unnecessary users get removed
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      if (auto gn = dynamic_cast<rvsdg::GammaNode *>(node))
      {
        // process subnodes first
        for (size_t n = 0; n < gn->nsubregions(); n++)
        {
          remove_unused_state(gn->subregion(n), false);
        }
        remove_gamma_passthrough(gn);
      }
      else if (auto ln = dynamic_cast<llvm::lambda::node *>(node))
      {
        remove_unused_state(structnode->subregion(0), false);
        remove_lambda_passthrough(ln);
      }
      else
      {
        JLM_ASSERT(structnode->nsubregions() == 1);
        remove_unused_state(structnode->subregion(0));
      }
    }
  }
  // exit will come before entry
  for (auto & node : rvsdg::BottomUpTraverser(region))
  {
    if (auto simplenode = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (dynamic_cast<const llvm::LambdaExitMemoryStateMergeOperation *>(&node->GetOperation()))
      {
        std::vector<jlm::rvsdg::output *> nv;
        for (size_t i = 0; i < simplenode->ninputs(); ++i)
        {
          if (auto so = dynamic_cast<jlm::rvsdg::simple_output *>(simplenode->input(i)->origin()))
          {
            if (dynamic_cast<const llvm::LambdaEntryMemoryStateSplitOperation *>(
                    &so->node()->GetOperation()))
            {
              // skip things coming from entry
              continue;
            }
          }
          nv.push_back(simplenode->input(i)->origin());
        }
        if (nv.size() == 0)
        {
          // special case were no entry/exit operator is needed
          auto entry_node =
              dynamic_cast<jlm::rvsdg::node_output *>(simplenode->input(0)->origin())->node();
          JLM_ASSERT(dynamic_cast<const llvm::LambdaEntryMemoryStateSplitOperation *>(
              &entry_node->GetOperation()));
          simplenode->output(0)->divert_users(entry_node->input(0)->origin());
          remove(simplenode);
          remove(entry_node);
        }
        else if (nv.size() != simplenode->ninputs())
        {
          auto & new_state = llvm::LambdaExitMemoryStateMergeOperation::Create(*region, nv);
          simplenode->output(0)->divert_users(&new_state);
          remove(simplenode);
        }
      }
      else if (dynamic_cast<const llvm::LambdaEntryMemoryStateSplitOperation *>(
                   &node->GetOperation()))
      {
        std::vector<jlm::rvsdg::output *> nv;
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          if (simplenode->output(i)->nusers())
          {
            nv.push_back(simplenode->output(i));
          }
        }
        if (nv.size() != simplenode->noutputs())
        {
          auto new_states = llvm::LambdaEntryMemoryStateSplitOperation::Create(
              *simplenode->input(0)->origin(),
              nv.size());
          for (size_t i = 0; i < nv.size(); ++i)
          {
            nv[i]->divert_users(new_states[i]);
          }
          remove(simplenode);
        }
      }
    }
  }
  if (can_remove_arguments)
  {
    // check if an input is passed through unnecessarily
    for (int i = region->narguments() - 1; i >= 0; --i)
    {
      auto arg = region->argument(i);
      if (is_passthrough(arg))
      {
        remove_region_passthrough(arg);
      }
    }
  }
}

void
remove_unused_state(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  remove_unused_state(root);
}

void
remove_gamma_passthrough(rvsdg::GammaNode * gn)
{ // remove inputs in reverse
  auto entryvars = gn->GetEntryVars();
  for (int i = entryvars.size() - 1; i >= 0; --i)
  {
    bool can_remove = true;
    size_t res_index = 0;
    auto arg = entryvars[i].branchArgument[0];
    if (arg->nusers() == 1)
    {
      auto res = dynamic_cast<rvsdg::RegionResult *>(*arg->begin());
      res_index = res ? res->index() : res_index;
    }
    for (size_t n = 0; n < gn->nsubregions(); n++)
    {
      auto sr = gn->subregion(n);
      can_remove &=
          is_passthrough(sr->argument(i)) &&
          // check that all subregions pass through to the same result
          dynamic_cast<rvsdg::RegionResult *>(*sr->argument(i)->begin())->index() == res_index;
    }
    if (can_remove)
    {
      auto origin = entryvars[i].input->origin();
      // divert users of output to origin of input

      gn->output(res_index)->divert_users(origin);
      gn->output(res_index)->results.clear();
      gn->RemoveOutput(res_index);
      // remove input
      gn->input(i + 1)->arguments.clear();
      gn->RemoveInput(i + 1);
      for (size_t j = 0; j < gn->nsubregions(); ++j)
      {
        JLM_ASSERT(gn->subregion(j)->result(res_index)->origin() == gn->subregion(j)->argument(i));
        JLM_ASSERT(gn->subregion(j)->argument(i)->nusers() == 1);
        gn->subregion(j)->RemoveResult(res_index);
        JLM_ASSERT(gn->subregion(j)->argument(i)->nusers() == 0);
        gn->subregion(j)->RemoveArgument(i);
      }
    }
  }
}

jlm::llvm::lambda::node *
remove_lambda_passthrough(llvm::lambda::node * ln)
{
  const auto & op = ln->GetOperation();
  auto old_fcttype = op.type();
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> new_argument_types;
  for (size_t i = 0; i < old_fcttype.NumArguments(); ++i)
  {
    auto arg = ln->subregion()->argument(i);
    auto argtype = old_fcttype.Arguments()[i];
    JLM_ASSERT(*argtype == arg->type());
    if (!is_passthrough(arg))
    {
      new_argument_types.push_back(argtype);
    }
  }
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> new_result_types;
  for (size_t i = 0; i < old_fcttype.NumResults(); ++i)
  {
    auto res = ln->subregion()->result(i);
    auto restype = &old_fcttype.ResultType(i);
    JLM_ASSERT(*restype == res->type());
    if (!is_passthrough(res))
    {
      new_result_types.push_back(old_fcttype.Results()[i]);
    }
  }
  auto new_fcttype = rvsdg::FunctionType::Create(new_argument_types, new_result_types);
  auto new_lambda = llvm::lambda::node::create(
      ln->region(),
      new_fcttype,
      op.name(),
      op.linkage(),
      op.attributes());

  rvsdg::SubstitutionMap smap;
  for (const auto & ctxvar : ln->GetContextVars())
  {
    // copy over context vars
    smap.insert(ctxvar.inner, new_lambda->AddContextVar(*ctxvar.input->origin()).inner);
  }

  size_t new_i = 0;
  auto args = ln->GetFunctionArguments();
  auto new_args = new_lambda->GetFunctionArguments();
  JLM_ASSERT(args.size() >= new_args.size());
  for (size_t i = 0; i < args.size(); ++i)
  {
    auto arg = args[i];
    if (!is_passthrough(arg))
    {
      smap.insert(arg, new_args[new_i]);
      new_i++;
    }
  }
  ln->subregion()->copy(new_lambda->subregion(), smap, false, false);

  std::vector<jlm::rvsdg::output *> new_results;
  for (auto res : ln->GetFunctionResults())
  {
    if (!is_passthrough(res))
    {
      new_results.push_back(smap.lookup(res->origin()));
    }
  }
  auto new_out = new_lambda->finalize(new_results);

  // TODO handle functions at other levels?
  JLM_ASSERT(ln->region() == &ln->region()->graph()->GetRootRegion());
  JLM_ASSERT((*ln->output()->begin())->region() == &ln->region()->graph()->GetRootRegion());

  //	ln->output()->divert_users(new_out); // can't divert since the type changed
  JLM_ASSERT(ln->output()->nusers() == 1);
  ln->region()->RemoveResult((*ln->output()->begin())->index());
  auto oldExport = jlm::llvm::ComputeCallSummary(*ln).GetRvsdgExport();
  jlm::llvm::GraphExport::Create(*new_out, oldExport ? oldExport->Name() : "");
  remove(ln);
  return new_lambda;
}

void
remove_region_passthrough(const rvsdg::RegionArgument * arg)
{
  auto res = dynamic_cast<rvsdg::RegionResult *>(*arg->begin());
  auto origin = arg->input()->origin();
  // divert users of output to origin of input
  arg->region()->node()->output(res->output()->index())->divert_users(origin);
  // remove result first so argument has no users
  arg->region()->RemoveResult(res->index());
  arg->region()->RemoveArgument(arg->index());
  arg->region()->node()->RemoveInput(arg->input()->index());
  arg->region()->node()->RemoveOutput(res->output()->index());
}

bool
is_passthrough(const rvsdg::input * res)
{
  auto arg = dynamic_cast<rvsdg::RegionArgument *>(res->origin());
  if (arg)
  {
    return true;
  }
  return false;
}

bool
is_passthrough(const rvsdg::output * arg)
{
  if (arg->nusers() == 1)
  {
    auto res = dynamic_cast<rvsdg::RegionResult *>(*arg->begin());
    // used only by a result
    if (res)
    {
      return true;
    }
  }
  return false;
}

void
RemoveInvariantMemoryStateEdges(jlm::rvsdg::RegionResult * memoryState)
{
  // We only apply this for memory state edges that is invariant between
  // LambdaEntryMemoryStateSplit and LambdaExitMemoryStateMerge nodes.
  // So we first check if we have a LambdaExitMemoryStateMerge node.
  if (memoryState->origin()->nusers() == 1)
  {
    auto exitNode = jlm::rvsdg::output::GetNode(*memoryState->origin());
    // Change to is<>
    if (dynamic_cast<const jlm::llvm::LambdaExitMemoryStateMergeOperation *>(
            &exitNode->GetOperation()))
    {
      // Check if we have any invariant edge(s) between the two nodes
      std::vector<jlm::rvsdg::input *> inputs;
      std::vector<jlm::rvsdg::output *> outputs;
      jlm::rvsdg::Node * entryNode = nullptr;
      for (size_t i = 0; i < exitNode->ninputs(); i++)
      {
        // Check if the output has only one user and if it is a LambdaEntryMemoryStateMerge
        if (exitNode->input(i)->origin()->nusers() == 1)
        {
          auto node = jlm::rvsdg::output::GetNode(*exitNode->input(i)->origin());
          // TODO change to is<>
          if (dynamic_cast<const jlm::llvm::LambdaEntryMemoryStateSplitOperation *>(
                  &node->GetOperation()))
          {
            // Found an invariant memory state edge, so going to replace the exitNode
            entryNode = node;
          }
          else
          {
            // Keep track of edges that is to be kept since they are not invariant
            inputs.push_back(node->input(i));
            outputs.push_back(node->input(i)->origin());
          }
        }
      }

      // If the exitNode is set, then we have found at least one invariant edge
      if (entryNode != nullptr)
      {
        // Replace LambdaEntryMemoryStateMerge and LambdaExitMemoryStateMergeOperation
        if (outputs.size() <= 1)
        {
          // Single or no edge that is not invariant
          // So we can elmintate the two MemoryState nodes
          memoryState->divert_to(exitNode->input(0)->origin());
          entryNode->output(0)->divert_users(entryNode->input(0)->origin());
        }
        else
        {
          // Replace the entry and exit node with new ones without the invariant edge(s)
          auto newEntryNodeOutputs = jlm::llvm::LambdaEntryMemoryStateSplitOperation::Create(
              *entryNode->input(0)->origin(),
              1 + inputs.size());
          auto newExitNodeOutput = &jlm::llvm::LambdaExitMemoryStateMergeOperation::Create(
              *exitNode->region(),
              outputs);
          exitNode->output(0)->divert_users(newExitNodeOutput);
//          exitNode->output(0)->divert_users(&jlm::llvm::LambdaExitMemoryStateMergeOperation::Create(
//              *exitNode->region(),
//              outputs));
          int i = 0;
          for (auto input : inputs)
          {
            entryNode->output(input->index())->divert_users(newEntryNodeOutputs.at(i));
          }
        }
        JLM_ASSERT(exitNode->IsDead());
        jlm::rvsdg::remove(exitNode);
        JLM_ASSERT(entryNode->IsDead());
        jlm::rvsdg::remove(entryNode);
      }
    }
  }
}

void
RemoveLambdaInvariantMemoryStateEdges(llvm::RvsdgModule & rvsdgModule)
{
  auto & root = rvsdgModule.Rvsdg().GetRootRegion();
  // R-HLS only has one function in the root region
  JLM_ASSERT(root.nnodes() == 1);
  auto lambda = jlm::util::AssertedCast<jlm::llvm::lambda::node>(root.Nodes().begin().ptr());

  for (auto result : lambda->subregion()->Results())
  {
    if (dynamic_cast<const jlm::llvm::MemoryStateType *>(&*result->Type()))
    {
      RemoveInvariantMemoryStateEdges(result);
    }
  }
}

void
RemoveLambdaInvariantStateEdges(llvm::RvsdgModule & rvsdgModule)
{
  auto & root = rvsdgModule.Rvsdg().GetRootRegion();
  // R-HLS only has one function in the root region
  JLM_ASSERT(root.nnodes() == 1);
  auto lambdaNode = jlm::util::AssertedCast<jlm::llvm::lambda::node>(root.Nodes().begin().ptr());
  remove_lambda_passthrough(lambdaNode);
}

} // namespace jlm::hls
