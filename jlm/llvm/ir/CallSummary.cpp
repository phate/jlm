/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/CallSummary.hpp>

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <deque>

namespace jlm::llvm
{

static void
AddToWorklist(std::deque<rvsdg::Input *> & worklist, rvsdg::Output::UserIteratorRange userRange)
{
  for (auto & user : userRange)
  {
    worklist.emplace_back(&user);
  }
}

CallSummary
ComputeCallSummary(const rvsdg::LambdaNode & lambdaNode)
{
  std::deque<rvsdg::Input *> worklist;
  AddToWorklist(worklist, lambdaNode.output()->Users());

  std::vector<rvsdg::SimpleNode *> directCalls;
  rvsdg::GraphExport * rvsdgExport = nullptr;
  std::vector<rvsdg::Input *> otherUsers;

  while (!worklist.empty())
  {
    auto input = worklist.front();
    worklist.pop_front();

    if (auto lambdaNode = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(*input))
    {
      auto & argument = *lambdaNode->MapInputContextVar(*input).inner;
      AddToWorklist(worklist, argument.Users());
      continue;
    }

    if (rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(*input))
    {
      otherUsers.emplace_back(input);
      continue;
    }

    if (auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*input))
    {
      auto rolevar = gammaNode->MapInput(*input);
      if (auto entryvar = std::get_if<rvsdg::GammaNode::EntryVar>(&rolevar))
      {
        for (auto & argument : entryvar->branchArgument)
        {
          AddToWorklist(worklist, argument->Users());
        }
      }
      continue;
    }

    if (auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*input))
    {
      auto output = gamma->MapBranchResultExitVar(*input).output;
      AddToWorklist(worklist, output->Users());
      continue;
    }

    if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*input))
    {
      auto loopvar = theta->MapInputLoopVar(*input);
      AddToWorklist(worklist, loopvar.pre->Users());
      continue;
    }

    if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*input))
    {
      auto loopvar = theta->MapPostLoopVar(*input);
      AddToWorklist(worklist, loopvar.output->Users());
      continue;
    }

    if (auto phi = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(*input))
    {
      auto ctxvar = phi->MapInputContextVar(*input);
      AddToWorklist(worklist, ctxvar.inner->Users());
      continue;
    }

    if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(*input))
    {
      auto fixvar = phi->MapResultFixVar(*input);
      AddToWorklist(worklist, fixvar.recref->Users());

      auto output = fixvar.output;
      AddToWorklist(worklist, output->Users());
      continue;
    }

    if (auto deltaNode = rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(*input))
    {
      auto ctxVar = deltaNode->MapInputContextVar(*input);
      AddToWorklist(worklist, ctxVar.inner->Users());
      continue;
    }

    if (rvsdg::TryGetRegionParentNode<rvsdg::DeltaNode>(*input))
    {
      otherUsers.emplace_back(input);
      continue;
    }

    auto inputNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*input);
    if (is<CallOperation>(inputNode) && input == inputNode->input(0))
    {
      directCalls.emplace_back(inputNode);
      continue;
    }

    if (auto graphExport = dynamic_cast<rvsdg::GraphExport *>(input))
    {
      rvsdgExport = graphExport;
      continue;
    }

    if (rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*input))
    {
      otherUsers.emplace_back(input);
      continue;
    }

    JLM_UNREACHABLE("This should have never happened!");
  }

  return CallSummary{ rvsdgExport, std::move(directCalls), std::move(otherUsers) };
}

}
