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

CallSummary
ComputeCallSummary(const rvsdg::LambdaNode & lambdaNode)
{
  std::deque<rvsdg::Input *> worklist;
  worklist.insert(worklist.end(), lambdaNode.output()->begin(), lambdaNode.output()->end());

  std::vector<rvsdg::SimpleNode *> directCalls;
  GraphExport * rvsdgExport = nullptr;
  std::vector<rvsdg::Input *> otherUsers;

  while (!worklist.empty())
  {
    auto input = worklist.front();
    worklist.pop_front();

    if (auto lambdaNode = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(*input))
    {
      auto & argument = *lambdaNode->MapInputContextVar(*input).inner;
      worklist.insert(worklist.end(), argument.begin(), argument.end());
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
          worklist.insert(worklist.end(), argument->begin(), argument->end());
        }
      }
      continue;
    }

    if (auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*input))
    {
      auto output = gamma->MapBranchResultExitVar(*input).output;
      worklist.insert(worklist.end(), output->begin(), output->end());
      continue;
    }

    if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*input))
    {
      auto loopvar = theta->MapInputLoopVar(*input);
      worklist.insert(worklist.end(), loopvar.pre->begin(), loopvar.pre->end());
      continue;
    }

    if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*input))
    {
      auto loopvar = theta->MapPostLoopVar(*input);
      worklist.insert(worklist.end(), loopvar.output->begin(), loopvar.output->end());
      continue;
    }

    if (auto phi = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(*input))
    {
      auto ctxvar = phi->MapInputContextVar(*input);
      worklist.insert(worklist.end(), ctxvar.inner->begin(), ctxvar.inner->end());
      continue;
    }

    if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(*input))
    {
      auto fixvar = phi->MapResultFixVar(*input);
      worklist.insert(worklist.end(), fixvar.recref->begin(), fixvar.recref->end());

      auto output = fixvar.output;
      worklist.insert(worklist.end(), output->begin(), output->end());
      continue;
    }

    if (auto cvinput = dynamic_cast<delta::cvinput *>(input))
    {
      auto argument = cvinput->arguments.first();
      worklist.insert(worklist.end(), argument->begin(), argument->end());
      continue;
    }

    if (auto deltaResult = dynamic_cast<delta::result *>(input))
    {
      otherUsers.emplace_back(deltaResult);
      continue;
    }

    auto inputNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*input);
    if (is<CallOperation>(inputNode) && input == inputNode->input(0))
    {
      directCalls.emplace_back(inputNode);
      continue;
    }

    if (auto graphExport = dynamic_cast<GraphExport *>(input))
    {
      rvsdgExport = graphExport;
      continue;
    }

    if (auto simpleInput = dynamic_cast<rvsdg::SimpleInput *>(input))
    {
      otherUsers.emplace_back(simpleInput);
      continue;
    }

    JLM_UNREACHABLE("This should have never happened!");
  }

  return CallSummary{ rvsdgExport, std::move(directCalls), std::move(otherUsers) };
}

}
