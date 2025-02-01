/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/CallSummary.hpp>

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <deque>

namespace jlm::llvm
{

CallSummary
ComputeCallSummary(const rvsdg::LambdaNode & lambdaNode)
{
  std::deque<rvsdg::input *> worklist;
  worklist.insert(worklist.end(), lambdaNode.output()->begin(), lambdaNode.output()->end());

  std::vector<CallNode *> directCalls;
  GraphExport * rvsdgExport = nullptr;
  std::vector<rvsdg::input *> otherUsers;

  while (!worklist.empty())
  {
    auto input = worklist.front();
    worklist.pop_front();

    auto inputNode = rvsdg::input::GetNode(*input);

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

    if (auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(inputNode))
    {
      for (auto & argument : gammaNode->MapInputEntryVar(*input).branchArgument)
      {
        worklist.insert(worklist.end(), argument->begin(), argument->end());
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

    if (auto cvinput = dynamic_cast<phi::cvinput *>(input))
    {
      auto argument = cvinput->argument();
      worklist.insert(worklist.end(), argument->begin(), argument->end());
      continue;
    }

    if (auto rvresult = dynamic_cast<phi::rvresult *>(input))
    {
      auto argument = rvresult->argument();
      worklist.insert(worklist.end(), argument->begin(), argument->end());

      auto output = rvresult->output();
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

    if (is<CallOperation>(inputNode) && input == inputNode->input(0))
    {
      directCalls.emplace_back(util::AssertedCast<CallNode>(inputNode));
      continue;
    }

    if (auto graphExport = dynamic_cast<GraphExport *>(input))
    {
      rvsdgExport = graphExport;
      continue;
    }

    if (auto simpleInput = dynamic_cast<rvsdg::simple_input *>(input))
    {
      otherUsers.emplace_back(simpleInput);
      continue;
    }

    JLM_UNREACHABLE("This should have never happened!");
  }

  return CallSummary{ rvsdgExport, std::move(directCalls), std::move(otherUsers) };
}

}
