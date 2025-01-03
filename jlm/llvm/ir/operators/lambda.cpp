/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <deque>

namespace jlm::llvm::lambda
{

operation::~operation() = default;

std::string
operation::debug_string() const
{
  return util::strfmt("LAMBDA[", name(), "]");
}

bool
operation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const lambda::operation *>(&other);
  return op && op->type() == type() && op->name() == name() && op->linkage() == linkage()
      && op->attributes() == attributes();
}

std::unique_ptr<rvsdg::Operation>
operation::copy() const
{
  return std::make_unique<lambda::operation>(*this);
}

/* lambda node class */

node::~node() = default;

node::node(rvsdg::Region & parent, lambda::operation op)
    : StructuralNode(std::move(op), &parent, 1)
{
  ArgumentAttributes_.resize(GetOperation().Type()->NumArguments());
}

const lambda::operation &
node::GetOperation() const noexcept
{
  return *jlm::util::AssertedCast<const lambda::operation>(&StructuralNode::GetOperation());
}

[[nodiscard]] std::vector<rvsdg::output *>
node::GetFunctionArguments() const
{
  std::vector<rvsdg::output *> arguments;
  const auto & type = GetOperation().Type();
  for (std::size_t n = 0; n < type->Arguments().size(); ++n)
  {
    arguments.push_back(subregion()->argument(n));
  }
  return arguments;
}

[[nodiscard]] std::vector<rvsdg::input *>
node::GetFunctionResults() const
{
  std::vector<rvsdg::input *> results;
  for (std::size_t n = 0; n < subregion()->nresults(); ++n)
  {
    results.push_back(subregion()->result(n));
  }
  return results;
}

[[nodiscard]] node::ContextVar
node::MapInputContextVar(const rvsdg::input & input) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<node>(input) == this);
  return ContextVar{ const_cast<rvsdg::input *>(&input),
                     subregion()->argument(GetOperation().Type()->NumArguments() + input.index()) };
}

[[nodiscard]] std::optional<node::ContextVar>
node::MapBinderContextVar(const rvsdg::output & output) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerRegion(output) == subregion());
  auto numArguments = GetOperation().Type()->NumArguments();
  if (output.index() >= numArguments)
  {
    return ContextVar{ input(output.index() - GetOperation().Type()->NumArguments()),
                       const_cast<rvsdg::output *>(&output) };
  }
  else
  {
    return std::nullopt;
  }
}

[[nodiscard]] std::vector<node::ContextVar>
node::GetContextVars() const noexcept
{
  std::vector<ContextVar> vars;
  for (size_t n = 0; n < ninputs(); ++n)
  {
    vars.push_back(
        ContextVar{ input(n), subregion()->argument(n + GetOperation().Type()->NumArguments()) });
  }
  return vars;
}

node::ContextVar
node::AddContextVar(jlm::rvsdg::output & origin)
{
  auto input = rvsdg::StructuralInput::create(this, &origin, origin.Type());
  auto argument = &rvsdg::RegionArgument::Create(*subregion(), input, origin.Type());
  return ContextVar{ input, argument };
}

rvsdg::output &
node::GetMemoryStateRegionArgument() const noexcept
{
  auto argument = GetFunctionArguments().back();
  JLM_ASSERT(is<MemoryStateType>(argument->type()));
  return *argument;
}

rvsdg::input &
node::GetMemoryStateRegionResult() const noexcept
{
  auto result = GetFunctionResults().back();
  JLM_ASSERT(is<MemoryStateType>(result->type()));
  return *result;
}

rvsdg::SimpleNode *
node::GetMemoryStateExitMerge(const lambda::node & lambdaNode) noexcept
{
  auto & result = lambdaNode.GetMemoryStateRegionResult();

  auto node = rvsdg::output::GetNode(*result.origin());
  return is<LambdaExitMemoryStateMergeOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                       : nullptr;
}

rvsdg::SimpleNode *
node::GetMemoryStateEntrySplit(const lambda::node & lambdaNode) noexcept
{
  auto & argument = lambdaNode.GetMemoryStateRegionArgument();

  // If a memory state entry split node is present, then we would expect the node to be the only
  // user of the memory state argument.
  if (argument.nusers() != 1)
    return nullptr;

  auto node = rvsdg::node_input::GetNode(**argument.begin());
  return is<LambdaEntryMemoryStateSplitOperation>(node) ? dynamic_cast<rvsdg::SimpleNode *>(node)
                                                        : nullptr;
}

lambda::node *
node::create(
    rvsdg::Region * parent,
    std::shared_ptr<const jlm::llvm::FunctionType> type,
    const std::string & name,
    const llvm::linkage & linkage,
    const attributeset & attributes)
{
  lambda::operation op(type, name, linkage, attributes);
  auto node = new lambda::node(*parent, std::move(op));

  for (auto & argumentType : type->Arguments())
    rvsdg::RegionArgument::Create(*node->subregion(), nullptr, argumentType);

  return node;
}

rvsdg::output *
node::finalize(const std::vector<jlm::rvsdg::output *> & results)
{
  /* check if finalized was already called */
  if (noutputs() > 0)
  {
    JLM_ASSERT(noutputs() == 1);
    return output();
  }

  if (type().NumResults() != results.size())
    throw util::error("Incorrect number of results.");

  for (size_t n = 0; n < results.size(); n++)
  {
    auto & expected = type().ResultType(n);
    auto & received = results[n]->type();
    if (results[n]->type() != type().ResultType(n))
      throw util::error("Expected " + expected.debug_string() + ", got " + received.debug_string());

    if (results[n]->region() != subregion())
      throw util::error("Invalid operand region.");
  }

  for (const auto & origin : results)
    rvsdg::RegionResult::Create(*origin->region(), *origin, nullptr, origin->Type());

  return append_output(std::make_unique<rvsdg::StructuralOutput>(this, PointerType::Create()));
}

rvsdg::output *
node::output() const noexcept
{
  return StructuralNode::output(0);
}

lambda::node *
node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  return util::AssertedCast<lambda::node>(rvsdg::Node::copy(region, operands));
}

lambda::node *
node::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  auto lambda = create(region, Type(), name(), linkage(), attributes());

  /* add context variables */
  rvsdg::SubstitutionMap subregionmap;
  for (const auto & cv : GetContextVars())
  {
    auto origin = smap.lookup(cv.input->origin());
    subregionmap.insert(cv.inner, lambda->AddContextVar(*origin).inner);
  }

  /* collect function arguments */
  auto args = GetFunctionArguments();
  auto newArgs = lambda->GetFunctionArguments();
  JLM_ASSERT(args.size() == newArgs.size());
  for (size_t n = 0; n < args.size(); n++)
  {
    lambda->SetArgumentAttributes(*newArgs[n], GetArgumentAttributes(*args[n]));
    subregionmap.insert(args[n], newArgs[n]);
  }

  /* copy subregion */
  subregion()->copy(lambda->subregion(), subregionmap, false, false);

  /* collect function results */
  std::vector<jlm::rvsdg::output *> results;
  for (auto result : GetFunctionResults())
    results.push_back(subregionmap.lookup(result->origin()));

  /* finalize lambda */
  auto o = lambda->finalize(results);
  smap.insert(output(), o);

  lambda->ArgumentAttributes_ = ArgumentAttributes_;

  return lambda;
}

std::unique_ptr<node::CallSummary>
node::ComputeCallSummary() const
{
  std::deque<rvsdg::input *> worklist;
  worklist.insert(worklist.end(), output()->begin(), output()->end());

  std::vector<CallNode *> directCalls;
  GraphExport * rvsdgExport = nullptr;
  std::vector<rvsdg::input *> otherUsers;

  while (!worklist.empty())
  {
    auto input = worklist.front();
    worklist.pop_front();

    auto inputNode = rvsdg::input::GetNode(*input);

    if (auto lambdaNode = rvsdg::TryGetOwnerNode<lambda::node>(*input))
    {
      auto & argument = *lambdaNode->MapInputContextVar(*input).inner;
      worklist.insert(worklist.end(), argument.begin(), argument.end());
      continue;
    }

    if (rvsdg::TryGetRegionParentNode<lambda::node>(*input))
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

  return CallSummary::Create(rvsdgExport, std::move(directCalls), std::move(otherUsers));
}

bool
node::IsExported(const lambda::node & lambdaNode)
{
  auto callSummary = lambdaNode.ComputeCallSummary();
  return callSummary->IsExported();
}

[[nodiscard]] const jlm::llvm::attributeset &
node::GetArgumentAttributes(const rvsdg::output & argument) const noexcept
{
  JLM_ASSERT(argument.index() < ArgumentAttributes_.size());
  return ArgumentAttributes_[argument.index()];
}

void
node::SetArgumentAttributes(rvsdg::output & argument, const jlm::llvm::attributeset & attributes)
{
  ArgumentAttributes_[argument.index()] = attributes;
}

}
