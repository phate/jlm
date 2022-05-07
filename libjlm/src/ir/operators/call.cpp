/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/RvsdgModule.hpp>

namespace jlm {

/**
 * Support functions
 */

using InvariantOutputMap = std::unordered_map<const jive::output*, jive::input*>;

static jive::input *
invariantInput(
  const jive::output & output,
  InvariantOutputMap & invariantOutputs);

static jive::structural_input *
invariantInput(
  const jive::gamma_output & output,
  InvariantOutputMap & invariantOutputs)
{
  size_t n;
  jive::structural_input * input = nullptr;
  for (n = 0; n < output.nresults(); n++) {
    auto origin = output.result(n)->origin();

    bool resultIsInvariant = false;
    while (true) {
      if (auto argument = dynamic_cast<const jive::argument*>(origin)) {
        resultIsInvariant = true;
        input = argument->input();
        break;
      }

      if (auto input = invariantInput(*origin, invariantOutputs)) {
        origin = input->origin();
        continue;
      }

      break;
    }

    if (resultIsInvariant == false)
      break;
  }

  if (n == output.nresults()) {
    invariantOutputs[&output] = input;
    return input;
  }

  invariantOutputs[&output] = nullptr;
  return nullptr;
}

static jive::theta_input *
invariantInput(
  const jive::theta_output & output,
  InvariantOutputMap & invariantOutputs)
{
  auto origin = output.result()->origin();

  while (true) {
    if (origin == output.argument()) {
      invariantOutputs[&output] = output.input();
      return output.input();
    }

    if (auto input = invariantInput(*origin, invariantOutputs)) {
      origin = input->origin();
      continue;
    }

    break;
  }

  invariantOutputs[&output] = nullptr;
  return nullptr;
}

static jive::input *
invariantInput(
  const jive::output & output,
  InvariantOutputMap & invariantOutputs)
{
  /*
    We already have seen the output, just return the corresponding input.
  */
  if (invariantOutputs.find(&output) != invariantOutputs.end())
    return invariantOutputs[&output];

  if (auto thetaOutput = dynamic_cast<const jive::theta_output*>(&output))
    return invariantInput(*thetaOutput, invariantOutputs);

  if (auto thetaArgument = is_theta_argument(&output)) {
    auto thetaInput = static_cast<const jive::theta_input*>(thetaArgument->input());
    return invariantInput(*thetaInput->output(), invariantOutputs);
  }

  if (auto gammaOutput = dynamic_cast<const jive::gamma_output*>(&output))
    return invariantInput(*gammaOutput, invariantOutputs);

  return nullptr;
}

static jive::input *
invariantInput(const jive::output & output)
{
  InvariantOutputMap invariantOutputs;
  return invariantInput(output, invariantOutputs);
}

/*
 * CallOperation class
 */

CallOperation::~CallOperation()
= default;

bool
CallOperation::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const CallOperation*>(&other);
	if (!op || op->narguments() != narguments() || op->nresults() != nresults())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (op->argument(n) != argument(n))
			return false;
	}

	for (size_t n = 0; n < nresults(); n++) {
		if (op->result(n) != result(n))
			return false;
	}

	return true;
}

std::string
CallOperation::debug_string() const
{
	return "CALL";
}

std::unique_ptr<jive::operation>
CallOperation::copy() const
{
	return std::unique_ptr<jive::operation>(new CallOperation(*this));
}

/**
 * CallNode class
 */

jive::output *
CallNode::TraceFunctionInput(const CallNode & callNode)
{
  auto origin = callNode.GetFunctionInput()->origin();

  while (true) {
    if (is<lambda::output>(origin))
      return origin;

    if (is<lambda::fctargument>(origin))
      return origin;

    if (is_import(origin))
      return origin;

    if (is<jive::simple_op>(jive::node_output::node(origin)))
      return origin;

    if (is<lambda::cvargument>(origin)) {
      auto argument = static_cast<const jive::argument*>(origin);
      origin = argument->input()->origin();
      continue;
    }

    if (auto output = is_gamma_output(origin)) {
      if (auto input = invariantInput(*output)) {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (auto argument = is_gamma_argument(origin)) {
      origin = argument->input()->origin();
      continue;
    }

    if (auto output = is_theta_output(origin)) {
      if (auto input = invariantInput(*output)) {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (auto argument = is_theta_argument(origin)) {
      if (auto input = invariantInput(*argument)) {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (is_phi_cv(origin)) {
      auto argument = static_cast<const jive::argument*>(origin);
      origin = argument->input()->origin();
      continue;
    }

    if (is_phi_recvar_argument(origin)) {
      auto argument = static_cast<const jive::argument*>(origin);
      /*
        FIXME: This assumes that all recursion variables where added before the dependencies. It
        would be better if we did not use the index for retrieving the result, but instead
        explicitly encoded it in an phi_argument.
      */
      origin = argument->region()->result(argument->index())->origin();
      continue;
    }

    if (auto rvoutput = dynamic_cast<const phi::rvoutput*>(origin)) {
      origin = rvoutput->result()->origin();
      continue;
    }

    JLM_UNREACHABLE("We should have never reached this statement.");
  }
}

std::unique_ptr<CallTypeClassifier>
CallNode::ClassifyCall(const CallNode &callNode)
{
  auto output = CallNode::TraceFunctionInput(callNode);
  auto region = output->region();

  if (dynamic_cast<const lambda::output*>(output))
    return CallTypeClassifier::CreateDirectCallClassifier(*output);

  if (dynamic_cast<const jive::argument*>(output)
  && region == region->graph()->root())
    return CallTypeClassifier::CreateExternalCallClassifier(*output);

  return CallTypeClassifier::CreateIndirectCallClassifier(*output);
}

lambda::node *
CallNode::IsDirectCall(const CallNode & callNode)
{
  auto output = CallNode::TraceFunctionInput(callNode);

  if (auto o = dynamic_cast<const lambda::output*>(output))
    return o->node();

  return nullptr;
}

}
