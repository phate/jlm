/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm
{

/*
 * CallOperation class
 */

CallOperation::~CallOperation() = default;

bool
CallOperation::operator==(const Operation & other) const noexcept
{
  auto callOperation = dynamic_cast<const CallOperation *>(&other);
  return callOperation && FunctionType_ == callOperation->FunctionType_;
}

std::string
CallOperation::debug_string() const
{
  return "CALL";
}

std::unique_ptr<rvsdg::Operation>
CallOperation::copy() const
{
  return std::make_unique<CallOperation>(*this);
}

rvsdg::Output &
CallOperation::TraceFunctionInput(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));
  const auto origin = GetFunctionInput(callNode).origin();
  return llvm::traceOutput(*origin);
}

std::unique_ptr<CallTypeClassifier>
CallOperation::ClassifyCall(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));
  auto & output = TraceFunctionInput(callNode);

  if (rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(output))
  {
    return CallTypeClassifier::CreateNonRecursiveDirectCallClassifier(output);
  }

  if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(output))
  {
    if (auto fix = phi->MapArgumentFixVar(output))
    {
      return CallTypeClassifier::CreateRecursiveDirectCallClassifier(output);
    }
  }

  if (auto argument = dynamic_cast<rvsdg::RegionArgument *>(&output))
  {
    if (argument->region() == &argument->region()->graph()->GetRootRegion())
    {
      return CallTypeClassifier::CreateExternalCallClassifier(*argument);
    }
  }

  return CallTypeClassifier::CreateIndirectCallClassifier(output);
}

bool
CallTypeClassifier::isSetjmpCall()
{
  if (!IsExternalCall())
    return false;

  if (const auto graphImport = dynamic_cast<const GraphImport *>(&GetImport()))
  {
    // In C and C++, setjmp is a macro to some underlying function. Clang uses _setjmp
    return graphImport->Name() == "_setjmp";
  }

  return false;
}

bool
CallTypeClassifier::isVaStartCall()
{
  if (!IsExternalCall())
    return false;

  if (const auto graphImport = dynamic_cast<const GraphImport *>(&GetImport()))
  {
    // LLVM intrinsics can contain extra suffixes like .p0, so check its prefix
    return graphImport->Name().rfind("llvm.va_start", 0) == 0;
  }

  return false;
}

}
