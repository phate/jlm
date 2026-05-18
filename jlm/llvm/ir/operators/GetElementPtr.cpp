/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/Trace.hpp>

namespace jlm::llvm
{

GetElementPtrOperation::~GetElementPtrOperation() noexcept = default;

bool
GetElementPtrOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const GetElementPtrOperation *>(&other);

  if (operation == nullptr || getPointeeType() != operation->getPointeeType()
      || narguments() != operation->narguments())
  {
    return false;
  }

  for (size_t n = 0; n < narguments(); n++)
  {
    if (operation->argument(n) != argument(n))
    {
      return false;
    }
  }

  return true;
}

std::string
GetElementPtrOperation::debug_string() const
{
  return "GetElementPtr";
}

std::unique_ptr<rvsdg::Operation>
GetElementPtrOperation::copy() const
{
  return std::make_unique<GetElementPtrOperation>(*this);
}

std::optional<std::vector<uint64_t>>
GetElementPtrOperation::tryGetConstantIndices(const rvsdg::Node & node) noexcept
{
  JLM_ASSERT(is<GetElementPtrOperation>(node.GetOperation()));

  std::vector<size_t> constants;
  for (auto & input : indices(node))
  {
    if (auto constant = tryGetConstantSignedInteger(*input.origin()))
    {
      constants.push_back(constant.value());
    }
    else
    {
      return std::nullopt;
    }
  }

  return constants;
}

}
