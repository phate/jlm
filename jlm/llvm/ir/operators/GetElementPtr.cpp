/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>

namespace jlm
{

GetElementPtrOperation::~GetElementPtrOperation() noexcept
= default;

bool
GetElementPtrOperation::operator==(const operation & other) const noexcept
{
  auto operation = dynamic_cast<const GetElementPtrOperation*>(&other);

  if (operation == nullptr
      || GetPointeeType() != operation->GetPointeeType()
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

std::unique_ptr<jive::operation>
GetElementPtrOperation::copy() const
{
  return std::unique_ptr<jive::operation>(new GetElementPtrOperation(*this));
}

}
