/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/tac.hpp>

#include <sstream>

namespace jlm::llvm
{

/* tacvariable */

tacvariable::~tacvariable()
{}

/* taclist */

taclist::~taclist()
{
  for (const auto & tac : tacs_)
    delete tac;
}

/* tac */

static void
check_operands(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const variable *> & operands)
{
  if (operands.size() != operation.narguments())
    throw util::error("invalid number of operands.");

  for (size_t n = 0; n < operands.size(); n++)
  {
    if (operands[n]->type() != *operation.argument(n))
      throw util::TypeError(
          operands[n]->type().debug_string(),
          operation.argument(n)->debug_string());
  }
}

static void
check_results(
    const rvsdg::SimpleOperation & operation,
    const std::vector<std::unique_ptr<tacvariable>> & results)
{
  if (results.size() != operation.nresults())
    throw util::error("invalid number of variables.");

  for (size_t n = 0; n < results.size(); n++)
  {
    if (results[n]->type() != *operation.result(n))
      throw util::error("invalid type.");
  }
}

tac::tac(const rvsdg::SimpleOperation & operation, const std::vector<const variable *> & operands)
    : operands_(operands),
      operation_(operation.copy())
{
  check_operands(operation, operands);

  auto names = create_names(operation.nresults());
  create_results(operation, names);
}

tac::tac(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const variable *> & operands,
    const std::vector<std::string> & names)
    : operands_(operands),
      operation_(operation.copy())
{
  check_operands(operation, operands);

  if (names.size() != operation.nresults())
    throw util::error("Invalid number of result names.");

  create_results(operation, names);
}

tac::tac(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const variable *> & operands,
    std::vector<std::unique_ptr<tacvariable>> results)
    : operands_(operands),
      operation_(operation.copy()),
      results_(std::move(results))
{
  check_operands(operation, operands);
  check_results(operation, results_);
}

void
tac::convert(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const variable *> & operands)
{
  check_operands(operation, operands);

  results_.clear();
  operands_ = operands;
  operation_ = operation.copy();

  auto names = create_names(operation.nresults());
  create_results(operation, names);
}

void
tac::replace(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const variable *> & operands)
{
  check_operands(operation, operands);
  check_results(operation, results_);

  operands_ = operands;
  operation_ = operation.copy();
}

std::string
tac::ToAscii(const jlm::llvm::tac & threeAddressCode)
{
  std::string resultString;
  for (size_t n = 0; n < threeAddressCode.nresults(); n++)
  {
    resultString += threeAddressCode.result(n)->debug_string();
    if (n != threeAddressCode.nresults() - 1)
      resultString += ", ";
  }

  std::string operandString;
  for (size_t n = 0; n < threeAddressCode.noperands(); n++)
  {
    operandString += threeAddressCode.operand(n)->debug_string();
    if (n != threeAddressCode.noperands() - 1)
      operandString += ", ";
  }

  std::string operationString = threeAddressCode.operation().debug_string();
  std::string resultOperationSeparator = resultString.empty() ? "" : " = ";
  std::string operationOperandSeparator = operandString.empty() ? "" : " ";
  return resultString + resultOperationSeparator + operationString + operationOperandSeparator
       + operandString;
}

}
