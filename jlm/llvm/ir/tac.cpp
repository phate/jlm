/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/tac.hpp>

#include <sstream>

namespace jlm::llvm
{

ThreeAddressCodeVariable::~ThreeAddressCodeVariable() noexcept = default;

ThreeAddressCodeList::~ThreeAddressCodeList() noexcept
{
  for (const auto & tac : tacs_)
    delete tac;
}

static void
check_operands(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const Variable *> & operands)
{
  if (operands.size() != operation.narguments())
    throw util::Error("invalid number of operands.");

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
    const std::vector<std::unique_ptr<ThreeAddressCodeVariable>> & results)
{
  if (results.size() != operation.nresults())
    throw util::Error("invalid number of variables.");

  for (size_t n = 0; n < results.size(); n++)
  {
    if (results[n]->type() != *operation.result(n))
      throw util::Error("invalid type.");
  }
}

ThreeAddressCode::ThreeAddressCode(
    std::unique_ptr<rvsdg::SimpleOperation> operation,
    const std::vector<const Variable *> & operands)
    : operands_(operands),
      operation_(std::move(operation))
{
  check_operands(this->operation(), operands);

  auto names = create_names(this->operation().nresults());
  create_results(this->operation(), names);
}

ThreeAddressCode::ThreeAddressCode(
    std::unique_ptr<rvsdg::SimpleOperation> operation,
    const std::vector<const Variable *> & operands,
    const std::vector<std::string> & names)
    : operands_(operands),
      operation_(std::move(operation))
{
  check_operands(this->operation(), operands);

  if (names.size() != this->operation().nresults())
    throw util::Error("Invalid number of result names.");

  create_results(this->operation(), names);
}

ThreeAddressCode::ThreeAddressCode(
    std::unique_ptr<rvsdg::SimpleOperation> operation,
    const std::vector<const Variable *> & operands,
    std::vector<std::unique_ptr<ThreeAddressCodeVariable>> results)
    : operands_(operands),
      operation_(std::move(operation)),
      results_(std::move(results))
{
  check_operands(this->operation(), operands);
  check_results(this->operation(), results_);
}

void
ThreeAddressCode::convert(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const Variable *> & operands)
{
  check_operands(operation, operands);

  results_.clear();
  operands_ = operands;
  operation_ = operation.copy();

  auto names = create_names(operation.nresults());
  create_results(operation, names);
}

void
ThreeAddressCode::replace(
    const rvsdg::SimpleOperation & operation,
    const std::vector<const Variable *> & operands)
{
  check_operands(operation, operands);
  check_results(operation, results_);

  operands_ = operands;
  operation_ = operation.copy();
}

std::string
ThreeAddressCode::ToAscii(const jlm::llvm::ThreeAddressCode & threeAddressCode)
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
