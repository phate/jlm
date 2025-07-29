/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"

namespace jlm::tests
{

GraphImport &
GraphImport::Copy(rvsdg::Region & region, rvsdg::StructuralInput *)
{
  return GraphImport::Create(*region.graph(), Type(), Name());
}

GraphExport &
GraphExport::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
{
  JLM_ASSERT(output == nullptr);
  return GraphExport::Create(origin, Name());
}

TestUnaryOperation::~TestUnaryOperation() noexcept = default;

bool
TestUnaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const TestUnaryOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

rvsdg::unop_reduction_path_t
TestUnaryOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
TestUnaryOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  return nullptr;
}

std::string
TestUnaryOperation::debug_string() const
{
  return "TestUnaryOperation";
}

std::unique_ptr<rvsdg::Operation>
TestUnaryOperation::copy() const
{
  return std::make_unique<TestUnaryOperation>(*this);
}

TestBinaryOperation::~TestBinaryOperation() noexcept = default;

bool
TestBinaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const TestBinaryOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

rvsdg::binop_reduction_path_t
TestBinaryOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
TestBinaryOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
TestBinaryOperation::flags() const noexcept
{
  return flags_;
}

std::string
TestBinaryOperation::debug_string() const
{
  return "TestBinaryOperation";
}

std::unique_ptr<rvsdg::Operation>
TestBinaryOperation::copy() const
{
  return std::make_unique<TestBinaryOperation>(*this);
}

TestOperation::~TestOperation() noexcept = default;

bool
TestOperation::operator==(const Operation & o) const noexcept
{
  auto other = dynamic_cast<const TestOperation *>(&o);
  if (!other)
    return false;

  if (narguments() != other->narguments() || nresults() != other->nresults())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (argument(n) != other->argument(n))
      return false;
  }

  for (size_t n = 0; n < nresults(); n++)
  {
    if (result(n) != other->result(n))
      return false;
  }

  return true;
}

std::string
TestOperation::debug_string() const
{
  return "TestOperation";
}

std::unique_ptr<rvsdg::Operation>
TestOperation::copy() const
{
  return std::make_unique<TestOperation>(*this);
}

TestStructuralOperation::~TestStructuralOperation() noexcept = default;

std::string
TestStructuralOperation::debug_string() const
{
  return "TestStructuralOperation";
}

std::unique_ptr<rvsdg::Operation>
TestStructuralOperation::copy() const
{
  return std::make_unique<TestStructuralOperation>(*this);
}

TestStructuralNode::~TestStructuralNode() noexcept = default;

[[nodiscard]] const TestStructuralOperation &
TestStructuralNode::GetOperation() const noexcept
{
  static TestStructuralOperation singleton;
  return singleton;
}

TestStructuralNode *
TestStructuralNode::copy(rvsdg::Region * parent, rvsdg::SubstitutionMap & smap) const
{
  auto node = create(parent, nsubregions());

  /* copy inputs */
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto origin = smap.lookup(input(n)->origin());
    auto neworigin = origin ? origin : input(n)->origin();
    auto new_input = rvsdg::StructuralInput::create(node, neworigin, input(n)->Type());
    smap.insert(input(n), new_input);
  }

  /* copy outputs */
  for (size_t n = 0; n < noutputs(); n++)
  {
    auto new_output = rvsdg::StructuralOutput::create(node, output(n)->Type());
    smap.insert(output(n), new_output);
  }

  /* copy regions */
  for (size_t n = 0; n < nsubregions(); n++)
    subregion(n)->copy(node->subregion(n), smap, true, true);

  return node;
}

TestStructuralNode::InputVar
TestStructuralNode::AddInput(rvsdg::Output & origin)
{
  const auto input = add_input(
      std::unique_ptr<StructuralNodeInput>(new StructuralNodeInput(*this, origin, origin.Type())));
  return { input, {} };
}

TestStructuralNode::InputVar
TestStructuralNode::AddInputWithArguments(rvsdg::Output & origin)
{
  auto inputVar = AddInput(origin);

  for (auto & subregion : Subregions())
  {
    const auto argument = &StructuralNodeArgument::Create(
        subregion,
        *util::AssertedCast<StructuralNodeInput>(inputVar.input));
    inputVar.argument.push_back(argument);
  }

  return inputVar;
}

TestStructuralNode::InputVar
TestStructuralNode::AddArguments(const std::shared_ptr<const rvsdg::Type> & type)
{
  std::vector<rvsdg::Output *> arguments;
  for (auto & subregion : Subregions())
  {
    const auto argument = &StructuralNodeArgument::Create(subregion, type);
    arguments.push_back(argument);
  }

  return { nullptr, std::move(arguments) };
}

TestStructuralNode::OutputVar
TestStructuralNode::AddOutput(std::shared_ptr<const rvsdg::Type> type)
{
  const auto output = add_output(std::make_unique<rvsdg::StructuralOutput>(this, std::move(type)));
  return { output, {} };
}

TestStructuralNode::OutputVar
TestStructuralNode::AddOutputWithResults(const std::vector<rvsdg::Output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::Error("Insufficient number of origins.");

  size_t n = 0;
  auto outputVar = AddOutput(origins[0]->Type());
  for (auto & subregion : Subregions())
  {
    const auto origin = origins[n++];
    const auto result = &rvsdg::RegionResult::Create(
        subregion,
        *origin,
        util::AssertedCast<rvsdg::StructuralOutput>(outputVar.output),
        origin->Type());
    outputVar.result.push_back(result);
  }

  return outputVar;
}

TestStructuralNode::OutputVar
TestStructuralNode::AddResults(const std::vector<rvsdg::Output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::Error("Insufficient number of origins.");

  size_t n = 0;
  std::vector<rvsdg::Input *> results;
  for (auto & subregion : Subregions())
  {
    const auto origin = origins[n++];
    const auto result = &rvsdg::RegionResult::Create(subregion, *origin, nullptr, origin->Type());
    results.push_back(result);
  }

  return { nullptr, std::move(results) };
}

StructuralNodeInput::~StructuralNodeInput() noexcept = default;

StructuralNodeArgument::~StructuralNodeArgument() noexcept = default;

StructuralNodeArgument &
StructuralNodeArgument::Copy(rvsdg::Region & region, rvsdg::StructuralInput * input)
{
  auto structuralNodeInput = util::AssertedCast<StructuralNodeInput>(input);
  return structuralNodeInput != nullptr ? Create(region, *structuralNodeInput)
                                        : Create(region, Type());
}

}
