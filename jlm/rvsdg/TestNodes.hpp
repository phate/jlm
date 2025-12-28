/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TESTNODES_HPP
#define JLM_RVSDG_TESTNODES_HPP

#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jlm::rvsdg
{

class TestStructuralOperation final : public StructuralOperation
{
public:
  ~TestStructuralOperation() noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

class TestStructuralNode final : public StructuralNode
{
public:
  ~TestStructuralNode() noexcept override;

private:
  TestStructuralNode(Region * parent, size_t numSubregions)
      : StructuralNode(parent, numSubregions)
  {}

public:
  /**
   * \brief A variable routed in a \ref TestStructuralNode
   */
  struct InputVar
  {
    StructuralInput * input{};
    std::vector<RegionArgument *> argument{};
  };

  /**
   * \brief A variable routed out of a \ref TestStructuralNode
   */
  struct OutputVar
  {
    StructuralOutput * output{};
    std::vector<RegionResult *> result{};
  };

  /**
   * Add an input WITHOUT subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return The created input variable.
   */
  StructuralInput &
  addInputOnly(Output & origin);

  /**
   * Add an input WITH subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return Description of input variable.
   */
  InputVar
  addInputWithArguments(Output & origin);

  /**
   * Removes the input with the given \p index,
   * and any subregion arguments associated with it.
   * All such arguments must be dead.
   * @param index the index of the input to remove.
   */
  void
  removeInputAndArguments(size_t index);

  /**
   * Add subregion arguments WITHOUT an input to a \ref TestStructuralNode.
   * @param type The argument type
   * @return Description of input variable.
   */
  InputVar
  addArguments(const std::shared_ptr<const Type> & type);

  /**
   * Add an output WITHOUT subregion results to a \ref TestStructuralNode.
   *
   * @param type The output type
   * @return The created output variable.
   */
  StructuralOutput &
  addOutputOnly(std::shared_ptr<const Type> type);

  /**
   * Add an output WITH subregion results to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  addOutputWithResults(const std::vector<Output *> & origins);

  /**
   * Removes the output with the given \p index,
   * and any subregion results associated with it.
   * @param index the index of the output to remove.
   */
  void
  removeOutputAndResults(size_t index);

  /**
   * Add subregion results WITHOUT output to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  addResults(const std::vector<Output *> & origins);

  [[nodiscard]] const TestStructuralOperation &
  GetOperation() const noexcept override;

  static TestStructuralNode *
  create(Region * parent, const size_t numSubregions)
  {
    return new TestStructuralNode(parent, numSubregions);
  }

  TestStructuralNode *
  copy(Region * region, SubstitutionMap & smap) const override;
};

}

#endif
