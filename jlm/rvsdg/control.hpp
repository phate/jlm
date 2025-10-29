/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_CONTROL_HPP
#define JLM_RVSDG_CONTROL_HPP

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/unary.hpp>
#include <jlm/util/strfmt.hpp>

#include <cstdint>
#include <unordered_map>

namespace jlm::rvsdg
{

class ControlType final : public Type
{
public:
  ~ControlType() noexcept override;

  explicit ControlType(size_t nalternatives);

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  std::size_t
  ComputeHash() const noexcept override;

  TypeKind
  Kind() const noexcept override;

  inline size_t
  nalternatives() const noexcept
  {
    return nalternatives_;
  }

  /**
   * \brief Instantiates control type
   *
   * \param nalternatives Number of alternatives
   *
   * \returns Control type instance
   *
   * Creates an instance of a control type capable of representing
   * the specified number of alternatives. The returned instance
   * will usually be a static singleton for the type.
   */
  static std::shared_ptr<const ControlType>
  Create(std::size_t nalternatives);

private:
  size_t nalternatives_;
};

class ControlValueRepresentation
{
public:
  ControlValueRepresentation(size_t alternative, size_t nalternatives);

  inline bool
  operator==(const ControlValueRepresentation & other) const noexcept
  {
    return alternative_ == other.alternative_ && nalternatives_ == other.nalternatives_;
  }

  inline bool
  operator!=(const ControlValueRepresentation & other) const noexcept
  {
    return !(*this == other);
  }

  inline size_t
  alternative() const noexcept
  {
    return alternative_;
  }

  inline size_t
  nalternatives() const noexcept
  {
    return nalternatives_;
  }

private:
  size_t alternative_;
  size_t nalternatives_;
};

/* control constant */

struct ControlValueRepresentationTypeOfValue
{
  std::shared_ptr<const ControlType>
  operator()(const ControlValueRepresentation & repr) const
  {
    return ControlType::Create(repr.nalternatives());
  }
};

struct ControlValueRepresentationFormatValue
{
  std::string
  operator()(const ControlValueRepresentation & repr) const
  {
    return jlm::util::strfmt("CTL(", repr.alternative(), ")");
  }
};

typedef DomainConstOperation<
    ControlType,
    ControlValueRepresentation,
    ControlValueRepresentationFormatValue,
    ControlValueRepresentationTypeOfValue>
    ControlConstantOperation;

/**
 * Match operator
 * Converts an n-bit integer input into a value of type ControlType.
 * The ControlType has a given number of alternative values, which are indexed starting at 0.
 * The match can represent any mapping from integers to alternatives, with a default alternative.
 * These alternatives represent the different outgoing edges from a basic block,
 * or the different regions of a gamma node.
 */
class MatchOperation final : public UnaryOperation
{
  typedef std::unordered_map<uint64_t, uint64_t>::const_iterator const_iterator;

public:
  ~MatchOperation() noexcept override;

  MatchOperation(
      size_t nbits,
      const std::unordered_map<uint64_t, uint64_t> & mapping,
      uint64_t default_alternative,
      size_t nalternatives);

  bool
  operator==(const Operation & other) const noexcept override;

  unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline uint64_t
  nalternatives() const noexcept
  {
    return std::static_pointer_cast<const ControlType>(result(0))->nalternatives();
  }

  inline uint64_t
  alternative(uint64_t value) const noexcept
  {
    auto it = mapping_.find(value);
    if (it != mapping_.end())
      return it->second;

    return default_alternative_;
  }

  inline uint64_t
  default_alternative() const noexcept
  {
    return default_alternative_;
  }

  inline size_t
  nbits() const noexcept
  {
    return std::static_pointer_cast<const BitType>(argument(0))->nbits();
  }

  inline const_iterator
  begin() const
  {
    return mapping_.begin();
  }

  inline const_iterator
  end() const
  {
    return mapping_.end();
  }

  static Output *
  Create(
      Output & predicate,
      const std::unordered_map<uint64_t, uint64_t> & mapping,
      uint64_t defaultAlternative,
      size_t numAlternatives)
  {
    auto bitType = CheckAndExtractBitType(*predicate.Type());
    return CreateOpNode<MatchOperation>(
               { &predicate },
               bitType.nbits(),
               mapping,
               defaultAlternative,
               numAlternatives)
        .output(0);
  }

private:
  static const BitType &
  CheckAndExtractBitType(const rvsdg::Type & type)
  {
    if (auto bitType = dynamic_cast<const BitType *>(&type))
    {
      return *bitType;
    }

    throw util::TypeError("BitType", type.debug_string());
  }

  uint64_t default_alternative_;
  std::unordered_map<uint64_t, uint64_t> mapping_;
};

jlm::rvsdg::Output *
match(
    size_t nbits,
    const std::unordered_map<uint64_t, uint64_t> & mapping,
    uint64_t default_alternative,
    size_t nalternatives,
    jlm::rvsdg::Output * operand);

// declare explicit instantiation
extern template class DomainConstOperation<
    ControlType,
    ControlValueRepresentation,
    ControlValueRepresentationFormatValue,
    ControlValueRepresentationTypeOfValue>;

jlm::rvsdg::Output *
control_constant(rvsdg::Region * region, size_t nalternatives, size_t alternative);

static inline jlm::rvsdg::Output *
control_false(rvsdg::Region * region)
{
  return control_constant(region, 2, 0);
}

static inline jlm::rvsdg::Output *
control_true(rvsdg::Region * region)
{
  return control_constant(region, 2, 1);
}

}

#endif
