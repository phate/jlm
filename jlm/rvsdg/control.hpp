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

#include <unordered_map>

#include <inttypes.h>

namespace jlm::rvsdg
{

/* control type */

class ctltype final : public jlm::rvsdg::statetype
{
public:
  virtual ~ctltype() noexcept;

  ctltype(size_t nalternatives);

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  std::size_t
  ComputeHash() const noexcept override;

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
  static std::shared_ptr<const ctltype>
  Create(std::size_t nalternatives);

private:
  size_t nalternatives_;
};

static inline bool
is_ctltype(const jlm::rvsdg::Type & type) noexcept
{
  return dynamic_cast<const ctltype *>(&type) != nullptr;
}

/* control value representation */

class ctlvalue_repr
{
public:
  ctlvalue_repr(size_t alternative, size_t nalternatives);

  inline bool
  operator==(const ctlvalue_repr & other) const noexcept
  {
    return alternative_ == other.alternative_ && nalternatives_ == other.nalternatives_;
  }

  inline bool
  operator!=(const ctlvalue_repr & other) const noexcept
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

struct ctltype_of_value
{
  std::shared_ptr<const ctltype>
  operator()(const ctlvalue_repr & repr) const
  {
    return ctltype::Create(repr.nalternatives());
  }
};

struct ctlformat_value
{
  std::string
  operator()(const ctlvalue_repr & repr) const
  {
    return jlm::util::strfmt("CTL(", repr.alternative(), ")");
  }
};

typedef domain_const_op<ctltype, ctlvalue_repr, ctlformat_value, ctltype_of_value> ctlconstant_op;

static inline bool
is_ctlconstant_op(const jlm::rvsdg::operation & op) noexcept
{
  return dynamic_cast<const ctlconstant_op *>(&op) != nullptr;
}

static inline const ctlconstant_op &
to_ctlconstant_op(const jlm::rvsdg::operation & op) noexcept
{
  JLM_ASSERT(is_ctlconstant_op(op));
  return *static_cast<const ctlconstant_op *>(&op);
}

/* match operator */

class match_op final : public jlm::rvsdg::unary_op
{
  typedef std::unordered_map<uint64_t, uint64_t>::const_iterator const_iterator;

public:
  virtual ~match_op() noexcept;

  match_op(
      size_t nbits,
      const std::unordered_map<uint64_t, uint64_t> & mapping,
      uint64_t default_alternative,
      size_t nalternatives);

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept override;

  virtual jlm::rvsdg::output *
  reduce_operand(unop_reduction_path_t path, jlm::rvsdg::output * arg) const override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  inline uint64_t
  nalternatives() const noexcept
  {
    return std::static_pointer_cast<const ctltype>(result(0))->nalternatives();
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
    return std::static_pointer_cast<const bittype>(argument(0))->nbits();
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

  static output *
  Create(
      output & predicate,
      const std::unordered_map<uint64_t, uint64_t> & mapping,
      uint64_t defaultAlternative,
      size_t numAlternatives)
  {
    auto bitType = CheckAndExtractBitType(predicate.type());

    match_op operation(bitType.nbits(), mapping, defaultAlternative, numAlternatives);
    return rvsdg::simple_node::create_normalized(predicate.region(), operation, { &predicate })[0];
  }

private:
  static const bittype &
  CheckAndExtractBitType(const rvsdg::Type & type)
  {
    if (auto bitType = dynamic_cast<const bittype *>(&type))
    {
      return *bitType;
    }

    throw util::type_error("bittype", type.debug_string());
  }

  uint64_t default_alternative_;
  std::unordered_map<uint64_t, uint64_t> mapping_;
};

jlm::rvsdg::output *
match(
    size_t nbits,
    const std::unordered_map<uint64_t, uint64_t> & mapping,
    uint64_t default_alternative,
    size_t nalternatives,
    jlm::rvsdg::output * operand);

// declare explicit instantiation
extern template class domain_const_op<ctltype, ctlvalue_repr, ctlformat_value, ctltype_of_value>;

static inline const match_op &
to_match_op(const jlm::rvsdg::operation & op) noexcept
{
  JLM_ASSERT(is<match_op>(op));
  return *static_cast<const match_op *>(&op);
}

jlm::rvsdg::output *
control_constant(rvsdg::Region * region, size_t nalternatives, size_t alternative);

static inline jlm::rvsdg::output *
control_false(rvsdg::Region * region)
{
  return control_constant(region, 2, 0);
}

static inline jlm::rvsdg::output *
control_true(rvsdg::Region * region)
{
  return control_constant(region, 2, 1);
}

}

#endif
