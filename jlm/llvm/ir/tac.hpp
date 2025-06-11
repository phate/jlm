/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_TAC_HPP
#define JLM_LLVM_IR_TAC_HPP

#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>

#include <list>
#include <memory>
#include <vector>

namespace jlm::llvm
{

class ThreeAddressCode;

class ThreeAddressCodeVariable final : public Variable
{
public:
  ~ThreeAddressCodeVariable() noexcept override;

  ThreeAddressCodeVariable(
      llvm::ThreeAddressCode * tac,
      std::shared_ptr<const jlm::rvsdg::Type> type,
      const std::string & name)
      : Variable(std::move(type), name),
        tac_(tac)
  {}

  [[nodiscard]] llvm::ThreeAddressCode *
  tac() const noexcept
  {
    return tac_;
  }

  static std::unique_ptr<ThreeAddressCodeVariable>
  create(
      llvm::ThreeAddressCode * tac,
      std::shared_ptr<const jlm::rvsdg::Type> type,
      const std::string & name)
  {
    return std::make_unique<ThreeAddressCodeVariable>(tac, std::move(type), name);
  }

private:
  llvm::ThreeAddressCode * tac_;
};

class ThreeAddressCode final
{
public:
  ~ThreeAddressCode() noexcept = default;

  ThreeAddressCode(
      const rvsdg::SimpleOperation & operation,
      const std::vector<const Variable *> & operands);

  ThreeAddressCode(
      const rvsdg::SimpleOperation & operation,
      const std::vector<const Variable *> & operands,
      const std::vector<std::string> & names);

  ThreeAddressCode(
      const rvsdg::SimpleOperation & operation,
      const std::vector<const Variable *> & operands,
      std::vector<std::unique_ptr<ThreeAddressCodeVariable>> results);

  ThreeAddressCode(const llvm::ThreeAddressCode &) = delete;

  ThreeAddressCode(llvm::ThreeAddressCode &&) = delete;

  ThreeAddressCode &
  operator=(const llvm::ThreeAddressCode &) = delete;

  ThreeAddressCode &
  operator=(llvm::ThreeAddressCode &&) = delete;

  inline const rvsdg::SimpleOperation &
  operation() const noexcept
  {
    return *static_cast<const rvsdg::SimpleOperation *>(operation_.get());
  }

  inline size_t
  noperands() const noexcept
  {
    return operands_.size();
  }

  inline const Variable *
  operand(size_t index) const noexcept
  {
    JLM_ASSERT(index < operands_.size());
    return operands_[index];
  }

  inline size_t
  nresults() const noexcept
  {
    return results_.size();
  }

  [[nodiscard]] const ThreeAddressCodeVariable *
  result(size_t index) const noexcept
  {
    JLM_ASSERT(index < results_.size());
    return results_[index].get();
  }

  /*
    FIXME: I am really not happy with this function exposing
    the results, but we need these results for the SSA destruction.
  */
  std::vector<std::unique_ptr<ThreeAddressCodeVariable>>
  results()
  {
    return std::move(results_);
  }

  void
  replace(const rvsdg::SimpleOperation & operation, const std::vector<const Variable *> & operands);

  void
  convert(const rvsdg::SimpleOperation & operation, const std::vector<const Variable *> & operands);

  static std::string
  ToAscii(const ThreeAddressCode & threeAddressCode);

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const rvsdg::SimpleOperation & operation, const std::vector<const Variable *> & operands)
  {
    return std::make_unique<llvm::ThreeAddressCode>(operation, operands);
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const rvsdg::SimpleOperation & operation,
      const std::vector<const Variable *> & operands,
      const std::vector<std::string> & names)
  {
    return std::make_unique<llvm::ThreeAddressCode>(operation, operands, names);
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const rvsdg::SimpleOperation & operation,
      const std::vector<const Variable *> & operands,
      std::vector<std::unique_ptr<ThreeAddressCodeVariable>> results)
  {
    return std::make_unique<llvm::ThreeAddressCode>(operation, operands, std::move(results));
  }

private:
  void
  create_results(const rvsdg::SimpleOperation & operation, const std::vector<std::string> & names)
  {
    JLM_ASSERT(names.size() == operation.nresults());

    for (size_t n = 0; n < operation.nresults(); n++)
    {
      auto & type = operation.result(n);
      results_.push_back(ThreeAddressCodeVariable::create(this, type, names[n]));
    }
  }

  static std::vector<std::string>
  create_names(size_t nnames)
  {
    static size_t c = 0;
    std::vector<std::string> names;
    for (size_t n = 0; n < nnames; n++)
      names.push_back(jlm::util::strfmt("tv", c++));

    return names;
  }

  std::vector<const Variable *> operands_;
  std::unique_ptr<rvsdg::Operation> operation_;
  std::vector<std::unique_ptr<ThreeAddressCodeVariable>> results_;
};

template<class T>
static inline bool
is(const llvm::ThreeAddressCode * tac)
{
  return tac && is<T>(tac->operation());
}

/* FIXME: Replace all occurences of tacsvector_t with taclist
  and then remove tacsvector_t.
*/
typedef std::vector<std::unique_ptr<llvm::ThreeAddressCode>> tacsvector_t;

/* taclist */

class taclist final
{
public:
  typedef std::list<ThreeAddressCode *>::const_iterator const_iterator;
  typedef std::list<ThreeAddressCode *>::const_reverse_iterator const_reverse_iterator;

  ~taclist();

  inline taclist()
  {}

  taclist(const taclist &) = delete;

  taclist(taclist && other)
      : tacs_(std::move(other.tacs_))
  {}

  taclist &
  operator=(const taclist &) = delete;

  taclist &
  operator=(taclist && other)
  {
    if (this == &other)
      return *this;

    for (const auto & tac : tacs_)
      delete tac;

    tacs_.clear();
    tacs_ = std::move(other.tacs_);

    return *this;
  }

  inline const_iterator
  begin() const noexcept
  {
    return tacs_.begin();
  }

  inline const_reverse_iterator
  rbegin() const noexcept
  {
    return tacs_.rbegin();
  }

  inline const_iterator
  end() const noexcept
  {
    return tacs_.end();
  }

  inline const_reverse_iterator
  rend() const noexcept
  {
    return tacs_.rend();
  }

  inline ThreeAddressCode *
  insert_before(const const_iterator & it, std::unique_ptr<llvm::ThreeAddressCode> tac)
  {
    return *tacs_.insert(it, tac.release());
  }

  inline void
  insert_before(const const_iterator & it, taclist & tl)
  {
    tacs_.insert(it, tl.begin(), tl.end());
  }

  inline void
  append_last(std::unique_ptr<llvm::ThreeAddressCode> tac)
  {
    tacs_.push_back(tac.release());
  }

  inline void
  append_first(std::unique_ptr<llvm::ThreeAddressCode> tac)
  {
    tacs_.push_front(tac.release());
  }

  inline void
  append_first(taclist & tl)
  {
    tacs_.insert(tacs_.begin(), tl.begin(), tl.end());
    tl.tacs_.clear();
  }

  inline size_t
  ntacs() const noexcept
  {
    return tacs_.size();
  }

  inline ThreeAddressCode *
  first() const noexcept
  {
    return ntacs() != 0 ? tacs_.front() : nullptr;
  }

  inline ThreeAddressCode *
  last() const noexcept
  {
    return ntacs() != 0 ? tacs_.back() : nullptr;
  }

  std::unique_ptr<ThreeAddressCode>
  pop_first() noexcept
  {
    std::unique_ptr<ThreeAddressCode> tac(tacs_.front());
    tacs_.pop_front();
    return tac;
  }

  std::unique_ptr<ThreeAddressCode>
  pop_last() noexcept
  {
    std::unique_ptr<ThreeAddressCode> tac(tacs_.back());
    tacs_.pop_back();
    return tac;
  }

  inline void
  drop_first()
  {
    delete tacs_.front();
    tacs_.pop_front();
  }

  inline void
  drop_last()
  {
    delete tacs_.back();
    tacs_.pop_back();
  }

private:
  std::list<ThreeAddressCode *> tacs_;
};

}

#endif
