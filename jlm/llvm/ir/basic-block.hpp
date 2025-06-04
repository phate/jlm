/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_BASIC_BLOCK_HPP
#define JLM_LLVM_IR_BASIC_BLOCK_HPP

#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/tac.hpp>

namespace jlm::llvm
{

class BasicBlock final : public cfg_node
{
public:
  ~BasicBlock() noexcept override;

private:
  explicit BasicBlock(llvm::cfg & cfg)
      : cfg_node(cfg)
  {}

  BasicBlock(const BasicBlock &) = delete;

  BasicBlock(BasicBlock &&) = delete;

  BasicBlock &
  operator=(const BasicBlock &) = delete;

  BasicBlock &
  operator=(BasicBlock &&) = delete;

public:
  const taclist &
  tacs() const noexcept
  {
    return tacs_;
  }

  taclist &
  tacs() noexcept
  {
    return tacs_;
  }

  inline taclist::const_iterator
  begin() const noexcept
  {
    return tacs_.begin();
  }

  inline taclist::const_reverse_iterator
  rbegin() const noexcept
  {
    return tacs_.rbegin();
  }

  inline taclist::const_iterator
  end() const noexcept
  {
    return tacs_.end();
  }

  inline taclist::const_reverse_iterator
  rend() const noexcept
  {
    return tacs_.rend();
  }

  inline size_t
  ntacs() const noexcept
  {
    return tacs_.ntacs();
  }

  inline tac *
  first() const noexcept
  {
    return tacs_.first();
  }

  inline tac *
  last() const noexcept
  {
    return tacs_.last();
  }

  inline void
  drop_first()
  {
    tacs_.drop_first();
  }

  inline void
  drop_last()
  {
    tacs_.drop_last();
  }

  llvm::tac *
  append_first(std::unique_ptr<llvm::tac> tac)
  {
    tacs_.append_first(std::move(tac));
    return tacs_.first();
  }

  void
  append_first(tacsvector_t & tacs)
  {
    for (auto it = tacs.rbegin(); it != tacs.rend(); it++)
      append_first(std::move(*it));
    tacs.clear();
  }

  void
  append_first(taclist & tl)
  {
    tacs_.append_first(tl);
  }

  llvm::tac *
  append_last(std::unique_ptr<llvm::tac> tac)
  {
    tacs_.append_last(std::move(tac));
    return tacs_.last();
  }

  void
  append_last(tacsvector_t & tacs)
  {
    for (auto & tac : tacs)
      append_last(std::move(tac));
    tacs.clear();
  }

  llvm::tac *
  insert_before(const taclist::const_iterator & it, std::unique_ptr<llvm::tac> tac)
  {
    return tacs_.insert_before(it, std::move(tac));
  }

  void
  insert_before(const taclist::const_iterator & it, tacsvector_t & tv)
  {
    for (auto & tac : tv)
      tacs_.insert_before(it, std::move(tac));
    tv.clear();
  }

  /**
   * Checks if there are any SsaPhiOperations in the basic block.
   * If any are present, they must be at the very beginning of the block.
   *
   * @return true if the basic block has phi operations, false otherwise
   */
  bool
  HasSsaPhiOperation() const;

  llvm::tac *
  insert_before_branch(std::unique_ptr<llvm::tac> tac);

  void
  insert_before_branch(tacsvector_t & tv);

  static BasicBlock *
  create(llvm::cfg & cfg);

private:
  taclist tacs_;
};

}

#endif
