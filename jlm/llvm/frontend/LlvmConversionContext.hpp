/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMCONVERSIONCONTEXT_HPP
#define JLM_LLVM_FRONTEND_LLVMCONVERSIONCONTEXT_HPP

#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/TypeConverter.hpp>

#include <llvm/IR/DerivedTypes.h>

#include <unordered_map>

namespace llvm
{
class BasicBlock;
class Function;
class Value;
}

namespace jlm::llvm
{

class ControlFlowGraph;
class ControlFlowGraphNode;
class InterProceduralGraphModule;
class Variable;

using BasicBlockMap = util::BijectiveMap<const ::llvm::BasicBlock *, BasicBlock *>;

class context final
{
public:
  explicit context(InterProceduralGraphModule & im)
      : module_(im),
        node_(nullptr),
        iostate_(nullptr),
        memory_state_(nullptr)
  {}

  const llvm::Variable *
  result() const noexcept
  {
    return result_;
  }

  inline void
  set_result(const llvm::Variable * result)
  {
    result_ = result;
  }

  llvm::Variable *
  iostate() const noexcept
  {
    return iostate_;
  }

  void
  set_iostate(llvm::Variable * state)
  {
    iostate_ = state;
  }

  inline llvm::Variable *
  memory_state() const noexcept
  {
    return memory_state_;
  }

  inline void
  set_memory_state(llvm::Variable * state)
  {
    memory_state_ = state;
  }

  inline bool
  has(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.HasKey(bb);
  }

  inline bool
  has(BasicBlock * bb) const noexcept
  {
    return bbmap_.HasValue(bb);
  }

  inline BasicBlock *
  get(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.LookupKey(bb);
  }

  inline const ::llvm::BasicBlock *
  get(BasicBlock * bb) const noexcept
  {
    return bbmap_.LookupValue(bb);
  }

  inline void
  set_basic_block_map(BasicBlockMap bbmap)
  {
    bbmap_ = std::move(bbmap);
  }

  inline bool
  has_value(const ::llvm::Value * value) const noexcept
  {
    return vmap_.find(value) != vmap_.end();
  }

  inline const llvm::Variable *
  lookup_value(const ::llvm::Value * value) const noexcept
  {
    JLM_ASSERT(has_value(value));
    return vmap_.find(value)->second;
  }

  inline void
  insert_value(const ::llvm::Value * value, const llvm::Variable * variable)
  {
    JLM_ASSERT(!has_value(value));
    vmap_[value] = variable;
  }

  [[nodiscard]] InterProceduralGraphModule &
  module() const noexcept
  {
    return module_;
  }

  inline void
  set_node(InterProceduralGraphNode * node) noexcept
  {
    node_ = node;
  }

  inline InterProceduralGraphNode *
  node() const noexcept
  {
    return node_;
  }

  TypeConverter &
  GetTypeConverter() noexcept
  {
    return TypeConverter_;
  }

private:
  InterProceduralGraphModule & module_;
  BasicBlockMap bbmap_;
  InterProceduralGraphNode * node_;
  const llvm::Variable * result_{};
  llvm::Variable * iostate_;
  llvm::Variable * memory_state_;
  std::unordered_map<const ::llvm::Value *, const llvm::Variable *> vmap_;
  TypeConverter TypeConverter_;
};

}

#endif
