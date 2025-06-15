/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_LLVM_IR_CFG_HPP
#define JLM_LLVM_IR_CFG_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

class clg_node;
class BasicBlock;
class InterProceduralGraphModule;
class ThreeAddressCode;

/** \brief Function argument
 */
class argument final : public Variable
{
public:
  ~argument() override;

  argument(
      const std::string & name,
      std::shared_ptr<const jlm::rvsdg::Type> type,
      const attributeset & attributes)
      : Variable(std::move(type), name),
        attributes_(attributes)
  {}

  argument(const std::string & name, std::shared_ptr<const jlm::rvsdg::Type> type)
      : Variable(std::move(type), name)
  {}

  argument(
      const std::string & name,
      std::unique_ptr<jlm::rvsdg::Type> type,
      const attributeset & attributes)
      : Variable(std::move(type), name),
        attributes_(attributes)
  {}

  const attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  static std::unique_ptr<argument>
  create(
      const std::string & name,
      std::shared_ptr<const jlm::rvsdg::Type> type,
      const attributeset & attributes)
  {
    return std::make_unique<argument>(name, std::move(type), attributes);
  }

  static std::unique_ptr<argument>
  create(const std::string & name, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    return create(name, std::move(type), {});
  }

private:
  attributeset attributes_;
};

class EntryNode final : public ControlFlowGraphNode
{
public:
  ~EntryNode() noexcept override;

  explicit EntryNode(ControlFlowGraph & cfg)
      : ControlFlowGraphNode(cfg)
  {}

  size_t
  narguments() const noexcept
  {
    return arguments_.size();
  }

  const llvm::argument *
  argument(size_t index) const
  {
    JLM_ASSERT(index < narguments());
    return arguments_[index].get();
  }

  llvm::argument *
  append_argument(std::unique_ptr<llvm::argument> arg)
  {
    arguments_.push_back(std::move(arg));
    return arguments_.back().get();
  }

  std::vector<llvm::argument *>
  arguments() const noexcept
  {
    std::vector<llvm::argument *> arguments;
    for (auto & argument : arguments_)
      arguments.push_back(argument.get());

    return arguments;
  }

private:
  std::vector<std::unique_ptr<llvm::argument>> arguments_;
};

class ExitNode final : public ControlFlowGraphNode
{
public:
  ~ExitNode() noexcept override;

  explicit ExitNode(ControlFlowGraph & cfg)
      : ControlFlowGraphNode(cfg)
  {}

  size_t
  nresults() const noexcept
  {
    return results_.size();
  }

  const Variable *
  result(size_t index) const
  {
    JLM_ASSERT(index < nresults());
    return results_[index];
  }

  inline void
  append_result(const Variable * v)
  {
    results_.push_back(v);
  }

  const std::vector<const Variable *>
  results() const noexcept
  {
    return results_;
  }

private:
  std::vector<const Variable *> results_;
};

class ControlFlowGraph final
{
  class iterator final
  {
  public:
    inline iterator(std::unordered_set<std::unique_ptr<BasicBlock>>::iterator it)
        : it_(it)
    {}

    inline bool
    operator==(const iterator & other) const noexcept
    {
      return it_ == other.it_;
    }

    inline bool
    operator!=(const iterator & other) const noexcept
    {
      return !(*this == other);
    }

    inline const iterator &
    operator++() noexcept
    {
      ++it_;
      return *this;
    }

    inline const iterator
    operator++(int) noexcept
    {
      iterator tmp(it_);
      it_++;
      return tmp;
    }

    inline BasicBlock *
    node() const noexcept
    {
      return it_->get();
    }

    inline BasicBlock &
    operator*() const noexcept
    {
      return *it_->get();
    }

    inline BasicBlock *
    operator->() const noexcept
    {
      return node();
    }

  private:
    std::unordered_set<std::unique_ptr<BasicBlock>>::iterator it_;
  };

  class const_iterator final
  {
  public:
    inline const_iterator(std::unordered_set<std::unique_ptr<BasicBlock>>::const_iterator it)
        : it_(it)
    {}

    inline bool
    operator==(const const_iterator & other) const noexcept
    {
      return it_ == other.it_;
    }

    inline bool
    operator!=(const const_iterator & other) const noexcept
    {
      return !(*this == other);
    }

    inline const const_iterator &
    operator++() noexcept
    {
      ++it_;
      return *this;
    }

    inline const const_iterator
    operator++(int) noexcept
    {
      const_iterator tmp(it_);
      it_++;
      return tmp;
    }

    inline const BasicBlock &
    operator*() noexcept
    {
      return *it_->get();
    }

    inline const BasicBlock *
    operator->() noexcept
    {
      return it_->get();
    }

  private:
    std::unordered_set<std::unique_ptr<BasicBlock>>::const_iterator it_;
  };

public:
  ~ControlFlowGraph() noexcept = default;

  explicit ControlFlowGraph(InterProceduralGraphModule & im);

  ControlFlowGraph(const ControlFlowGraph &) = delete;

  ControlFlowGraph(ControlFlowGraph &&) = delete;

  ControlFlowGraph &
  operator=(const ControlFlowGraph &) = delete;

  ControlFlowGraph &
  operator=(ControlFlowGraph &&) = delete;

  inline const_iterator
  begin() const
  {
    return const_iterator(nodes_.begin());
  }

  inline iterator
  begin()
  {
    return iterator(nodes_.begin());
  }

  inline const_iterator
  end() const
  {
    return const_iterator(nodes_.end());
  }

  inline iterator
  end()
  {
    return iterator(nodes_.end());
  }

  EntryNode *
  entry() const noexcept
  {
    return entry_.get();
  }

  ExitNode *
  exit() const noexcept
  {
    return exit_.get();
  }

  inline BasicBlock *
  add_node(std::unique_ptr<BasicBlock> bb)
  {
    auto tmp = bb.get();
    nodes_.insert(std::move(bb));
    return tmp;
  }

  ControlFlowGraph::iterator
  find_node(BasicBlock * bb)
  {
    std::unique_ptr<BasicBlock> up(bb);
    auto it = nodes_.find(up);
    up.release();
    return iterator(it);
  }

  static ControlFlowGraph::iterator
  remove_node(ControlFlowGraph::iterator & it);

  static ControlFlowGraph::iterator
  remove_node(BasicBlock * bb);

  inline size_t
  nnodes() const noexcept
  {
    return nodes_.size();
  }

  [[nodiscard]] InterProceduralGraphModule &
  module() const noexcept
  {
    return module_;
  }

  rvsdg::FunctionType
  fcttype() const
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments;
    for (size_t n = 0; n < entry()->narguments(); n++)
      arguments.push_back(entry()->argument(n)->Type());

    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results;
    for (size_t n = 0; n < exit()->nresults(); n++)
      results.push_back(exit()->result(n)->Type());

    return rvsdg::FunctionType(arguments, results);
  }

  static std::unique_ptr<ControlFlowGraph>
  create(InterProceduralGraphModule & im)
  {
    return std::make_unique<ControlFlowGraph>(im);
  }

  static std::string
  ToAscii(const ControlFlowGraph & controlFlowGraph);

private:
  static std::string
  ToAscii(const EntryNode & entryNode);

  static std::string
  ToAscii(const ExitNode & exitNode);

  static std::string
  ToAscii(
      const BasicBlock & basicBlock,
      const std::unordered_map<ControlFlowGraphNode *, std::string> & labels);

  static std::string
  CreateTargets(
      const ControlFlowGraphNode & node,
      const std::unordered_map<ControlFlowGraphNode *, std::string> & labels);

  static std::unordered_map<ControlFlowGraphNode *, std::string>
  CreateLabels(const std::vector<ControlFlowGraphNode *> & nodes);

  InterProceduralGraphModule & module_;
  std::unique_ptr<ExitNode> exit_;
  std::unique_ptr<EntryNode> entry_;
  std::unordered_set<std::unique_ptr<BasicBlock>> nodes_;
};

std::vector<ControlFlowGraphNode *>
postorder(const ControlFlowGraph & cfg);

std::vector<ControlFlowGraphNode *>
reverse_postorder(const ControlFlowGraph & cfg);

/** Order CFG nodes breadth-first
 *
 * Note, all nodes that are not dominated by the entry node are ignored.
 *
 * param cfg Control flow graph
 *
 * return A vector with all CFG nodes ordered breadth-first
 */
std::vector<ControlFlowGraphNode *>
breadth_first(const ControlFlowGraph & cfg);

size_t
ntacs(const ControlFlowGraph & cfg);

}

#endif
