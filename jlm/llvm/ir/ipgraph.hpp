/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_IPGRAPH_HPP
#define JLM_LLVM_IR_IPGRAPH_HPP

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>

#include <unordered_map>
#include <unordered_set>

namespace jlm::llvm
{

class InterProceduralGraphNode;

class InterProceduralGraph final
{
  using const_iterator = util::PtrIterator<
      const InterProceduralGraphNode,
      std::vector<std::unique_ptr<InterProceduralGraphNode>>::const_iterator>;

public:
  ~InterProceduralGraph() noexcept = default;

  InterProceduralGraph() noexcept = default;

  InterProceduralGraph(const InterProceduralGraph &) = delete;

  InterProceduralGraph(InterProceduralGraph &&) = delete;

  InterProceduralGraph &
  operator=(const InterProceduralGraph &) = delete;

  InterProceduralGraph &
  operator=(InterProceduralGraph &&) = delete;

  inline const_iterator
  begin() const noexcept
  {
    return const_iterator(nodes_.begin());
  }

  inline const_iterator
  end() const noexcept
  {
    return const_iterator(nodes_.end());
  }

  void
  add_node(std::unique_ptr<InterProceduralGraphNode> node);

  inline size_t
  nnodes() const noexcept
  {
    return nodes_.size();
  }

  [[nodiscard]] std::vector<std::unordered_set<const InterProceduralGraphNode *>>
  find_sccs() const;

  [[nodiscard]] const InterProceduralGraphNode *
  find(const std::string & name) const noexcept;

private:
  std::vector<std::unique_ptr<InterProceduralGraphNode>> nodes_;
};

class output;

class InterProceduralGraphNode
{
  typedef std::unordered_set<const InterProceduralGraphNode *>::const_iterator const_iterator;

public:
  virtual ~InterProceduralGraphNode() noexcept;

protected:
  explicit InterProceduralGraphNode(InterProceduralGraph & clg)
      : clg_(clg)
  {}

public:
  InterProceduralGraph &
  clg() const noexcept
  {
    return clg_;
  }

  void
  add_dependency(const InterProceduralGraphNode * dep)
  {
    dependencies_.insert(dep);
  }

  inline const_iterator
  begin() const
  {
    return dependencies_.begin();
  }

  inline const_iterator
  end() const
  {
    return dependencies_.end();
  }

  bool
  is_selfrecursive() const noexcept
  {
    if (dependencies_.find(this) != dependencies_.end())
      return true;

    return false;
  }

  virtual const std::string &
  name() const noexcept = 0;

  virtual const jlm::rvsdg::Type &
  type() const noexcept = 0;

  virtual std::shared_ptr<const jlm::rvsdg::Type>
  Type() const = 0;

  virtual const llvm::linkage &
  linkage() const noexcept = 0;

  virtual bool
  hasBody() const noexcept = 0;

private:
  InterProceduralGraph & clg_;
  std::unordered_set<const InterProceduralGraphNode *> dependencies_;
};

class FunctionNode final : public InterProceduralGraphNode
{
public:
  ~FunctionNode() noexcept override;

private:
  FunctionNode(
      InterProceduralGraph & clg,
      const std::string & name,
      std::shared_ptr<const rvsdg::FunctionType> type,
      const llvm::linkage & linkage,
      const AttributeSet & attributes)
      : InterProceduralGraphNode(clg),
        FunctionType_(type),
        name_(name),
        linkage_(linkage),
        attributes_(attributes)
  {}

public:
  inline llvm::ControlFlowGraph *
  cfg() const noexcept
  {
    return cfg_.get();
  }

  [[nodiscard]] const jlm::rvsdg::Type &
  type() const noexcept override;

  std::shared_ptr<const jlm::rvsdg::Type>
  Type() const override;

  const rvsdg::FunctionType &
  fcttype() const noexcept
  {
    return *FunctionType_;
  }

  const std::shared_ptr<const rvsdg::FunctionType> &
  GetFunctionType() const noexcept
  {
    return FunctionType_;
  }

  [[nodiscard]] const llvm::linkage &
  linkage() const noexcept override;

  [[nodiscard]] const std::string &
  name() const noexcept override;

  [[nodiscard]] bool
  hasBody() const noexcept override;

  const AttributeSet &
  attributes() const noexcept
  {
    return attributes_;
  }

  /**
  * \brief Adds \p cfg to the function node. If the function node already has a CFG, then it is
    replaced with \p cfg.
  **/
  void
  add_cfg(std::unique_ptr<ControlFlowGraph> cfg);

  static FunctionNode *
  create(
      InterProceduralGraph & ipg,
      const std::string & name,
      std::shared_ptr<const rvsdg::FunctionType> type,
      const llvm::linkage & linkage,
      const AttributeSet & attributes)
  {
    std::unique_ptr<FunctionNode> node(
        new FunctionNode(ipg, name, std::move(type), linkage, attributes));
    auto tmp = node.get();
    ipg.add_node(std::move(node));
    return tmp;
  }

  static FunctionNode *
  create(
      InterProceduralGraph & ipg,
      const std::string & name,
      std::shared_ptr<const rvsdg::FunctionType> type,
      const llvm::linkage & linkage)
  {
    return create(ipg, name, std::move(type), linkage, {});
  }

private:
  std::shared_ptr<const rvsdg::FunctionType> FunctionType_;
  std::string name_;
  llvm::linkage linkage_;
  AttributeSet attributes_;
  std::unique_ptr<ControlFlowGraph> cfg_;
};

class fctvariable final : public GlobalVariable
{
public:
  ~fctvariable() noexcept override;

  explicit fctvariable(FunctionNode * node)
      : GlobalVariable(node->Type(), node->name()),
        node_(node)
  {}

  FunctionNode *
  function() const noexcept
  {
    return node_;
  }

private:
  FunctionNode * node_;
};

class DataNodeInit final
{
public:
  explicit DataNodeInit(const Variable * value)
      : value_(value)
  {}

  explicit DataNodeInit(tacsvector_t tacs)
      : tacs_(std::move(tacs))
  {
    if (tacs_.empty())
      throw util::Error("Initialization cannot be empty.");

    auto & tac = tacs_.back();
    if (tac->nresults() != 1)
      throw util::Error("Last TAC of initialization needs exactly one result.");

    value_ = tac->result(0);
  }

  DataNodeInit(const DataNodeInit &) = delete;

  DataNodeInit(DataNodeInit && other) noexcept
      : tacs_(std::move(other.tacs_)),
        value_(other.value_)
  {}

  DataNodeInit &
  operator=(const DataNodeInit &) = delete;

  DataNodeInit &
  operator=(DataNodeInit &&) = delete;

  const Variable *
  value() const noexcept
  {
    return value_;
  }

  const tacsvector_t &
  tacs() const noexcept
  {
    return tacs_;
  }

private:
  tacsvector_t tacs_{};
  const Variable * value_;
};

class DataNode final : public InterProceduralGraphNode
{
public:
  ~DataNode() noexcept override;

private:
  DataNode(
      InterProceduralGraph & clg,
      const std::string & name,
      std::shared_ptr<const jlm::rvsdg::ValueType> valueType,
      const llvm::linkage & linkage,
      std::string section,
      bool constant)
      : InterProceduralGraphNode(clg),
        constant_(constant),
        name_(name),
        Section_(std::move(section)),
        linkage_(linkage),
        ValueType_(std::move(valueType))
  {}

public:
  [[nodiscard]] const PointerType &
  type() const noexcept override;

  std::shared_ptr<const jlm::rvsdg::Type>
  Type() const override;

  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::ValueType> &
  GetValueType() const noexcept
  {
    return ValueType_;
  }

  [[nodiscard]] const std::string &
  name() const noexcept override;

  [[nodiscard]] const llvm::linkage &
  linkage() const noexcept override;

  [[nodiscard]] bool
  hasBody() const noexcept override;

  inline bool
  constant() const noexcept
  {
    return constant_;
  }

  const std::string &
  Section() const noexcept
  {
    return Section_;
  }

  const DataNodeInit *
  initialization() const noexcept
  {
    return init_.get();
  }

  void
  set_initialization(std::unique_ptr<DataNodeInit> init)
  {
    if (!init)
      return;

    if (init->value()->type() != *GetValueType())
      throw util::Error("Invalid type.");

    init_ = std::move(init);
  }

  static DataNode *
  Create(
      InterProceduralGraph & clg,
      const std::string & name,
      std::shared_ptr<const jlm::rvsdg::ValueType> valueType,
      const llvm::linkage & linkage,
      std::string section,
      bool constant)
  {
    std::unique_ptr<DataNode> node(
        new DataNode(clg, name, std::move(valueType), linkage, std::move(section), constant));
    auto ptr = node.get();
    clg.add_node(std::move(node));
    return ptr;
  }

private:
  bool constant_;
  std::string name_;
  std::string Section_;
  llvm::linkage linkage_;
  std::shared_ptr<const jlm::rvsdg::ValueType> ValueType_;
  std::unique_ptr<DataNodeInit> init_;
};

}

#endif
