/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP

#include <memory>
#include <vector>

namespace jlm::rvsdg
{
class GammaNode;
class LambdaNode;
class Output;
class Region;
class RvsdgModule;
class SimpleNode;
class ThetaNode;
}

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm::aa
{

class ModRefSummary;

/** \brief Memory State Encoder
 *
 * A memory state encoder encodes a points-to graph in the RVSDG. The basic idea is that there
 * exists a one-to-one correspondence between memory nodes in the points-to graph and memory states
 * in the RVSDG, i.e., for each memory node in the points-to graph, there exists a memory state edge
 * in the RVSDG. A memory state encoder routes these state edges through the RVSDG's structural
 * nodes and ensures that simple nodes operating on a memory location represented by a corresponding
 * memory node in the points-to graph are sequentialized with the respective memory state edge. For
 * example, a store node that modifies a global variable needs to have the respective state edge
 * that corresponds to its memory location routed through it, i.e., the store node is sequentialized
 * by this state edge. Such an encoding ensures that the ordering of side-effecting operations
 * touching on the same memory locations is preserved, while rendering operations independent that
 * are not operating on the same memory locations.
 */
class MemoryStateEncoder final
{
public:
  struct MemoryStateTypeCounters;
  class StateMap;

  ~MemoryStateEncoder() noexcept;

  MemoryStateEncoder();

  MemoryStateEncoder(const MemoryStateEncoder &) = delete;

  MemoryStateEncoder(MemoryStateEncoder &&) = delete;

  MemoryStateEncoder &
  operator=(const MemoryStateEncoder &) = delete;

  MemoryStateEncoder &
  operator=(MemoryStateEncoder &&) = delete;

  void
  Encode(
      rvsdg::RvsdgModule & rvsdgModule,
      const ModRefSummary & modRefSummary,
      util::StatisticsCollector & statisticsCollector);

private:
  void
  EncodeRegion(rvsdg::Region & region, StateMap & stateMap) const;

  void
  EncodeAlloca(const rvsdg::SimpleNode & allocaNode, StateMap & stateMap) const;

  void
  EncodeMalloc(const rvsdg::SimpleNode & mallocNode, StateMap & stateMap) const;

  void
  EncodeLoad(const rvsdg::SimpleNode & node, StateMap & stateMap) const;

  void
  EncodeStore(const rvsdg::SimpleNode & node, StateMap & stateMap) const;

  void
  EncodeFree(const rvsdg::SimpleNode & freeNode, StateMap & stateMap) const;

  void
  EncodeCall(const rvsdg::SimpleNode & callNode, StateMap & stateMap) const;

  void
  EncodeMemcpy(const rvsdg::SimpleNode & memcpyNode, StateMap & stateMap) const;

  void
  EncodeMemset(const rvsdg::SimpleNode & memsetNode, StateMap & stateMap) const;

  void
  EncodeMemmove(const rvsdg::SimpleNode & memmoveNode, StateMap & stateMap) const;

  void
  EncodeLambda(const rvsdg::LambdaNode & lambda) const;

  void
  EncodeGamma(rvsdg::GammaNode & gammaNode, StateMap & stateMap) const;

  void
  EncodeTheta(rvsdg::ThetaNode & thetaNode, StateMap & stateMap) const;

  std::unique_ptr<MemoryStateTypeCounters>
  gatherStatistics(const rvsdg::Region & region) const;

  /**
   * Replace \p loadNode with a new copy that takes the provided \p memoryStates. All users of the
   * outputs of \p loadNode are redirected to the respective outputs of the newly created copy.
   *
   * @param node A LoadNode.
   * @param memoryStates The memory states the new LoadNode should consume.
   *
   * @return The newly created LoadNode.
   */
  [[nodiscard]] static rvsdg::SimpleNode &
  ReplaceLoadNode(
      const rvsdg::SimpleNode & node,
      const std::vector<rvsdg::Output *> & memoryStates);

  /**
   * Replace \p storeNode with a new copy that takes the provided \p memoryStates. All users of the
   * outputs of \p storeNode are redirected to the respective outputs of the newly created copy.
   *
   * @param node A StoreNode.
   * @param memoryStates The memory states the new StoreNode should consume.
   *
   * @return The newly created StoreNode.
   */
  [[nodiscard]] static rvsdg::SimpleNode &
  ReplaceStoreNode(
      const rvsdg::SimpleNode & node,
      const std::vector<rvsdg::Output *> & memoryStates);

  /**
   * Replace \p memCpyNode with a new copy that takes the provided \p memoryStates. All users of
   * the outputs of \p memCpyNode are redirected to the respective outputs of the newly created
   * copy.
   *
   * @param memCpyNode A rvsdg::SimpleNode representing a MemCpyOperation.
   * @param memoryStates The memory states the new \ref MemCpyOperation node should consume.
   *
   * @return A vector with the memory states of the newly created copy.
   */
  [[nodiscard]] static rvsdg::SimpleNode &
  ReplaceMemcpyNode(
      const rvsdg::SimpleNode & memCpyNode,
      const std::vector<rvsdg::Output *> & memoryStates);

  /**
   * Replace \p memsetNode with a new copy that takes the provided \p memoryStates. All users of
   * the outputs of \p memsetNode are redirected to the respective outputs of the newly created
   * copy.
   *
   * @param memsetNode A rvsdg::SimpleNode representing a MemSetOperation.
   * @param memoryStates The memory states the new memset node should consume.
   *
   * @return A vector with the memory states of the newly created copy.
   */
  [[nodiscard]] static rvsdg::SimpleNode &
  ReplaceMemsetNode(
      const rvsdg::SimpleNode & memsetNode,
      const std::vector<rvsdg::Output *> & memoryStates);

  /**
   * Replace \p memmoveNode with a new copy that takes the provided \p memoryStates. All users of
   * the outputs of \p memmoveNode are redirected to the respective outputs of the newly created
   * copy.
   *
   * @param memmoveNode A rvsdg::SimpleNode representing a \ref MemMoveOperation.
   * @param memoryStates The memory states the new \ref MemMoveOperation node should consume.
   *
   * @return A vector with the memory states of the newly created copy.
   */
  [[nodiscard]] static rvsdg::SimpleNode &
  ReplaceMemmoveNode(
      const rvsdg::SimpleNode & memmoveNode,
      const std::vector<rvsdg::Output *> & memoryStates);

  const ModRefSummary * modRefSummary_ = nullptr;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
