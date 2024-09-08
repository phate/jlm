/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP

#include <memory>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

class CallNode;
class RvsdgModule;

namespace delta
{
class node;
}

namespace lambda
{
class node;
}

namespace phi
{
class node;
}

class LoadNode;
class StoreNode;

namespace aa
{

class MemoryNodeProvisioning;

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
  class Context;

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
      RvsdgModule & rvsdgModule,
      const MemoryNodeProvisioning & provisioning,
      util::StatisticsCollector & statisticsCollector);

private:
  void
  EncodeRegion(rvsdg::region & region);

  void
  EncodeStructuralNode(rvsdg::structural_node & structuralNode);

  void
  EncodeSimpleNode(const rvsdg::simple_node & simpleNode);

  void
  EncodeAlloca(const rvsdg::simple_node & allocaNode);

  void
  EncodeMalloc(const rvsdg::simple_node & mallocNode);

  void
  EncodeLoad(const LoadNode & loadNode);

  void
  EncodeStore(const StoreNode & storeNode);

  void
  EncodeFree(const rvsdg::simple_node & freeNode);

  void
  EncodeCall(const CallNode & callNode);

  void
  EncodeCallEntry(const CallNode & callNode);

  void
  EncodeCallExit(const CallNode & callNode);

  void
  EncodeMemcpy(const rvsdg::simple_node & memcpyNode);

  void
  EncodeLambda(const lambda::node & lambda);

  void
  EncodeLambdaEntry(const lambda::node & lambdaNode);

  void
  EncodeLambdaExit(const lambda::node & lambdaNode);

  void
  EncodePhi(const phi::node & phiNode);

  void
  EncodeDelta(const delta::node & deltaNode);

  void
  EncodeGamma(rvsdg::GammaNode & gammaNode);

  void
  EncodeGammaEntry(rvsdg::GammaNode & gammaNode);

  void
  EncodeGammaExit(rvsdg::GammaNode & gammaNode);

  void
  EncodeTheta(rvsdg::theta_node & thetaNode);

  std::vector<rvsdg::theta_output *>
  EncodeThetaEntry(rvsdg::theta_node & thetaNode);

  void
  EncodeThetaExit(
      rvsdg::theta_node & thetaNode,
      const std::vector<rvsdg::theta_output *> & thetaStateOutputs);

  /**
   * Replace \p loadNode with a new copy that takes the provided \p memoryStates. All users of the
   * outputs of \p loadNode are redirected to the respective outputs of the newly created copy.
   *
   * @param loadNode A LoadNode.
   * @param memoryStates The memory states the new LoadNode should consume.
   *
   * @return The newly created LoadNode.
   */
  [[nodiscard]] static LoadNode &
  ReplaceLoadNode(const LoadNode & loadNode, const std::vector<rvsdg::output *> & memoryStates);

  /**
   * Replace \p storeNode with a new copy that takes the provided \p memoryStates. All users of the
   * outputs of \p storeNode are redirected to the respective outputs of the newly created copy.
   *
   * @param storeNode A StoreNode.
   * @param memoryStates The memory states the new StoreNode should consume.
   *
   * @return The newly created StoreNode.
   */
  [[nodiscard]] static StoreNode &
  ReplaceStoreNode(const StoreNode & storeNode, const std::vector<rvsdg::output *> & memoryStates);

  /**
   * Replace \p memcpyNode with a new copy that takes the provided \p memoryStates. All users of
   * the outputs of \p memcpyNode are redirected to the respective outputs of the newly created
   * copy.
   *
   * @param memcpyNode A rvsdg::simple_node representing a MemCpyOperation.
   * @param memoryStates The memory states the new memcpy node should consume.
   *
   * @return A vector with the memory states of the newly created copy.
   */
  [[nodiscard]] static std::vector<rvsdg::output *>
  ReplaceMemcpyNode(
      const rvsdg::simple_node & memcpyNode,
      const std::vector<rvsdg::output *> & memoryStates);

  /**
   * Determines whether \p simpleNode should be handled by the MemoryStateEncoder.
   *
   * @param simpleNode A simple_node.
   * @return True, if \p simpleNode should be handled, otherwise false.
   */
  [[nodiscard]] static bool
  ShouldHandle(const rvsdg::simple_node & simpleNode) noexcept;

  std::unique_ptr<Context> Context_;
};

}
}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
