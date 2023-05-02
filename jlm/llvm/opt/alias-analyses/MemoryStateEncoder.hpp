/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP

#include <memory>

namespace jive {

class gamma_node;
class region;
class simple_node;
class structural_node;
class theta_node;

}

namespace jlm {

class CallNode;
class RvsdgModule;
class StatisticsCollector;

namespace delta { class node; }
namespace lambda { class node; }
namespace phi { class node; }

class LoadNode;
class StoreNode;

namespace aa {

class MemoryNodeProvisioning;

/** \brief Memory State Encoder
 *
 * A memory state encoder encodes a points-to graph in the RVSDG. The basic idea is that there exists a one-to-one
 * correspondence between memory nodes in the points-to graph and memory states in the RVSDG, i.e., for each memory
 * node in the points-to graph, there exists a memory state edge in the RVSDG. A memory state encoder routes these
 * state edges through the RVSDG's structural nodes and ensures that simple nodes operating on a
 * memory location represented by a corresponding memory node in the points-to graph are sequentialized with the
 * respective memory state edge. For example, a store node that modifies a global variable needs to have the respective
 * state edge that corresponds to its memory location routed through it, i.e., the store node
 * is sequentialized by this state edge. Such an encoding ensures that the ordering of side-effecting operations
 * touching on the same memory locations is preserved, while rendering operations independent that are not operating on
 * the same memory locations.
 */
class MemoryStateEncoder final {
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
    StatisticsCollector & statisticsCollector);

private:
  void
  EncodeRegion(jive::region & region);

  void
  EncodeStructuralNode(jive::structural_node & structuralNode);

  void
  EncodeSimpleNode(const jive::simple_node & simpleNode);

  void
  EncodeAlloca(const jive::simple_node & allocaNode);

  void
  EncodeMalloc(const jive::simple_node & mallocNode);

  void
  EncodeLoad(const LoadNode & loadNode);

  void
  EncodeStore(const StoreNode & storeNode);

  void
  EncodeFree(const jive::simple_node & freeNode);

  void
  EncodeCall(const CallNode & callNode);

  void
  EncodeMemcpy(const jive::simple_node & memcpyNode);

  void
  EncodeLambda(const lambda::node & lambda);

  void
  EncodePhi(const phi::node & phi);

  void
  EncodeDelta(const delta::node & delta);

  void
  EncodeGamma(jive::gamma_node & gamma);

  void
  EncodeTheta(jive::theta_node & theta);

  std::unique_ptr <Context> Context_;
};

}}

#endif //JLM_LLVM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
