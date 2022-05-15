/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP
#define JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP

#include <jlm/opt/alias-analyses/MemoryStateEncoder.hpp>

namespace jlm::aa {

/** \brief Basic memory state encoder
 *
 * The key idea of the basic memory state encoder is that \b all memory states are routed through \b all structural
 * nodes irregardless of whether these states are required by any simple nodes within the structural nodes. This
 * strategy ensures that the state of a memory location is always present for encoding while avoiding the complexity of
 * an additional analysis for determining the required routing path of the states. The drawback is that
 * a lot of states are routed through structural nodes where they are not needed, potentially leading to a significant
 * runtime of the encoder for bigger RVSDGs.
 *
 * @see MemoryStateEncoder
 */
class BasicEncoder final : public MemoryStateEncoder {
public:
  class Context;

  ~BasicEncoder() override;

  explicit
  BasicEncoder(PointsToGraph &pointsToGraph);

  BasicEncoder(const BasicEncoder &) = delete;

  BasicEncoder(BasicEncoder &&) = delete;

  BasicEncoder &
  operator=(const BasicEncoder &) = delete;

  BasicEncoder &
  operator=(BasicEncoder &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept
  {
    return PointsToGraph_;
  }

  void
  Encode(
    RvsdgModule & rvsdgModule,
    const StatisticsDescriptor & statisticsDescriptor) override;

  static void
  Encode(
    PointsToGraph & pointsToGraph,
    RvsdgModule & rvsdgModule,
    const StatisticsDescriptor & statisticsDescriptor);

private:
  void
  EncodeAlloca(const jive::simple_node &allocaNode) override;

  void
  EncodeMalloc(const jive::simple_node &mallocNode) override;

  void
  EncodeLoad(const LoadNode & loadNode) override;

  void
  EncodeStore(const StoreNode & storeNode) override;

  void
  EncodeFree(const jive::simple_node &freeNode) override;

  void
  EncodeCall(const CallNode & callNode) override;

  void
  EncodeMemcpy(const jive::simple_node &memcpyNode) override;

  void
  Encode(const lambda::node &lambda) override;

  void
  Encode(const phi::node &phi) override;

  void
  Encode(const delta::node &delta) override;

  void
  Encode(jive::gamma_node &gamma) override;

  void
  Encode(jive::theta_node &theta) override;

  static void
  UnlinkUnknownMemoryNode(PointsToGraph &pointsToGraph);

  PointsToGraph& PointsToGraph_;
  std::unique_ptr <Context> Context_;
};

}

#endif //JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP