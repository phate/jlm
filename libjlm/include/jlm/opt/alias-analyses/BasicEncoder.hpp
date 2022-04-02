/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP
#define JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP

#include <jlm/opt/alias-analyses/MemoryStateEncoder.hpp>

namespace jlm::aa {

/** \brief BasicEncoder class
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