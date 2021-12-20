/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP
#define JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP

#include <jlm/opt/alias-analyses/encoders.hpp>

namespace jlm {
namespace aa
{

/** \brief BasicEncoder class
*/
class BasicEncoder final : public MemoryStateEncoder {
public:
  class Context;

  ~BasicEncoder() override;

  explicit
  BasicEncoder(jlm::aa::ptg &ptg);

  BasicEncoder(const BasicEncoder &) = delete;

  BasicEncoder(BasicEncoder &&) = delete;

  BasicEncoder &
  operator=(const BasicEncoder &) = delete;

  BasicEncoder &
  operator=(BasicEncoder &&) = delete;

  const jlm::aa::ptg &
  Ptg() const noexcept
  {
    return Ptg_;
  }

  void
  Encode(rvsdg_module &module) override;

  static void
  Encode(
    jlm::aa::ptg &ptg,
    rvsdg_module &module);

private:
  void
  EncodeAlloca(const jive::simple_node &node) override;

  void
  EncodeMalloc(const jive::simple_node &node) override;

  void
  EncodeLoad(const jive::simple_node &node) override;

  void
  EncodeStore(const jive::simple_node &node) override;

  void
  EncodeFree(const jive::simple_node &node) override;

  void
  EncodeCall(const jive::simple_node &node) override;

  void
  EncodeMemcpy(const jive::simple_node &node) override;

  void
  Encode(const lambda::node &lambda) override;

  void
  Encode(const jive::phi::node &phi) override;

  void
  Encode(const delta::node &delta) override;

  void
  Encode(jive::gamma_node &gamma) override;

  void
  Encode(jive::theta_node &theta) override;

  static void
  UnlinkMemUnknown(jlm::aa::ptg &ptg);

  jlm::aa::ptg &Ptg_;
  std::unique_ptr <Context> Context_;
};

}}

#endif //JLM_OPT_ALIAS_ANALYSES_BASICENCODER_HPP