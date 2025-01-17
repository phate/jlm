/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RHLS2FIRRTL_JSON_HLS_HPP
#define JLM_BACKEND_HLS_RHLS2FIRRTL_JSON_HLS_HPP

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

namespace jlm::hls
{

class JsonHLS : public BaseHLS
{
  std::string
  extension() override
  {
    return ".json";
  }

  std::string
  GetText(llvm::RvsdgModule & rm) override;

private:
};

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RHLS2FIRRTL_JSON_HLS_HPP
