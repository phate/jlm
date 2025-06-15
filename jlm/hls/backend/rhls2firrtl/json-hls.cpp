/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/json-hls.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>

namespace jlm::hls
{

std::string
JsonHLS::GetText(llvm::RvsdgModule & rm)
{
  std::ostringstream json;
  const auto & ln = *get_hls_lambda(rm);
  auto function_name = dynamic_cast<llvm::LlvmLambdaOperation &>(ln.GetOperation()).name();
  auto file_name = get_base_file_name(rm);
  json << "{\n";

  auto reg_args = get_reg_args(ln);
  auto reg_results = get_reg_results(ln);

  json << "\"addr_width\": " << GetPointerSizeInBits() << ",\n";
  json << "\"arguments\": [";
  for (size_t i = 0; i < reg_args.size(); ++i)
  {
    // don't generate ports for state edges
    if (rvsdg::is<rvsdg::StateType>(reg_args[i]->Type()))
      continue;
    if (i != 0)
    {
      json << ", ";
    }
    json << JlmSize(reg_args[i]->Type().get());
  }
  json << "],\n";
  json << "\"results\": [";
  for (size_t i = 0; i < reg_results.size(); ++i)
  {
    // don't generate ports for state edges
    if (rvsdg::is<rvsdg::StateType>(reg_results[i]->Type()))
      continue;
    if (i != 0)
    {
      json << ", ";
    }
    json << JlmSize(reg_results[i]->Type().get());
  }
  json << "],\n";
  // TODO: memory ports
  auto mem_reqs = get_mem_reqs(ln);
  auto mem_resps = get_mem_resps(ln);
  json << "\"mem\": [";
  for (size_t i = 0; i < mem_reqs.size(); ++i)
  {
    if (i != 0)
    {
      json << ", ";
    }
    auto req_bt = std::dynamic_pointer_cast<const BundleType>(mem_reqs[i]->Type());
    auto resp_bt = std::dynamic_pointer_cast<const BundleType>(mem_resps[i]->Type());
    auto size = JlmSize(&*resp_bt->get_element_type("data"));
    auto has_write = req_bt->get_element_type("write") != nullptr;
    json << "{ \"size\": " << size << ", \"has_write\": " << has_write << "}";
  }
  json << "]\n";
  json << "}\n";
  return json.str();
}

} // namespace jlm::hls
