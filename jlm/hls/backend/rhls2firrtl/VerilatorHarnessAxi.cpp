/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/hls/backend/rhls2firrtl/VerilatorHarnessAxi.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>

#include <sstream>

namespace jlm::hls
{

std::string
VerilatorHarnessAxi::GetText(llvm::RvsdgModule & rm)
{
  std::ostringstream cpp;
  const auto & kernel = *get_hls_lambda(rm);
  const auto & function_name =
      dynamic_cast<llvm::LlvmLambdaOperation &>(kernel.GetOperation()).name();

  // The request and response parts of memory queues
  const auto mem_reqs = get_mem_reqs(kernel);
  const auto mem_resps = get_mem_resps(kernel);
  JLM_ASSERT(mem_reqs.size() == mem_resps.size());

  // All inputs that are not memory queues
  const auto reg_args = get_reg_args(kernel);

  // Extract info about the kernel's function signature in C
  const auto c_return_type = GetReturnTypeAsC(kernel);
  const auto [num_c_params, c_params, c_call_args] = GetParameterListAsC(kernel);

  cpp << R"(
#include "mmio.h"
#include "mm.h"
#include <memory>
#include <cassert>
#include <cmath>
#include <verilated.h>

#include "Vtop.h"

#if VM_TRACE_FST
#include <verilated_fst_c.h>
#elif VM_TRACE_VCD
#include <verilated_vcd_c.h>
#endif // VM_TRACE

#include <cstdint>
#include <cassert>
#include <iostream>
#include "verilator_sim.h"
#include "RhlsAxi.h"
std::unique_ptr<mm_magic_t> memories[)"
      << mem_reqs.size() << R"(];
// ======== Entry point for calling kernel from host device (C code) ========
extern "C" )"
      << c_return_type.value_or("void") << " " << function_name << "(" << c_params << ")" << R"(
{
    // initialize memories before verilator
    master = std::make_unique<mmio_t>(CTRL_BEAT_BYTES);)"
      << std::endl;
  size_t m = 0;
  for (size_t i = 0; i < kernel.GetOperation().type().Arguments().size(); ++i)
  {
    if (rvsdg::is<llvm::PointerType>(*kernel.GetOperation().type().Arguments()[i].get()))
    {
      const auto res_bundle = util::AssertedCast<const bundletype>(mem_resps[m]->Type().get());
      auto size = JlmSize(&*res_bundle->get_element_type("data")) / 8;
      cpp << "    memories[" << m << "] = std::make_unique<mm_magic_t>();" << std::endl;
      cpp << "    memories[" << m << "]->init((uint8_t *) a" << i << ", 1UL << 31, " << size
          << ", 64, MEMORY_LATENCY);" << std::endl;
      m++;
    }
  }
  // TODO: handle globals/ctxvars and ports without argument
  cpp << R"(
    verilator_init(0, nullptr);

    wait_cycles(10);

    // initialize struct with control addresses
    RHLSAXI_substruct_create
    // wait for accelerator to be ready
    while (!read(RHLSAXI_substruct->ready));

    // use relative addressing for now - i.e. skip pointer arguments
)";
  // TODO: handle 64 bit parameters & floats
  for (size_t i = 0; i < num_c_params; ++i)
  {
    auto type = kernel.GetOperation().type().Arguments()[i].get();
    if (!rvsdg::is<llvm::PointerType>(*type))
    {
      JLM_ASSERT(JlmSize(type) <= 32);
      cpp << "    write(RHLSAXI_substruct->i_data_" << i << ", a" << i << ");" << std::endl;
    }
  }
  // TODO: handle return value
  cpp << R"(

    // start the accelerator
    write(RHLSAXI_substruct->start, 1);
    size_t start_cycles = get_cycles();

    // wait for computation to be done
    while (!read(RHLSAXI_substruct->done));
    size_t end_cycles = get_cycles();
)";
  if (c_return_type.has_value())
    cpp << "auto result = read(RHLSAXI_substruct->o_data_0);" << std::endl;
  cpp << R"(
    std::cerr << "Accelerator took " << end_cycles - start_cycles << " cycles" << std::endl;
    write(RHLSAXI_substruct->finish, 1);
)";
  if (c_return_type.has_value())
    cpp << "return *(" << c_return_type.value() << "*)&result;" << std::endl;
  cpp << R"(
}

extern "C" void reference_load(void *addr, uint64_t width) {
}

extern "C" void reference_store(void *addr, uint64_t width) {
}

extern uint64_t main_time;
uint64_t clock_cycles = 0;
extern std::unique_ptr<mmio_t> master;

extern V_NAME *top;
#if VM_TRACE_FST
extern VerilatedFstC *tfp;
#elif VM_TRACE_VCD
extern VerilatedVcdC *tfp;
#endif // VM_TRACE

void tick() {
    mmio_t *m;
    assert(m = dynamic_cast<mmio_t *>(master.get()));

    // ASSUMPTION: All models have *no* combinational paths through I/O
    // Step 1: Clock lo -> propagate signals between DUT and software models
    AXI_LITE_SLAVE_CONNECT_VERILATOR_POS(top, io_s_ctrl, m, CTRL_BEAT_BYTES)

)";
  size_t m_read = 0, m_write = 0;
  for (size_t i = 0; i < mem_reqs.size(); i++)
  {
    const auto req_bundle = util::AssertedCast<const bundletype>(&mem_reqs[i]->type());
    const auto res_bundle = util::AssertedCast<const bundletype>(mem_resps[i]->Type().get());
    auto size = JlmSize(&*res_bundle->get_element_type("data")) / 8;
    const auto has_write = req_bundle->get_element_type("write") != nullptr;
    if (has_write)
    {
      cpp << "    AXI_FULL_MM_CONNECT_VERILATOR_POS(top, io_m_write_" << m_write++ << ", memories["
          << i << "], " << size << ")" << std::endl;
    }
    else
    {
      cpp << "    AXI_FULL_READ_ONLY_MM_CONNECT_VERILATOR_POS(top, io_m_read_" << m_read++
          << ", memories[" << i << "], " << size << ")" << std::endl;
    }
  }
  cpp << R"(
  top->eval();
#if VM_TRACE
  if (tfp)
    tfp->dump((double)main_time);
#ifdef TRACE_FLUSH
  tfp->flush();
#endif
#endif // VM_TRACE
  main_time += 5;

  top->clock = 0;
  top->eval(); // This shouldn't do much
#if VM_TRACE
  if (tfp)
    tfp->dump((double)main_time);
#ifdef TRACE_FLUSH
  tfp->flush();
#endif
#endif // VM_TRACE
  main_time += 5;

  // Step 2: Clock high, tick all software models and evaluate DUT with posedge
  AXI_LITE_SLAVE_CONNECT_VERILATOR_POS_EDGE(top, io_s_ctrl, m)

)";
  m_read = 0;
  m_write = 0;
  for (size_t i = 0; i < mem_reqs.size(); i++)
  {
    const auto bundle = util::AssertedCast<const bundletype>(&mem_reqs[i]->type());
    const auto has_write = bundle->get_element_type("write") != nullptr;
    if (has_write)
    {
      cpp << "    AXI_FULL_MM_CONNECT_VERILATOR_POS_EDGE(top, io_m_write_" << m_write++
          << ", memories[" << i << "])" << std::endl;
    }
    else
    {
      cpp << "    AXI_FULL_READ_ONLY_MM_CONNECT_VERILATOR_POS_EDGE(top, io_m_read_" << m_read++
          << ", memories[" << i << "])" << std::endl;
    }
  }
  cpp << R"(
  top->clock = 1;
  top->eval();
  clock_cycles++;
}
)";

  return cpp.str();
}

} // namespace jlm::hls
