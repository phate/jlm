/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/Trace.hpp>

namespace jlm::llvm
{

class OutputTracer : public rvsdg::OutputTracer
{
public:
  OutputTracer(bool isDeep, bool isInterProcedural)
      : rvsdg::OutputTracer(isDeep, isInterProcedural)
  {}

protected:
  [[nodiscard]] rvsdg::Output &
  traceStep(rvsdg::Output & output, bool mayLeaveRegion) override
  {
    auto & trace1 = rvsdg::OutputTracer::traceStep(output, mayLeaveRegion);

    if (const auto [node, ioBarrierOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(trace1);
        node && ioBarrierOp)
    {
      return *node->input(0)->origin();
    }

    return trace1;
  }
};

rvsdg::Output &
traceOutput(rvsdg::Output & output)
{
  OutputTracer tracer(true, true);
  return tracer.trace(output);
}

std::optional<int64_t>
tryGetConstantSignedInteger(const rvsdg::Output & output)
{
  const auto & normalized = llvm::traceOutput(output);

  if (const auto [_, constant] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(normalized);
      constant)
  {
    const auto & rep = constant->Representation();
    if (rep.is_known() && rep.nbits() <= 64)
      return rep.to_int();
    return std::nullopt;
  }

  if (const auto [_, constant] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::BitConstantOperation>(normalized);
      constant)
  {
    const auto & rep = constant->value();
    if (rep.is_known() && rep.nbits() <= 64)
      return rep.to_int();
    return std::nullopt;
  }

  return std::nullopt;
}

}
