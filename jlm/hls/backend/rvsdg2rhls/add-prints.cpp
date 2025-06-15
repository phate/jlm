/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-prints.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
add_prints(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        add_prints(structnode->subregion(n));
      }
    }
    if (dynamic_cast<jlm::rvsdg::SimpleNode *>(node) && node->noutputs() == 1
        && jlm::rvsdg::is<rvsdg::bittype>(node->output(0)->Type())
        && !jlm::rvsdg::is<llvm::UndefValueOperation>(node))
    {
      auto out = node->output(0);
      std::vector<jlm::rvsdg::Input *> old_users(out->begin(), out->end());
      auto new_out = PrintOperation::create(*out)[0];
      for (auto user : old_users)
      {
        user->divert_to(new_out);
      }
    }
  }
}

void
add_prints(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  add_prints(root);
}

void
convert_prints(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  // TODO: make this less hacky by using the correct state types
  auto fct =
      rvsdg::FunctionType::Create({ rvsdg::bittype::Create(64), rvsdg::bittype::Create(64) }, {});
  auto & printf =
      llvm::GraphImport::Create(graph, fct, fct, "printnode", llvm::linkage::external_linkage);
  convert_prints(root, &printf, fct);
}

// TODO: get rid of this and use inlining version instead
jlm::rvsdg::Output *
route_to_region_rvsdg(jlm::rvsdg::Output * output, rvsdg::Region * region)
{
  JLM_ASSERT(region != nullptr);

  if (region == output->region())
    return output;

  output = route_to_region_rvsdg(output, region->node()->region());

  if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(region->node()))
  {
    gamma->AddEntryVar(output);
    output = region->argument(region->narguments() - 1);
  }
  else if (auto theta = dynamic_cast<rvsdg::ThetaNode *>(region->node()))
  {
    output = theta->AddLoopVar(output).pre;
  }
  else if (auto lambda = dynamic_cast<rvsdg::LambdaNode *>(region->node()))
  {
    output = lambda->AddContextVar(*output).inner;
  }
  else
  {
    JLM_ASSERT(0);
  }

  return output;
}

void
convert_prints(
    rvsdg::Region * region,
    jlm::rvsdg::Output * printf,
    const std::shared_ptr<const rvsdg::FunctionType> & functionType)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        convert_prints(structnode->subregion(n), printf, functionType);
      }
    }
    else if (auto po = dynamic_cast<const PrintOperation *>(&(node->GetOperation())))
    {
      auto printf_local = route_to_region_rvsdg(printf, region); // TODO: prevent repetition?
      auto & constantNode = llvm::IntegerConstantOperation::Create(*region, 64, po->id());
      jlm::rvsdg::Output * val = node->input(0)->origin();
      if (*val->Type() != *jlm::rvsdg::bittype::Create(64))
      {
        auto bt = std::dynamic_pointer_cast<const rvsdg::bittype>(val->Type());
        JLM_ASSERT(bt);
        val = &llvm::ZExtOperation::Create(*val, rvsdg::bittype::Create(64));
      }
      llvm::CallOperation::Create(printf_local, functionType, { constantNode.output(0), val });
      node->output(0)->divert_users(node->input(0)->origin());
      jlm::rvsdg::remove(node);
    }
  }
}

}
