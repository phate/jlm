/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-prints.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
add_prints(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        add_prints(structnode->subregion(n));
      }
    }
    //		if (auto lo = dynamic_cast<const jlm::load_op *>(&(node->operation()))) {
    //
    //		} else if (auto so = dynamic_cast<const jlm::store_op *>(&(node->operation()))) {
    //			auto po = hls::print_op::create(*node->input(1)->origin())[0];
    //			node->input(1)->divert_to(po);
    //		}
    if (dynamic_cast<jlm::rvsdg::simple_node *>(node) && node->noutputs() == 1
        && jlm::rvsdg::is<jlm::rvsdg::bittype>(node->output(0)->type())
        && !jlm::rvsdg::is<llvm::UndefValueOperation>(node))
    {
      auto out = node->output(0);
      std::vector<jlm::rvsdg::input *> old_users(out->begin(), out->end());
      auto new_out = hls::print_op::create(*out)[0];
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
  auto root = graph.root();
  add_prints(root);
}

void
convert_prints(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  // TODO: make this less hacky by using the correct state types
  auto fct =
      llvm::FunctionType::Create({ rvsdg::bittype::Create(64), rvsdg::bittype::Create(64) }, {});
  auto & printf =
      llvm::GraphImport::Create(graph, fct, "printnode", llvm::linkage::external_linkage);
  convert_prints(root, &printf, fct);
}

jlm::rvsdg::output *
route_to_region(jlm::rvsdg::output * output, jlm::rvsdg::region * region)
{
  JLM_ASSERT(region != nullptr);

  if (region == output->region())
    return output;

  output = route_to_region(output, region->node()->region());

  if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(region->node()))
  {
    gamma->add_entryvar(output);
    output = region->argument(region->narguments() - 1);
  }
  else if (auto theta = dynamic_cast<jlm::rvsdg::theta_node *>(region->node()))
  {
    output = theta->add_loopvar(output)->argument();
  }
  else if (auto lambda = dynamic_cast<llvm::lambda::node *>(region->node()))
  {
    output = lambda->add_ctxvar(output);
  }
  else
  {
    JLM_ASSERT(0);
  }

  return output;
}

void
convert_prints(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * printf,
    const std::shared_ptr<const llvm::FunctionType> & functionType)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        convert_prints(structnode->subregion(n), printf, functionType);
      }
    }
    else if (auto po = dynamic_cast<const print_op *>(&(node->operation())))
    {
      auto printf_local = route_to_region(printf, region); // TODO: prevent repetition?
      auto bc = jlm::rvsdg::create_bitconstant(region, 64, po->id());
      jlm::rvsdg::output * val = node->input(0)->origin();
      if (val->type() != *jlm::rvsdg::bittype::Create(64))
      {
        auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&val->type());
        JLM_ASSERT(bt);
        auto op = llvm::zext_op(bt->nbits(), 64);
        val = jlm::rvsdg::simple_node::create_normalized(region, op, { val })[0];
      }
      llvm::CallNode::Create(printf_local, functionType, { bc, val });
      node->output(0)->divert_users(node->input(0)->origin());
      jlm::rvsdg::remove(node);
    }
  }
}

}
