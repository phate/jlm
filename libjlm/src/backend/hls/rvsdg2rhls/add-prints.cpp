/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/backend/hls/rvsdg2rhls/add-prints.hpp>
#include <jlm/ir/hls/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/traverser.hpp>

void
jlm::hls::add_prints(jive::region *region) {
	for (auto &node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node *>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++) {
				add_prints(structnode->subregion(n));
			}
		}
//		if (auto lo = dynamic_cast<const jlm::load_op *>(&(node->operation()))) {
//
//		} else if (auto so = dynamic_cast<const jlm::store_op *>(&(node->operation()))) {
//			auto po = hls::print_op::create(*node->input(1)->origin())[0];
//			node->input(1)->divert_to(po);
//		}
		if( dynamic_cast<jive::simple_node *>(node)
				&& node->noutputs()==1
				&& is<jive::bittype>(node->output(0)->type())
				&& !is<jlm::UndefValueOperation>(node)){
			auto out = node->output(0);
			std::vector<jive::input *> old_users(out->begin(), out->end());
			auto new_out = hls::print_op::create(*out)[0];
			for (auto user: old_users) {
				user->divert_to(new_out);
			}
		}
	}
}

void
jlm::hls::add_prints(jlm::RvsdgModule &rm) {
	auto &graph = rm.Rvsdg();
	auto root = graph.root();
	add_prints(root);
}

void
jlm::hls::convert_prints(jlm::RvsdgModule &rm) {
	auto &graph = rm.Rvsdg();
	auto root = graph.root();
	//TODO: make this less hacky by using the correct state types
	FunctionType fct({&jive::bit64, &jive::bit64}, {loopstatetype::create().get()});
	impport imp(fct, "printnode", linkage::external_linkage);
	auto printf = graph.add_import(imp);
	convert_prints(
    root,
    printf,
    fct);
}

jive::output *
jlm::hls::route_to_region(jive::output * output, jive::region * region)
{
	JLM_ASSERT(region != nullptr);

	if (region == output->region())
		return output;

	output = route_to_region(output, region->node()->region());

	if (auto gamma = dynamic_cast<jive::gamma_node*>(region->node())) {
		gamma->add_entryvar(output);
		output = region->argument(region->narguments()-1);
	}	else if (auto theta = dynamic_cast<jive::theta_node*>(region->node())) {
		output = theta->add_loopvar(output)->argument();
	} else if (auto lambda = dynamic_cast<jlm::lambda::node*>(region->node())) {
		output = lambda->add_ctxvar(output);
	} else {
		JLM_ASSERT(0);
	}

	return output;
}

void
jlm::hls::convert_prints(
  jive::region *region,
  jive::output * printf,
  const FunctionType & functionType)
{
	for (auto &node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node *>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++) {
				convert_prints(structnode->subregion(n), printf, functionType);
			}
		} else if (auto po = dynamic_cast<const jlm::hls::print_op *>(&(node->operation()))) {
			auto printf_local = route_to_region(printf, region); //TODO: prevent repetition?
			auto bc = jive::create_bitconstant(region, 64, po->id());
			jive::output * val = node->input(0)->origin();
			if(val->type()!=jive::bit64){
				auto bt = dynamic_cast<const jive::bittype*>(&val->type());
				JLM_ASSERT(bt);
				auto op = jlm::zext_op(bt->nbits(), 64);
				val = jive::simple_node::create_normalized(region, op, {val})[0];
			}
			jlm::CallNode::Create(printf_local, functionType, {bc, val});
			node->output(0)->divert_users(node->input(0)->origin());
			jive::remove(node);
		}
	}
}
