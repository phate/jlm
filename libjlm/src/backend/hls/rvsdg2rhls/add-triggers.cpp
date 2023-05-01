/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/backend/hls/rvsdg2rhls/add-triggers.hpp>
#include <jlm/backend/hls/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/ir/hls/hls.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

jive::output *
jlm::hls::get_trigger(jive::region *region) {
	for (size_t i = 0; i < region->narguments(); ++i) {
		if (region->argument(i)->type() == hls::trigger) {
			return region->argument(i);
		}
	}
	return nullptr;
}

jlm::lambda::node *
jlm::hls::add_lambda_argument(jlm::lambda::node *ln, const jive::type *type) {
	auto old_fcttype = ln->type();
	std::vector<const jive::type *> new_argument_types;
	for (size_t i = 0; i < old_fcttype.NumArguments(); ++i) {
		new_argument_types.push_back(&old_fcttype.ArgumentType(i));
	}
	new_argument_types.push_back(type);
	std::vector<const jive::type *> new_result_types;
	for (size_t i = 0; i < old_fcttype.NumResults(); ++i) {
		new_result_types.push_back(&old_fcttype.ResultType(i));
	}
	FunctionType new_fcttype(new_argument_types, new_result_types);
	auto new_lambda = jlm::lambda::node::create(ln->region(), new_fcttype, ln->name(), ln->linkage(),
												ln->attributes());

	jive::substitution_map smap;
	for (size_t i = 0; i < ln->ncvarguments(); ++i) {
		// copy over cvarguments
		smap.insert(ln->cvargument(i), new_lambda->add_ctxvar(ln->cvargument(i)->input()->origin()));
	}
	for (size_t i = 0; i < ln->nfctarguments(); ++i) {
		smap.insert(ln->fctargument(i), new_lambda->fctargument(i));
	}
//	jive::view(ln->subregion(), stdout);
//	jive::view(new_lambda->subregion(), stdout);
	ln->subregion()->copy(new_lambda->subregion(), smap, false, false);

	std::vector<jive::output *> new_results;
	for (size_t i = 0; i < ln->nfctresults(); ++i) {
		new_results.push_back(smap.lookup(ln->fctresult(i)->origin()));
	}
	auto new_out = new_lambda->finalize(new_results);

	// TODO handle functions at other levels?
	assert(ln->region() == ln->region()->graph()->root());
	assert((*ln->output()->begin())->region() == ln->region()->graph()->root());

//            ln->output()->divert_users(new_out);
	ln->region()->remove_result((*ln->output()->begin())->index());
	remove(ln);
	jive::result::create(new_lambda->region(), new_out, nullptr, new_out->type());
	return new_lambda;
}

void
jlm::hls::add_triggers(jive::region *region) {
	auto trigger = get_trigger(region);
	for (auto &node : jive::topdown_traverser(region)) {
		if (dynamic_cast<jive::structural_node *>(node)) {
			if (auto ln = dynamic_cast<jlm::lambda::node *>(node)) {
				// check here in order not to process removed and re-added node twice
				if (!get_trigger(ln->subregion())) {
					auto new_lambda = add_lambda_argument(ln, &(hls::trigger));
					add_triggers(new_lambda->subregion());
				}
			} else if (auto t = dynamic_cast<jive::theta_node *>(node)) {
				assert(trigger != nullptr);
				assert(get_trigger(t->subregion()) == nullptr);
				t->add_loopvar(trigger);
				add_triggers(t->subregion());
			} else if (auto gn = dynamic_cast<jive::gamma_node *>(node)) {
				assert(trigger != nullptr);
				assert(get_trigger(gn->subregion(0)) == nullptr);
				gn->add_entryvar(trigger);
				for (size_t i = 0; i < gn->nsubregions(); ++i) {
					add_triggers(gn->subregion(i));
				}
			} else {
				throw jlm::error("Unexpected node type: " + node->operation().debug_string());
			}
		} else if (auto sn = dynamic_cast<jive::simple_node *>(node)) {
			assert(trigger != nullptr);
			if (is_constant(node)) {
				auto orig_out = sn->output(0);
				std::vector<jive::input *> previous_users(orig_out->begin(), orig_out->end());
				auto gated = hls::trigger_op::create(*trigger, *orig_out)[0];
				for (auto user: previous_users) {
					user->divert_to(gated);
				}
			}
		} else {
			throw jlm::error("Unexpected node type: " + node->operation().debug_string());
		}
	}
}

void
jlm::hls::add_triggers(jlm::RvsdgModule &rm) {
	auto &graph = rm.Rvsdg();
	auto root = graph.root();
	add_triggers(root);
}
