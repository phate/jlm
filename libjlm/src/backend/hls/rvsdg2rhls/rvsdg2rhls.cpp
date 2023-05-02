/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/backend/hls/rvsdg2rhls/add-forks.hpp>
#include <jlm/backend/hls/rvsdg2rhls/add-prints.hpp>
#include <jlm/backend/hls/rvsdg2rhls/add-sinks.hpp>
#include <jlm/backend/hls/rvsdg2rhls/add-triggers.hpp>
#include <jlm/backend/hls/rvsdg2rhls/check-rhls.hpp>
#include <jlm/backend/hls/rvsdg2rhls/gamma-conv.hpp>
#include <jlm/backend/hls/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/backend/hls/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/backend/hls/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/backend/hls/rvsdg2rhls/theta-conv.hpp>
#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <regex>

void
pre_opt(jlm::RvsdgModule &rm) {
	// TODO: figure out which optimizations to use here
	jlm::DeadNodeElimination dne;
	jlm::cne cne;
	jlm::InvariantValueRedirection ivr;
	jlm::tginversion tgi;
	jlm::StatisticsCollector statisticsCollector;
	tgi.run(rm, statisticsCollector);
	dne.run(rm, statisticsCollector);
	cne.run(rm, statisticsCollector);
	ivr.run(rm, statisticsCollector);
}

namespace jlm {

	bool
	function_match(lambda::node *ln, const std::string &function_name) {
		const std::regex fn_regex(function_name);
		if (std::regex_match(ln->name(), fn_regex)) {// TODO: handle C++ name mangling
			return true;
		}
		return false;
	}

	const jive::output *
	trace_call(jive::input *input) {
		auto graph = input->region()->graph();

		auto argument = dynamic_cast<const jive::argument *>(input->origin());
		const jive::output * result;
		if (auto to = dynamic_cast<const jive::theta_output*>(input->origin())){
			result = trace_call(to->input());
		} else if (argument == nullptr) {
			result = input->origin();
		} else if (argument->region() == graph->root()){
			result = argument;
		} else{
			JLM_ASSERT(argument->input() != nullptr);
			result = trace_call(argument->input());
		}
		auto so = dynamic_cast<const jive::structural_output *>(result);
		if(!so){
			auto arg = dynamic_cast<const jive::argument *>(result);
			auto ip = dynamic_cast<const impport *>(&arg->port());
			if(ip){
				throw jive::compiler_error("can not inline external function "+ip->name());
			}
		}
		JLM_ASSERT(so);
		JLM_ASSERT(is<lambda::operation>(so->node()));
		return result;
	}

	void
	inline_calls(jive::region *region) {
		for (auto &node : jive::topdown_traverser(region)) {
			if (auto structnode = dynamic_cast<jive::structural_node *>(node)) {
				for (size_t n = 0; n < structnode->nsubregions(); n++) {
					inline_calls(structnode->subregion(n));
				}
			} else if (dynamic_cast<const jlm::CallOperation *>(&(node->operation()))) {
				auto func = trace_call(node->input(0));
				auto ln = dynamic_cast<const jive::structural_output *>(func)->node();
				jlm::inlineCall(dynamic_cast<jive::simple_node *>(node), dynamic_cast<const lambda::node *>(ln));
				// restart for this region
				inline_calls(region);
				return;
			}
		}
	}

	size_t alloca_cnt = 0;

	void
	convert_alloca(jive::region *region) {
		for (auto &node : jive::topdown_traverser(region)) {
			if (auto structnode = dynamic_cast<jive::structural_node *>(node)) {
				for (size_t n = 0; n < structnode->nsubregions(); n++) {
					convert_alloca(structnode->subregion(n));
				}
			} else if (auto po = dynamic_cast<const jlm::alloca_op *>(&(node->operation()))) {
				auto rr = region->graph()->root();
				auto delta_name = jive::detail::strfmt("hls_alloca_", alloca_cnt++);
        PointerType delta_type;
        std::cout << "alloca " << delta_name << ": " << po->value_type().debug_string() << "\n";
				auto db = delta::node::Create(rr, po->value_type(), delta_name, linkage::external_linkage, "", false);
				// create zero constant of allocated type
				jive::output *cout;
				if (auto bt = dynamic_cast<const jive::bittype *>(&po->value_type())) {
					cout = jive::create_bitconstant(db->subregion(), bt->nbits(), 0);
				} else {
					ConstantAggregateZero cop(po->value_type());
					cout = jive::simple_node::create_normalized(db->subregion(), cop, {})[0];
				}
				auto delta = db->finalize(cout);
				region->graph()->add_export(delta, {delta_type, delta_name});
				auto delta_local = jlm::hls::route_to_region(delta, region);
				node->output(0)->divert_users(delta_local);
				// TODO: check that the input to alloca is a bitconst 1
				JLM_ASSERT(node->output(1)->nusers() == 1);
				auto mux_in = *node->output(1)->begin();
				auto mux_node = input_node(mux_in);
				JLM_ASSERT(dynamic_cast<const jlm::MemStateMergeOperator *>(&mux_node->operation()));
				JLM_ASSERT(mux_node->ninputs() == 2);
				auto other_index = mux_in->index() ? 0 : 1;
				mux_node->output(0)->divert_users(mux_node->input(other_index)->origin());
				jive::remove(mux_node);
				jive::remove(node);
			}
		}
	}

	delta::node *
	rename_delta(delta::node *odn) {
		auto name = odn->name();
		std::replace_if(name.begin(), name.end(), [](char c) { return c == '.'; }, '_');
		std::cout << "renaming delta node " << odn->name() << " to " << name << "\n";
		auto db = delta::node::Create(odn->region(), odn->type(), name, linkage::external_linkage, "", odn->constant());
		/* add dependencies */
		jive::substitution_map rmap;
		for (size_t i=0; i<odn->ncvarguments(); i++) {
			auto input = odn->input(i);
			auto nd = db->add_ctxvar(input->origin());
			rmap.insert(input->argument(), nd);
		}

		/* copy subregion */
		odn->subregion()->copy(db->subregion(), rmap, false, false);

		auto result = rmap.lookup(odn->subregion()->result(0)->origin());
		auto data = db->finalize(result);

		odn->output()->divert_users(data);
		jive::remove(odn);
		return static_cast<delta::node *>(jive::node_output::node(data));
	}

    lambda::node * change_linkage(lambda::node * ln, linkage link){
        auto lambda = lambda::node::create(ln->region(), ln->type(), ln->name(), link, ln->attributes());

        /* add context variables */
        jive::substitution_map subregionmap;
        for (auto & cv : ln->ctxvars()) {
            auto origin = cv.origin();
            auto newcv = lambda->add_ctxvar(origin);
            subregionmap.insert(cv.argument(), newcv);
        }

        /* collect function arguments */
        for (size_t n = 0; n < ln->nfctarguments(); n++)
            subregionmap.insert(ln->fctargument(n), lambda->fctargument(n));

        /* copy subregion */
        ln->subregion()->copy(lambda->subregion(), subregionmap, false, false);

        /* collect function results */
        std::vector<jive::output*> results;
        for (auto & result : ln->fctresults())
            results.push_back(subregionmap.lookup(result.origin()));

        /* finalize lambda */
        lambda->finalize(results);

        divert_users(ln,outputs(lambda));
        jive::remove(ln);

        return lambda;
    }
}

std::unique_ptr<jlm::RvsdgModule>
jlm::hls::split_hls_function(jlm::RvsdgModule &rm, const std::string &function_name) {
    // TODO: use a different datastructure for rhls?
    // create a copy of rm
    auto rhls = jlm::RvsdgModule::Create(rm.SourceFileName(), rm.TargetTriple(), rm.DataLayout());
    std::cout << "processing " << rm.SourceFileName().name() << "\n";
    auto root = rm.Rvsdg().root();
    for (auto node : jive::topdown_traverser(root)) {
        if (auto ln = dynamic_cast<lambda::node *>(node)) {
            if (!function_match(ln, function_name)) {
                continue;
            }
            inline_calls(ln->subregion());
            // TODO: have a seperate set of optimizations here
            pre_opt(rm);
            convert_alloca(ln->subregion());
            jive::substitution_map smap;
            for (size_t i = 0; i < ln->ninputs(); ++i) {
                auto orig_node = dynamic_cast<jive::node_output *>(ln->input(i)->origin())->node();
                if (auto oln = dynamic_cast<lambda::node *>(orig_node)) {
                    throw jlm::error("Inlining of function " + oln->name() + " not supported");
                } else if (auto odn = dynamic_cast<delta::node *>(orig_node)) {
                    // modify name to not contain .
                    if (odn->name().find('.') != std::string::npos) {
                        odn = rename_delta(odn);
                    }
                    std::cout << "delta node " << odn->name() << ": " << odn->type().debug_string() << "\n";
                    // add import for delta to rhls
                    impport im(odn->type(), odn->name(), linkage::external_linkage);
//						JLM_ASSERT(im.name()==odn->name());
                    auto arg = rhls->Rvsdg().add_import(im);
                    auto tmp = dynamic_cast<const impport *>(&arg->port());
                    assert(tmp && tmp->name() == odn->name());
                    smap.insert(ln->input(i)->origin(), arg);
                    // add export for delta to rm
                    // TODO: check if not already exported and maybe adjust linkage?
                    rm.Rvsdg().add_export(odn->output(), {odn->output()->type(), odn->name()});
                } else {
                    throw jlm::error("Unsupported node type: " + orig_node->operation().debug_string());
                }
            }
            // copy function into rhls
            auto new_ln = ln->copy(rhls->Rvsdg().root(), smap);
            new_ln = change_linkage(new_ln, linkage::external_linkage);
            jive::result::create(rhls->Rvsdg().root(), new_ln->output(), nullptr, new_ln->output()->type());
            // add function as input to rm and remove it
            impport im(ln->type(), ln->name(), linkage::external_linkage); //TODO: change linkage?
            auto arg = rm.Rvsdg().add_import(im);
            ln->output()->divert_users(arg);
            remove(ln);
            std::cout << "function " << new_ln->name() << " extracted for HLS\n";
            return rhls;
        }
    }
    throw jlm::error("HLS function " + function_name + " not found");
}

void
jlm::hls::rvsdg2rhls(jlm::RvsdgModule &rhls) {
	pre_opt(rhls);

//	jlm::hls::add_prints(rhls);
//	dump_ref(rhls);

	// run conversion on copy
	jlm::hls::remove_unused_state(rhls);
	// main conversion steps
	jlm::hls::add_triggers(rhls); // TODO: make compatible with loop nodes?
	jlm::hls::gamma_conv(rhls, true);
	jlm::hls::theta_conv(rhls);
	// rhls optimization
	jlm::hls::dne(rhls);
	// enforce 1:1 input output relationship
	jlm::hls::add_sinks(rhls);
	jlm::hls::add_forks(rhls);
//	jlm::hls::add_buffers(*rhls, true);
	// ensure that all rhls rules are met
	jlm::hls::check_rhls(rhls);
}

void
jlm::hls::dump_ref(jlm::RvsdgModule &rhls) {
	auto reference = RvsdgModule::Create(rhls.SourceFileName(), rhls.TargetTriple(), rhls.DataLayout());
	jive::substitution_map smap;
	rhls.Rvsdg().root()->copy(reference->Rvsdg().root(), smap, true, true);
	convert_prints(*reference);
	for (size_t i = 0; i < reference->Rvsdg().root()->narguments(); ++i) {
		auto arg = reference->Rvsdg().root()->argument(i);
		auto imp = dynamic_cast<const impport *>(&arg->port());
		std::cout << "impport " << imp->name() << ": " << imp->type().debug_string() << "\n";
	}
	llvm::LLVMContext ctx;
	StatisticsCollector statisticsCollector;
	auto jm2 = rvsdg2jlm::rvsdg2jlm(*reference, statisticsCollector);
	auto lm2 = jlm2llvm::convert(*jm2, ctx);
	std::error_code EC;
	llvm::raw_fd_ostream os(reference->SourceFileName().base() + ".ref.ll", EC);
	lm2->print(os, nullptr);
}
