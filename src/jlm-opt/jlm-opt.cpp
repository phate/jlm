/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/view.h>
#include <jive/rvsdg/binary.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/statemux.h>
#include <jive/rvsdg/unary.h>

#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/opt/cne.hpp>
#include <jlm/opt/dne.hpp>
#include <jlm/opt/inlining.hpp>
#include <jlm/opt/invariance.hpp>
#include <jlm/opt/inversion.hpp>
#include <jlm/opt/pull.hpp>
#include <jlm/opt/push.hpp>
#include <jlm/opt/unroll.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <chrono>
#include <iostream>

enum class opt {cne, dne, iln, inv, psh, red, ivt, url, pll};

struct cmdflags {
	inline
	cmdflags()
	: xml(false)
	, llvm(false)
	{}

	bool xml;
	bool llvm;
	std::vector<opt> passes;
};

static void
print_usage(const std::string & app)
{
	std::cerr << "Usage: " << app << " [OPTIONS] FILE\n";
	std::cerr << "OPTIONS:\n";
	std::cerr << "--cne: Perform common node elimination.\n";
	std::cerr << "--dne: Perform dead node elimination.\n";
	std::cerr << "--iln: Perform function inlining.\n";
	std::cerr << "--inv: Perform invariant value redirection.\n";
	std::cerr << "--pll: Perform node pull in.\n";
	std::cerr << "--psh: Perform node push out.\n";
	std::cerr << "--red: Perform node reductions.\n";
	std::cerr << "--ivt: Perform theta-gamma inversion.\n";
	std::cerr << "--url: Perform loop unrolling.\n";
	std::cerr << "--llvm: Output LLVM IR.\n";
	std::cerr << "--xml: Output RVSDG as XML.\n";
}

static std::string
parse_cmdflags(int argc, char ** argv, cmdflags & flags)
{
	if (argc < 2) {
		std::cerr << "Expected LLVM IR file as input.\n";
		print_usage(argv[0]);
		exit(1);
	}

	static std::unordered_map<std::string, void(*)(cmdflags&)> map({
	  {"--cne", [](cmdflags & flags){ flags.passes.push_back(opt::cne); }}
	, {"--dne", [](cmdflags & flags){ flags.passes.push_back(opt::dne); }}
	, {"--iln", [](cmdflags & flags){ flags.passes.push_back(opt::iln); }}
	, {"--inv", [](cmdflags & flags){ flags.passes.push_back(opt::inv); }}
	, {"--pll", [](cmdflags & flags){ flags.passes.push_back(opt::pll); }}
	, {"--psh", [](cmdflags & flags){ flags.passes.push_back(opt::psh); }}
	, {"--red", [](cmdflags & flags){ flags.passes.push_back(opt::red); }}
	, {"--ivt", [](cmdflags & flags){ flags.passes.push_back(opt::ivt); }}
	, {"--url", [](cmdflags & flags){ flags.passes.push_back(opt::url); }}
	, {"--llvm", [](cmdflags & flags){ flags.llvm = true; }}
	, {"--xml", [](cmdflags & flags){ flags.xml = true; }}
	});

	for (int n = 1; n < argc-1; n++) {
		std::string flag(argv[n]);
		if (map.find(flag) != map.end()) {
			map[flag](flags);
			continue;
		}

		std::cerr << "Unknown command line flag: " << flag << "\n";
		print_usage(argv[0]);
		exit(1);
	}

	return std::string(argv[argc-1]);
}

static void
perform_reductions(jive::graph & graph)
{
	/* alloca operation */
	{
		auto nf = graph.node_normal_form(typeid(jlm::alloca_op));
		auto allocanf = static_cast<jlm::alloca_normal_form*>(nf);
		allocanf->set_mutable(true);
		allocanf->set_alloca_alloca_reducible(true);
		allocanf->set_alloca_mux_reducible(true);
	}

	/* mux operation */
	{
		auto nf = graph.node_normal_form(typeid(jive::mux_op));
		auto mnf = static_cast<jive::mux_normal_form*>(nf);
		mnf->set_mutable(true);
		mnf->set_mux_mux_reducible(true);
		mnf->set_multiple_origin_reducible(true);
	}

	/* store operation */
	{
		auto nf = jlm::store_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_store_mux_reducible(true);
		nf->set_store_store_reducible(true);
		nf->set_store_alloca_reducible(true);
		nf->set_multiple_origin_reducible(true);
	}

	/* load operation */
	{
		auto nf = jlm::load_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_load_mux_reducible(true);
		nf->set_load_store_reducible(true);
		nf->set_load_alloca_reducible(true);
		nf->set_multiple_origin_reducible(true);
		nf->set_load_store_state_reducible(true);
		nf->set_load_store_alloca_reducible(true);
	}

	/* gamma operation */
	{
		auto nf = jive::gamma_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_predicate_reduction(true);
		nf->set_control_constant_reduction(true);
	}

	/* unary operation */
	{
		auto nf = jive::unary_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_reducible(true);
	}

	/* binary operation */
	{
		auto nf = jive::binary_op::normal_form(&graph);
		nf->set_mutable(true);
		nf->set_reducible(true);
	}

	graph.normalize();
}

static void
perform_optimizations(jive::graph * graph, const std::vector<opt> & opts)
{
	static std::unordered_map<opt, void(*)(jive::graph&)> map({
	  {opt::cne, [](jive::graph & graph){ jlm::cne(graph); }}
	, {opt::dne, [](jive::graph & graph){ jlm::dne(graph); }}
	, {opt::iln, [](jive::graph & graph){ jlm::inlining(graph); }}
	, {opt::inv, [](jive::graph & graph){ jlm::invariance(graph); }}
	, {opt::pll, [](jive::graph & graph){ jlm::pull(graph); }}
	, {opt::psh, [](jive::graph & graph){ jlm::push(graph); }}
	, {opt::ivt, [](jive::graph & graph){ jlm::invert(graph); }}
	, {opt::url, [](jive::graph & graph){ jlm::unroll(graph, 4); }}
	, {opt::red, perform_reductions}
	});

	for (const auto & opt : opts) {
		if (map.find(opt) != map.end()) {
			map[opt](*graph);
			continue;
		}

		JLM_ASSERT(0);
	}
}

int
main(int argc, char ** argv)
{
	cmdflags flags;
	auto file = parse_cmdflags(argc, argv, flags);

	llvm::SMDiagnostic d;
	llvm::LLVMContext ctx;
	auto lm = llvm::parseIRFile(file, d, ctx);
	if (!lm) {
		d.print(argv[0], llvm::errs());
		exit(1);
	}

	auto jm = jlm::convert_module(*lm);

	#ifdef RVSDGTIME
		size_t ntacs = jlm::ntacs(*jm);
		auto start = std::chrono::high_resolution_clock::now();
	#endif

	auto rvsdg = jlm::construct_rvsdg(*jm);

	#ifdef RVSDGTIME
		size_t nnodes = jive::nnodes(rvsdg->graph()->root());
	#endif

	perform_optimizations(rvsdg->graph(), flags.passes);

	if (flags.llvm) {
		jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);

		#ifdef RVSDGTIME
			auto end = std::chrono::high_resolution_clock::now();
			std::cerr << "RVSDGTIME: "
			          << ntacs << " "
			          << nnodes << " "
			          << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()
			          << "\n";
		#endif

		lm = jlm::jlm2llvm::convert(*jm, ctx);
		llvm::raw_os_ostream os(std::cout);
		lm->print(os, nullptr);
	}

	if (flags.xml)
		jive::view_xml(rvsdg->graph()->root(), stdout);

	return 0;
}
