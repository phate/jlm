/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm-opt/cmdline.hpp>

#include <jlm/opt/dne.hpp>
#include <jlm/opt/cne.hpp>
#include <jlm/opt/inlining.hpp>
#include <jlm/opt/invariance.hpp>
#include <jlm/opt/pull.hpp>
#include <jlm/opt/push.hpp>
#include <jlm/opt/inversion.hpp>
#include <jlm/opt/unroll.hpp>
#include <jlm/opt/reduction.hpp>
#include <jlm/opt/optimization.hpp>

#include <llvm/Support/CommandLine.h>

namespace jlm {

enum class optimizationid {cne, dne, iln, inv, psh, red, ivt, url, pll};

static jlm::optimization *
mapoptid(enum optimizationid id)
{
	static jlm::cne cne;
	static jlm::dne dne;
	static jlm::fctinline fctinline;
	static jlm::ivr ivr;
	static jlm::pullin pullin;
	static jlm::pushout pushout;
	static jlm::tginversion tginversion;
	static jlm::loopunroll loopunroll(4);
	static jlm::nodereduction nodereduction;

	static std::unordered_map<optimizationid, jlm::optimization*>
	map({
	  {optimizationid::cne, &cne}
	, {optimizationid::dne, &dne}
	, {optimizationid::iln, &fctinline}
	, {optimizationid::inv, &ivr}
	, {optimizationid::pll, &pullin}
	, {optimizationid::psh, &pushout}
	, {optimizationid::ivt, &tginversion}
	, {optimizationid::url, &loopunroll}
	, {optimizationid::red, &nodereduction}
	});

	JLM_ASSERT(map.find(id) != map.end());
	return map[id];
}

void
parse_cmdline(int argc, char ** argv, jlm::cmdline_options & options)
{
	using namespace llvm;

	/*
		FIXME: The command line parser setup is currently redone
		for every invocation of parse_cmdline. We should be able
		to do it only once and then reset the parser on every
		invocation of parse_cmdline.
	*/

	cl::TopLevelSubCommand->reset();

	cl::opt<bool> show_help(
	  "help"
	, cl::ValueDisallowed
	, cl::desc("Display available options."));

	cl::opt<std::string> ifile(
	  cl::Positional
	, cl::desc("<input>"));

	cl::opt<std::string> ofile(
	  "o"
	, cl::desc("Write output to <file>")
	, cl::value_desc("file"));

	std::string desc("Write stats to <file>. Default is " + options.sd.filepath().to_str() + ".");
	cl::opt<std::string> sfile(
	  "s"
	, cl::desc(desc)
	, cl::value_desc("file"));

	cl::opt<bool> print_cfr_time(
	  "print-cfr-time"
	, cl::ValueDisallowed
	, cl::desc("Write CFR time to stats file."));

	cl::opt<bool> print_aggregation_time(
	  "print-aggregation-time"
	, cl::ValueDisallowed
	, cl::desc("Write aggregation time to stats file."));

	cl::opt<bool> print_annotation_time(
	  "print-annotation-time"
	, cl::ValueDisallowed
	, cl::desc("Write annotation time to stats file."));

	cl::opt<bool> print_rvsdg_construction(
	  "print-rvsdg-construction"
	, cl::ValueDisallowed
	, cl::desc("Write RVSDG construction stats to file."));

	cl::opt<bool> print_rvsdg_destruction(
	  "print-rvsdg-destruction"
	, cl::ValueDisallowed
	, cl::desc("Write RVSDG destruction stats to file."));

	cl::opt<bool> print_rvsdg_optimization(
	  "print-rvsdg-optimization"
	, cl::ValueDisallowed
	, cl::desc("Write RVSDG optimization stats to file."));

	cl::opt<bool> print_dne_stat(
	  "print-dne-stat"
	, cl::ValueDisallowed
	, cl::desc("Write dead node elimination statistics to file."));

	cl::opt<bool> print_cne_stat(
	  "print-cne-stat"
	, cl::ValueDisallowed
	, cl::desc("Write common node elimination statistics to file."));

	cl::opt<bool> print_iln_stat(
	  "print-iln-stat"
	, cl::ValueDisallowed
	, cl::desc("Write function inlining statistics to file."));

	cl::opt<bool> print_inv_stat(
	  "print-inv-stat"
	, cl::ValueDisallowed
	, cl::desc("Write invariant value reduction statistics to file."));

	cl::opt<bool> print_ivt_stat(
	  "print-ivt-stat"
	, cl::ValueDisallowed
	, cl::desc("Write theta-gamma inversion statistics to file."));

	cl::opt<bool> print_pull_stat(
	  "print-pull-stat"
	, cl::ValueDisallowed
	, cl::desc("Write pull statistics to file."));

	cl::opt<bool> print_push_stat(
	  "print-push-stat"
	, cl::ValueDisallowed
	, cl::desc("Write push statistics to file."));

	cl::opt<bool> print_reduction_stat(
	  "print-reduction-stat"
	, cl::ValueDisallowed
	, cl::desc("Write reduction statistics to file."));

	cl::opt<bool> print_unroll_stat(
	  "print-unroll-stat"
	, cl::ValueDisallowed
	, cl::desc("Write loop unrolling statistics to file."));

	cl::opt<outputformat> format(
	  cl::values(
		  clEnumValN(outputformat::llvm, "llvm", "Output LLVM IR [default]")
		, clEnumValN(outputformat::xml, "xml", "Output XML"))
	, cl::desc("Select output format"));

	cl::list<jlm::optimizationid> optids(
		cl::values(
		  clEnumValN(jlm::optimizationid::cne, "cne", "Common node elimination")
		, clEnumValN(jlm::optimizationid::dne, "dne", "Dead node elimination")
		, clEnumValN(jlm::optimizationid::iln, "iln", "Function inlining")
		, clEnumValN(jlm::optimizationid::inv, "inv", "Invariant value reduction")
		, clEnumValN(jlm::optimizationid::psh, "psh", "Node push out")
		, clEnumValN(jlm::optimizationid::pll, "pll", "Node pull in")
		, clEnumValN(jlm::optimizationid::red, "red", "Node reductions")
		, clEnumValN(jlm::optimizationid::ivt, "ivt", "Theta-gamma inversion")
		, clEnumValN(jlm::optimizationid::url, "url", "Loop unrolling"))
	, cl::desc("Perform optimization"));

	cl::ParseCommandLineOptions(argc, argv);

	if (show_help) {
		cl::PrintHelpMessage();
		exit(EXIT_SUCCESS);
	}

	if (!ofile.empty())
		options.ofile = ofile;

	if (!sfile.empty())
		options.sd.set_file(sfile);

	std::vector<jlm::optimization*> optimizations;
	for (auto & optid : optids)
		optimizations.push_back(mapoptid(optid));

	options.ifile = ifile;
	options.format = format;
	options.optimizations = optimizations;
	options.sd.print_cfr_time = print_cfr_time;
	options.sd.print_cne_stat = print_cne_stat;
	options.sd.print_dne_stat = print_dne_stat;
	options.sd.print_iln_stat = print_iln_stat;
	options.sd.print_inv_stat = print_inv_stat;
	options.sd.print_ivt_stat = print_ivt_stat;
	options.sd.print_pull_stat = print_pull_stat;
	options.sd.print_push_stat = print_push_stat;
	options.sd.print_reduction_stat = print_reduction_stat;
	options.sd.print_unroll_stat = print_unroll_stat;
	options.sd.print_annotation_time = print_annotation_time;
	options.sd.print_aggregation_time = print_aggregation_time;
	options.sd.print_rvsdg_construction = print_rvsdg_construction;
	options.sd.print_rvsdg_destruction = print_rvsdg_destruction;
	options.sd.print_rvsdg_optimization = print_rvsdg_optimization;
}

}
