/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm-opt/cmdline.hpp>

#include <jlm/opt/alias-analyses/Optimization.hpp>
#include <jlm/opt/cne.hpp>
#include <jlm/opt/dne.hpp>
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

enum class optimizationid {aasteensgaard, cne, dne, iln, inv, psh, red, ivt, url, pll};

static jlm::optimization *
mapoptid(enum optimizationid id)
{
	static jlm::aa::SteensgaardBasic aasteensgaard;
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
	  {optimizationid::aasteensgaard, &aasteensgaard}
	, {optimizationid::cne, &cne}
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

  cl::list<StatisticsDescriptor::StatisticsId> printStatistics(
    cl::values(
      clEnumValN(StatisticsDescriptor::StatisticsId::Aggregation,
                 "print-aggregation-time",
                 "Write aggregation statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::Annotation,
                 "print-annotation-time",
                 "Write annotation statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::CommonNodeElimination,
                 "print-cne-stat",
                 "Write common node elimination statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::ControlFlowRecovery,
                 "print-cfr-time",
                 "Write control flow recovery statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::DeadNodeElimination,
                 "print-dne-stat",
                 "Write dead node elimination statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::FunctionInlining,
                 "print-iln-stat",
                 "Write function inlining statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::InvariantValueReduction,
                 "print-inv-stat",
                 "Write invariant value reduction statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::JlmToRvsdgConversion,
                 "print-jlm-rvsdg-conversion",
                 "Write Jlm to RVSDG conversion statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::LoopUnrolling,
                 "print-unroll-stat",
                 "Write loop unrolling statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::PullNodes,
                 "print-pull-stat",
                 "Write node pull statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::PushNodes,
                 "print-push-stat",
                 "Write node push statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::ReduceNodes,
                 "print-reduction-stat",
                 "Write node reduction statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::RvsdgConstruction,
                 "print-rvsdg-construction",
                 "Write RVSDG construction statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::RvsdgDestruction,
                 "print-rvsdg-destruction",
                 "Write RVSDG destruction statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::RvsdgOptimization,
                 "print-rvsdg-optimization",
                 "Write RVSDG optimization statistics to file."),
      clEnumValN(StatisticsDescriptor::StatisticsId::ThetaGammaInversion,
                 "print-ivt-stat",
                 "Write theta-gamma inversion statistics to file.")),
    cl::desc("Write statistics"));

	cl::opt<outputformat> format(
	  cl::values(
		  clEnumValN(outputformat::llvm, "llvm", "Output LLVM IR [default]")
		, clEnumValN(outputformat::xml, "xml", "Output XML"))
	, cl::desc("Select output format"));

	cl::list<jlm::optimizationid> optids(
		cl::values(
		  clEnumValN(jlm::optimizationid::aasteensgaard,
		    "aa-steensgaard", "Steensgaard alias analysis")
		, clEnumValN(jlm::optimizationid::cne, "cne", "Common node elimination")
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

  std::unordered_set<StatisticsDescriptor::StatisticsId> printStatisticsIds(
    printStatistics.begin(), printStatistics.end());

	options.ifile = ifile;
	options.format = format;
	options.optimizations = optimizations;
  options.sd.SetPrintStatisticsIds(printStatisticsIds);
}

}
