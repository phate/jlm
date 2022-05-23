/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/command.hpp>
#include <jlc/llvmpaths.hpp>
#include <jlm/util/strfmt.hpp>

#include <deque>
#include <functional>
#include <iostream>
#include <memory>

namespace jlm {

/* command generation */

static LlcCommand::OptimizationLevel
ToLlcCommandOptimizationLevel(optlvl optimizationLevel)
{
  static std::unordered_map<optlvl, LlcCommand::OptimizationLevel>
    map({
          {optlvl::O0, LlcCommand::OptimizationLevel::O0},
          {optlvl::O1, LlcCommand::OptimizationLevel::O1},
          {optlvl::O2, LlcCommand::OptimizationLevel::O2},
          {optlvl::O3, LlcCommand::OptimizationLevel::O3}
        });

  JLM_ASSERT(map.find(optimizationLevel) != map.end());
  return map[optimizationLevel];
}

static std::string
create_optcmd_ofile(const std::string & ifile)
{
  return strfmt("tmp-", ifile, "-jlm-opt-out.ll");
}

static std::string
create_prscmd_ofile(const std::string & ifile)
{
  return strfmt("tmp-", ifile, "-clang-out.ll");
}

std::unique_ptr<CommandGraph>
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<CommandGraph> pgraph(new CommandGraph());

	std::vector<CommandGraph::Node*> leaves;
	for (const auto & c : opts.compilations) {
		auto last = &pgraph->GetEntryNode();

		if (c.parse()) {
			auto prsnode = prscmd::create(
				pgraph.get(),
				c.ifile(),
				c.DependencyFile(),
				opts.includepaths,
				opts.macros,
				opts.warnings,
				opts.flags,
				opts.verbose,
				opts.rdynamic,
				opts.suppress,
				opts.pthread,
				opts.MD,
				c.Mt(),
				opts.std);

      last->AddEdge(*prsnode);
			last = prsnode;
		}

    if (c.optimize()) {
      /*
       * If a default optimization level has been specified (-O) and no specific jlm options
       * have been specified (-J) then use a default set of optimizations.
       */
      std::vector<optcmd::Optimization> optimizations;
      if (opts.jlmopts.empty() && opts.Olvl == optlvl::O3) {
        /*
         * Only -O3 sets default optimizations
         */
        optimizations = {
          optcmd::Optimization::FunctionInlining,
          optcmd::Optimization::InvariantValueRedirection,
          optcmd::Optimization::NodeReduction,
          optcmd::Optimization::DeadNodeElimination,
          optcmd::Optimization::ThetaGammaInversion,
          optcmd::Optimization::InvariantValueRedirection,
          optcmd::Optimization::DeadNodeElimination,
          optcmd::Optimization::NodePushOut,
          optcmd::Optimization::InvariantValueRedirection,
          optcmd::Optimization::DeadNodeElimination,
          optcmd::Optimization::NodeReduction,
          optcmd::Optimization::CommonNodeElimination,
          optcmd::Optimization::DeadNodeElimination,
          optcmd::Optimization::NodePullIn,
          optcmd::Optimization::InvariantValueRedirection,
          optcmd::Optimization::DeadNodeElimination,
          optcmd::Optimization::LoopUnrolling,
          optcmd::Optimization::InvariantValueRedirection
        };
      }

      auto optnode = optcmd::create(
        pgraph.get(),
        "/tmp/" + create_prscmd_ofile(c.ifile().base()),
        "/tmp/" + create_optcmd_ofile(c.ifile().base()),
        optimizations);
      last->AddEdge(*optnode);
      last = optnode;
    }

		if (c.assemble()) {
			auto & asmnode = LlcCommand::Create(
        *pgraph,
        "/tmp/" + create_optcmd_ofile(c.ifile().base()),
        c.ofile(),
        ToLlcCommandOptimizationLevel(opts.Olvl),
        LlcCommand::RelocationModel::Static);
      last->AddEdge(asmnode);
			last = &asmnode;
		}

		leaves.push_back(last);
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

  if (!lnkifiles.empty()) {
    auto & linkerCommandNode = ClangCommand::Create(
      *pgraph,
      lnkifiles,
      opts.lnkofile,
      opts.libpaths,
      opts.libs,
      opts.pthread);

    for (const auto & leave : leaves)
      leave->AddEdge(linkerCommandNode);

    leaves.clear();
    leaves.push_back(&linkerCommandNode);
  }

	for (const auto & leave : leaves)
    leave->AddEdge(pgraph->GetExitNode());

  if (opts.only_print_commands)
    pgraph = PrintCommandsCommand::Create(std::move(pgraph));

	return pgraph;
}

/* parser command */

prscmd::~prscmd()
{}

std::string
prscmd::replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

std::string
prscmd::ToString() const
{
	auto f = ifile_.base();

	std::string Ipaths;
	for (const auto & Ipath : Ipaths_)
		Ipaths += "-I" + Ipath + " ";

	std::string Dmacros;
	for (const auto & Dmacro : Dmacros_)
		Dmacros += "-D" + Dmacro + " ";

	std::string Wwarnings;
	for (const auto & Wwarning : Wwarnings_)
		Wwarnings += "-W" + Wwarning + " ";

	std::string flags;
	for (const auto & flag : flags_)
		flags += "-f" + flag + " ";

	std::string arguments;
	if (verbose_)
	  arguments += "-v ";

	if (rdynamic_)
	  arguments += "-rdynamic ";

	if (suppress_)
	  arguments += "-w ";

	if (pthread_)
	  arguments += "-pthread ";

	if (MD_) {
		arguments += "-MD ";
		arguments += "-MF " + dependencyFile_.to_str() + " ";
		arguments += "-MT " + mT_ + " ";
	}

	return strfmt(
	  clangpath.to_str() + " "
	, arguments, " "
	, Wwarnings, " "
	, flags, " "
	, std_ != standard::none ? "-std="+jlm::to_str(std_)+" " : ""
	, replace_all(Dmacros, std::string("\""), std::string("\\\"")), " "
	, Ipaths, " "
	, "-S -emit-llvm "
	, "-o /tmp/", create_prscmd_ofile(f), " "
	, ifile_.to_str()
	);
}

void
prscmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

}
