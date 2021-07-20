/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/command.hpp>
#include <jlc/llvmpaths.hpp>
#include <jlc/toolpaths.hpp>
#include <jlm/util/strfmt.hpp>

#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <algorithm>
#include <sys/stat.h>

namespace jlm {

/* command generation */

std::unique_ptr<passgraph>
generate_commands(const jlm::cmdline_options & opts)
{
	std::unique_ptr<passgraph> pgraph(new passgraph());

	std::vector<passgraph_node*> leaves;
	std::vector<passgraph_node*> llir;
	std::vector<jlm::filepath> llir_files;

	// Create directory in /tmp for storing temporary files
	std::string tmp_identifier;
	for (const auto & c : opts.compilations) {
		tmp_identifier += c.ifile().name() + "_";
		if (tmp_identifier.length() > 30)
			break;
	}
	srandom(time(nullptr));
	tmp_identifier += std::to_string(random());
	jlm::filepath tmp_folder("/tmp/" + tmp_identifier + "/");
	auto mkdir = mkdircmd::create(pgraph.get(), tmp_folder);
	pgraph->entry()->add_edge(mkdir);

	for (const auto & c : opts.compilations) {
		passgraph_node* last = mkdir;

		if (c.parse()) {
			auto prsnode = prscmd::create(
				pgraph.get(),
				c.ifile(),
				c.DependencyFile(),
				tmp_folder,
				opts.includepaths,
				opts.macros,
				opts.warnings,
				opts.flags,
				opts.verbose,
				opts.rdynamic,
				opts.suppress,
				opts.pthread,
				opts.MD,
				opts.hls,
				c.Mt(),
				opts.std);

			last->add_edge(prsnode);
			last = prsnode;

			// HLS links all files to a single IR
			// Need to know when the IR has been generated for all input files
			llir.push_back(prsnode);
			llir_files.push_back(
				dynamic_cast<prscmd*>(&prsnode->cmd())->ofile());
		}

		if (c.optimize() && !opts.hls) {
			auto optnode = optcmd::create(
				pgraph.get(),
				c.ifile(),
				tmp_folder,
				opts.jlmopts,
				opts.Olvl);
			last->add_edge(optnode);
			last = optnode;
		}

		if (c.assemble() && !opts.hls) {
			auto asmnode = cgencmd::create(
				pgraph.get(),
				c.ifile(),
				c.ofile(),
				tmp_folder,
				opts.hls,
				opts.Olvl);
			last->add_edge(asmnode);
			last = asmnode;
		}

		leaves.push_back(last);
	}

	if (opts.hls) {
		// link all llir into one so inlining can be done across files for HLS
		jlm::filepath ll_merged(tmp_folder.to_str()+"merged.ll");
		auto ll_link = lllnkcmd::create(pgraph.get(), llir_files, ll_merged);
		// Add edges between each c.parse and the ll_link
		for (const auto & ll : llir) {
			ll->add_edge(ll_link);
		}

		// need to already run m2r here
		jlm::filepath  ll_m2r1(tmp_folder.to_str()+"merged.m2r.ll");
		auto m2r1 = m2rcmd::create(pgraph.get(), ll_merged, ll_m2r1);
		ll_link->add_edge(m2r1);
		auto extract = extractcmd::create(
				pgraph.get(),
				dynamic_cast<m2rcmd*>(&m2r1->cmd())->ofile(),
				opts.hls_function_regex,
				tmp_folder.to_str());
		m2r1->add_edge(extract);
		jlm::filepath  ll_m2r2(tmp_folder.to_str()+"function.m2r.ll");
		auto m2r2 = m2rcmd::create(
				pgraph.get(),
				dynamic_cast<extractcmd*>(&extract->cmd())->functionfile(),
				ll_m2r2);
		extract->add_edge(m2r2);
		// hls
		auto hls = hlscmd::create(
				pgraph.get(),
				dynamic_cast<m2rcmd*>(&m2r2->cmd())->ofile(),
				tmp_folder.to_str(),
				opts.circt);
		m2r2->add_edge(hls);
		//	}

		//	if (!opts.generate_firrtl) {
		jlm::filepath verilogfile(tmp_folder.to_str()+"jlm_hls.v");
		auto firrtl = firrtlcmd::create(
				pgraph.get(),
				dynamic_cast<hlscmd*>(&hls->cmd())->firfile(),
				verilogfile);
		hls->add_edge(firrtl);
		jlm::filepath asmofile(tmp_folder.to_str()+"hls.o");
		auto asmnode = cgencmd::create(
				pgraph.get(),
				dynamic_cast<hlscmd*>(&hls->cmd())->llfile(),
				asmofile,
				tmp_folder,
				opts.hls,
				opts.Olvl);
		hls->add_edge(asmnode);

		std::vector<jlm::filepath> lnkifiles;
		for (const auto & c : opts.compilations) {
			if (c.link() && !c.parse())
				lnkifiles.push_back(c.ofile());
		}
		lnkifiles.push_back(asmofile);
		// TODO: remove old verilator folder first
		auto verinode = verilatorcmd::create(
				pgraph.get(),
				verilogfile,
				lnkifiles,
				dynamic_cast<hlscmd*>(&hls->cmd())->harnessfile(),
				opts.lnkofile,
				tmp_folder,
				opts.libpaths,
				opts.libs);
		firrtl->add_edge(verinode);
		verinode->add_edge(pgraph->exit());
	}

	std::vector<jlm::filepath> lnkifiles;
	for (const auto & c : opts.compilations) {
		if (c.link())
			lnkifiles.push_back(c.ofile());
	}

	if (!lnkifiles.empty() && !opts.hls) {
		auto lnknode = lnkcmd::create(pgraph.get(), lnkifiles, opts.lnkofile,
			opts.libpaths, opts.libs, opts.pthread);
		for (const auto & leave : leaves)
			leave->add_edge(lnknode);

		leaves.clear();
		leaves.push_back(lnknode);
	}

	for (const auto & leave : leaves)
		leave->add_edge(pgraph->exit());

	if (opts.only_print_commands) {
		std::unique_ptr<passgraph> pg(new passgraph());
		auto printnode = printcmd::create(pg.get(), std::move(pgraph));
		pg->entry()->add_edge(printnode);
		printnode->add_edge(pg->exit());
		pgraph = std::move(pg);
	}

	return pgraph;
}

/* parser command */

static std::string
create_prscmd_ofile(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-clang-out.ll");
}

prscmd::~prscmd()
{}

jlm::filepath
prscmd::ofile() const
{
	auto f = ifile_.base();
	return tmpfolder_.to_str()+create_prscmd_ofile(f);
}

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
prscmd::to_str() const
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

	if (hls_) {
		return strfmt(
		clangpath.to_str() + " "
		, arguments, " "
		, Wwarnings, " "
		, flags, " "
		, std_ != standard::none ? "-std="+jlm::to_str(std_)+" " : ""
		, replace_all(Dmacros, std::string("\""), std::string("\\\"")), " "
		, Ipaths, " "
		, "-S -emit-llvm -Xclang -disable-O0-optnone "
		, "-o ", tmpfolder_.to_str()+create_prscmd_ofile(f), " "
		, ifile_.to_str()
		);
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
	, "-o ", tmpfolder_.to_str()+create_prscmd_ofile(f), " "
	, ifile_.to_str()
	);
}

void
prscmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* optimization command */

static std::string
create_optcmd_ofile(const std::string & ifile)
{
	return strfmt("tmp-", ifile, "-jlm-opt-out.ll");
}

optcmd::~optcmd()
{}

std::string
optcmd::to_str() const
{
	auto f = ifile_.base();

	std::string jlmopts;
	for (const auto & jlmopt : jlmopts_)
		jlmopts += "--" + jlmopt + " ";

	/*
		If a default optimization level has been specified (-O) and no specific jlm-options 
		have been specified (-J) then use a default set of optimizations.
	 */
	if (jlmopts.empty()) {
		/*
			Only -O3 sets default optimizations
		*/
		if (ol_ == optlvl::O3) {
			jlmopts  = "--iln --inv --red --dne --ivt --inv --dne --psh --inv --dne ";
			jlmopts += "--red --cne --dne --pll --inv --dne --url --inv ";
		}
	}

	return strfmt(
	  "jlm-opt "
	, "--llvm "
	, jlmopts
	, tmpfolder_.to_str(), create_prscmd_ofile(f), " > "
	, tmpfolder_.to_str(), create_optcmd_ofile(f)
	);
}

void
optcmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* code generator command */

cgencmd::~cgencmd()
{}

std::string
cgencmd::to_str() const
{
	if (hls_) {
		return strfmt(
		llcpath.to_str() + " "
		, "-", jlm::to_str(ol_), " "
		, "--relocation-model=pic "
		, "-filetype=obj "
		, "-o ", ofile_.to_str()
		, " ", ifile_.to_str()
		);

	}

	return strfmt(
	llcpath.to_str() + " "
	, "-", jlm::to_str(ol_), " "
	, "-filetype=obj "
	, "-o ", ofile_.to_str()
	, " ", tmpfolder_.to_str(), create_optcmd_ofile(ifile_.base())
	);
}

void
cgencmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* linker command */

lnkcmd::~lnkcmd()
{}

std::string
lnkcmd::to_str() const
{
	std::string ifiles;
	for (const auto & ifile : ifiles_)
		ifiles += ifile.to_str() + " ";

	std::string Lpaths;
	for (const auto & Lpath : Lpaths_)
		Lpaths += "-L" + Lpath + " ";

	std::string libs;
	for (const auto & lib : libs_)
		libs += "-l" + lib + " ";

	std::string arguments;
	if (pthread_)
	  arguments += "-pthread ";

	return strfmt(
	  clangpath.to_str() + " "
	, "-no-pie -O0 "
        , arguments
	, ifiles
	, "-o ", ofile_.to_str(), " "
	, Lpaths
	, libs
	);
}

void
lnkcmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

/* print command */

printcmd::~printcmd()
{}

std::string
printcmd::to_str() const
{
	return "PRINTCMD";
}

void
printcmd::run() const
{
	for (const auto & node : topsort(pgraph_.get())) {
		if (node != pgraph_->entry() && node != pgraph_->exit())
			std::cout << node->cmd().to_str() << "\n";
	}
}

/* HLS */

std::string
lllnkcmd::to_str() const {

	auto llvm_link = clangpath.path() + "llvm-link";
	std::string ifiles;
	for (const auto & ifile : ifiles_)
		ifiles += ifile.to_str() + " ";
	return strfmt(
			llvm_link, " "
			, "-S -v "
			, "-o ", ofile_.to_str(), " "
			, ifiles
	);
}

void
lllnkcmd::run() const {
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

void
hlscmd::run() const {
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

std::string
hlscmd::to_str() const {
	if (circt_)
		return strfmt(
			      "jlm-opt"
			      , " --hls-file ", ifile().to_str()
			      , " --outfolder ", outfolder_
			      , " --circt "
			      );
	else
		return strfmt(
			      "jlm-opt"
			      , " --hls-file ", ifile().to_str()
			      , " --outfolder ", outfolder_
			      );
}

void
extractcmd::run() const {
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

std::string
extractcmd::to_str() const {
        return strfmt(
		      "jlm-opt"
		      , " --hls-file ", ifile().to_str()
		      , " --hls-function ", function()
		      , " --outfolder ", outfolder_
		      );
}

void
firrtlcmd::run() const {
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

std::string
firrtlcmd::to_str() const {
	return strfmt(
		      firtoolpath.to_str()
		      , " -format=fir --verilog "
		      , ifile().to_str()
		      , " > ", ofile().to_str()
		      );
}

void
verilatorcmd::run() const {
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

std::string
gcd() {
	char tmp[256];
	getcwd(tmp, 256);
	auto cd = std::string(tmp);
	return cd;
}

std::string
verilatorcmd::to_str() const {
	std::string lfiles;
	for (const auto & ifile : lfiles_)
		lfiles += ifile.to_str() + " ";

	std::string Lpaths;
	for (const auto & Lpath : Lpaths_)
		Lpaths += "-L" + Lpath + " ";

	std::string libs;
	for (const auto & lib : libs_)
		libs += "-l" + lib + " ";

	std::string cflags;
//	if(!libs.empty()||!Lpaths.empty()){
	cflags = " -CFLAGS \"" + libs + Lpaths + " -fPIC\"";
//	}

	std::string ofile = ofile_.to_str();
	if(ofile.at(0)!='/'){
		ofile = gcd() + "/" + ofile;
	}
        std::string verilator_root;
        if(verilatorrootpath.to_str().size()){
		verilator_root = strfmt(
					"VERILATOR_ROOT="
					, verilatorrootpath.to_str()
					, " "
					);
        }
	return strfmt(
		      verilator_root
		      ,verilatorpath.to_str()
		      , " --cc"
		      , " --build"
		      , " --exe"
#ifndef HLS_USE_VCD
		      , " --trace-fst"
#else
		      , " --trace"
#endif
		      , " -Wno-WIDTH" //divisions otherwise cause errors
		      , " -j"
		      , " -Mdir ", tmpfolder_.to_str(), "verilator/"
		      , " -MAKEFLAGS CXX=g++"
		      , " -CFLAGS -g" // TODO: switch for this
		      , " --assert"
		      , cflags
		      , " -o ", ofile
		      , " ", vfile().to_str()
		      , " ", hfile().to_str()
		      , " ", lfiles
		      );
}

jlm::filepath
m2rcmd::ofile() const
{
	return ofile_;
}

std::string
m2rcmd::to_str() const
{
	auto opt = clangpath.path() + "opt";

	return strfmt(
		      opt + " "
		      , "-mem2reg -S "
		      , "-o ", ofile().to_str(), " "
		      , ifile_.to_str()
		      );
}

void
m2rcmd::run() const
{
	if (system(to_str().c_str()))
		exit(EXIT_FAILURE);
}

std::string
mkdircmd::to_str() const
{
	return strfmt(
		      "mkdir "
		      , path_.to_str()
		      );
}

void
mkdircmd::run() const
{
	if (mkdir(path_.to_str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0)
		throw jlm::error("mkdir failed: "+path_.to_str());
}

} // jlm
