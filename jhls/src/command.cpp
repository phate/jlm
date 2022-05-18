/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/llvmpaths.hpp>
#include <jhls/command.hpp>
#include <jhls/toolpaths.hpp>
#include <jlm/util/strfmt.hpp>

#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <algorithm>
#include <sys/stat.h>

namespace jlm {


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
prscmd::Run() const
{
	if (system(ToString().c_str()))
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
optcmd::ToString() const
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
optcmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

/* code generator command */

cgencmd::~cgencmd()
{}

std::string
cgencmd::ToString() const
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
cgencmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

/* linker command */

lnkcmd::~lnkcmd()
{}

std::string
lnkcmd::ToString() const
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
lnkcmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

/* print command */

printcmd::~printcmd()
{}

std::string
printcmd::ToString() const
{
	return "PRINTCMD";
}

void
printcmd::Run() const
{
	for (const auto & node : topsort(pgraph_.get())) {
		if (node != pgraph_->entry() && node != pgraph_->exit())
			std::cout << node->cmd().ToString() << "\n";
	}
}

/* HLS */

std::string
lllnkcmd::ToString() const {

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
lllnkcmd::Run() const {
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

void
hlscmd::Run() const {
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

std::string
hlscmd::ToString() const {
	if (circt_)
		return strfmt(
			      "jlm-hls"
			      , " -o ", outfolder_
			      , " --circt "
			      , ifile().to_str()
			      //, " --hls-file ", ifile().to_str()
			      );
	else
		return strfmt(
			      "jlm-hls"
			      , " -o ", outfolder_, " "
			      , ifile().to_str()
			      //, " --hls-file ", ifile().to_str()
			      );
}

void
extractcmd::Run() const {
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

std::string
extractcmd::ToString() const {
        return strfmt(
		      "jlm-hls"
		      , " --extract"
		      , " --hls-function ", function()
		      , " -o ", outfolder_, " "
		      , ifile().to_str()
		      //, " --hls-file ", ifile().ToString()
		      );
}

void
firrtlcmd::Run() const {
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

std::string
firrtlcmd::ToString() const {
	return strfmt(
		      firtoolpath.to_str()
		      , " -format=fir --verilog "
		      , ifile().to_str()
		      , " > ", ofile().to_str()
		      );
}

void
verilatorcmd::Run() const {
	if (system(ToString().c_str()))
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
verilatorcmd::ToString() const {
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
m2rcmd::ToString() const
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
m2rcmd::Run() const
{
	if (system(ToString().c_str()))
		exit(EXIT_FAILURE);
}

std::string
mkdircmd::ToString() const
{
	return strfmt(
		      "mkdir "
		      , path_.to_str()
		      );
}

void
mkdircmd::Run() const
{
	if (mkdir(path_.to_str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0)
		throw jlm::error("mkdir failed: "+path_.to_str());
}

} // jlm
