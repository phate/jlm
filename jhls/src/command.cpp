/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/llvmpaths.hpp>
#include <jhls/command.hpp>
#include <jhls/toolpaths.hpp>
#include <jlm/util/strfmt.hpp>

#include <functional>
#include <iostream>
#include <memory>

namespace jlm {

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

} // jlm
