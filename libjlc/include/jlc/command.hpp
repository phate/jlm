/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_COMMAND_HPP
#define JLM_JLC_COMMAND_HPP

#include <jlc/cmdline.hpp>
#include <jlm/driver/passgraph.hpp>

#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

namespace jlm {

std::unique_ptr<passgraph>
generate_commands(const jlm::cmdline_options & options);

/* parser command */

class prscmd final : public command {
public:
	virtual
	~prscmd();

	prscmd(
		const jlm::filepath & ifile,
		const jlm::filepath & dependencyFile,
		const jlm::filepath & tmpfolder,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const std::vector<std::string> & flags,
		bool verbose,
		bool rdynamic,
		bool suppress,
		bool pthread,
		bool MD,
		bool hls,
		const std::string & mT,
		const standard & std)
	: std_(std)
	, ifile_(ifile)
	, tmpfolder_(tmpfolder)
	, Ipaths_(Ipaths)
	, Dmacros_(Dmacros)
	, Wwarnings_(Wwarnings)
	, flags_(flags)
	, verbose_(verbose)
	, rdynamic_(rdynamic)
	, suppress_(suppress)
	, pthread_(pthread)
	, MD_(MD)
	, hls_(hls)
	, mT_(mT)
	, dependencyFile_(dependencyFile)
	{}

	virtual std::string
	to_str() const override;

	jlm::filepath
	ofile() const;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const jlm::filepath & dependencyFile,
		const jlm::filepath & tmpfolder,
		const std::vector<std::string> & Ipaths,
		const std::vector<std::string> & Dmacros,
		const std::vector<std::string> & Wwarnings,
		const std::vector<std::string> & flags,
		bool verbose,
		bool rdynamic,
		bool suppress,
		bool pthread,
		bool MD,
		bool hls,
		const std::string & mT,
		const standard & std)
	{
		std::unique_ptr<prscmd> cmd(new prscmd(
			ifile,
			dependencyFile,
			tmpfolder,
			Ipaths,
			Dmacros,
			Wwarnings,
			flags,
			verbose,
			rdynamic,
			suppress,
			pthread,
			MD,
			hls,
			mT,
			std));

		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	static std::string
	replace_all(std::string str, const std::string& from, const std::string& to);

	standard std_;
	jlm::filepath ifile_;
	jlm::filepath tmpfolder_;
	std::vector<std::string> Ipaths_;
	std::vector<std::string> Dmacros_;
	std::vector<std::string> Wwarnings_;
	std::vector<std::string> flags_;
	bool verbose_;
	bool rdynamic_;
	bool suppress_;
	bool pthread_;
	bool MD_;
	bool hls_;
	std::string mT_;
	jlm::filepath dependencyFile_;
};

/* optimization command */

class optcmd final : public command {
public:
	virtual
	~optcmd();

	optcmd(
		const jlm::filepath & ifile,
		const jlm::filepath & tmpfolder,
		const std::vector<std::string> & jlmopts,
		const optlvl & ol)
	: ifile_(ifile)
	, tmpfolder_(tmpfolder)
	, jlmopts_(jlmopts)
	, ol_(ol)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const jlm::filepath & tmpfolder,
		const std::vector<std::string> & jlmopts,
		const optlvl & ol)
	{
		return passgraph_node::create(pgraph, std::make_unique<optcmd>(ifile, tmpfolder, jlmopts, ol));
	}

private:
	jlm::filepath ifile_;
	jlm::filepath tmpfolder_;
	std::vector<std::string> jlmopts_;
	optlvl ol_;
};

/* code generator command */

class cgencmd final : public command {
public:
	virtual
	~cgencmd();

	cgencmd(
		const jlm::filepath & ifile,
		const jlm::filepath & ofile,
		const jlm::filepath & tmpfolder,
		const bool hls,
		const optlvl & ol)
	: ol_(ol)
	, ifile_(ifile)
	, ofile_(ofile)
	, tmpfolder_(tmpfolder)
	, hls_(hls)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const jlm::filepath & ofile,
		const jlm::filepath & tmpfolder,
		const bool hls,
		const optlvl & ol)
	{
		return passgraph_node::create(pgraph, std::make_unique<cgencmd>(ifile, ofile, tmpfolder, hls, ol));
	}

private:
	optlvl ol_;
	jlm::filepath ifile_;
	jlm::filepath ofile_;
	jlm::filepath tmpfolder_;
	bool hls_;
};

/* linker command */

class lnkcmd final : public command {
public:
	virtual
	~lnkcmd();

	lnkcmd(
		const std::vector<jlm::filepath> & ifiles,
		const jlm::filepath & ofile,
		const std::vector<std::string> & Lpaths,
		const std::vector<std::string> & libs,
		bool pthread)
	: ofile_(ofile)
	, libs_(libs)
	, ifiles_(ifiles)
	, Lpaths_(Lpaths)
	, pthread_(pthread)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const std::vector<jlm::filepath> &
	ifiles() const noexcept
	{
		return ifiles_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const std::vector<jlm::filepath> & ifiles,
		const jlm::filepath & ofile,
		const std::vector<std::string> & Lpaths,
		const std::vector<std::string> & libs,
		bool pthread)
	{
		std::unique_ptr<lnkcmd> cmd(new lnkcmd(ifiles, ofile, Lpaths, libs, pthread));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	std::vector<std::string> libs_;
	std::vector<jlm::filepath> ifiles_;
	std::vector<std::string> Lpaths_;
	bool pthread_;
};

/* print command */

class printcmd final : public command {
public:
	virtual
	~printcmd();

	printcmd(
		std::unique_ptr<passgraph> pgraph)
	: pgraph_(std::move(pgraph))
	{}

	printcmd(const printcmd&) = delete;

	printcmd(printcmd&&) = delete;

	printcmd &
	operator=(const printcmd&) = delete;

	printcmd &
	operator=(printcmd&&)	= delete;

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		std::unique_ptr<passgraph> pg)
	{
		return passgraph_node::create(pgraph, std::make_unique<printcmd>(std::move(pg)));
	}

private:
	std::unique_ptr<passgraph> pgraph_;
};

class m2rcmd final : public command {
public:
	virtual
	~m2rcmd(){}

	m2rcmd(
		const jlm::filepath & ifile,
		const jlm::filepath & ofile)
	: ifile_(ifile),
	  ofile_(ofile)
	{}

	virtual std::string
	to_str() const override;

	jlm::filepath
	ofile() const;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const jlm::filepath & ofile)
	{
		std::unique_ptr<m2rcmd> cmd(new m2rcmd(ifile, ofile));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ifile_;
	jlm::filepath ofile_;
};

class mkdircmd final : public command {
public:
	virtual
	~mkdircmd(){}

    mkdircmd(
		const jlm::filepath & path)
	: path_(path)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & path)
	{
		std::unique_ptr<mkdircmd> cmd(new mkdircmd(path));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath path_;
};

class verilatorcmd final : public command {
public:
	virtual
	~verilatorcmd(){}

	verilatorcmd(
			const jlm::filepath & vfile,
			const std::vector<jlm::filepath> & lfiles,
			const jlm::filepath & hfile,
			const jlm::filepath & ofile,
            const jlm::filepath & tmpfolder,
			const std::vector<std::string> & Lpaths,
			const std::vector<std::string> & libs)
	: ofile_(ofile)
	, vfile_(vfile)
	, hfile_(hfile)
	, tmpfolder_(tmpfolder)
	, libs_(libs)
	, lfiles_(lfiles)
	, Lpaths_(Lpaths)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath &
	vfile() const noexcept
	{
		return vfile_;
	}

	inline const jlm::filepath &
	hfile() const noexcept
	{
		return hfile_;
	}

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const std::vector<jlm::filepath> &
	lfiles() const noexcept
	{
		return lfiles_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & vfile,
		const std::vector<jlm::filepath> & lfiles,
		const jlm::filepath & hfile,
		const jlm::filepath & ofile,
		const jlm::filepath & tmpfolder,
		const std::vector<std::string> & Lpaths,
		const std::vector<std::string> & libs)
	{
		std::unique_ptr<verilatorcmd> cmd(new verilatorcmd(vfile, lfiles, hfile, ofile, tmpfolder, Lpaths, libs));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	jlm::filepath vfile_;
	jlm::filepath hfile_;
	jlm::filepath tmpfolder_;
	std::vector<std::string> libs_;
	std::vector<jlm::filepath> lfiles_;
	std::vector<std::string> Lpaths_;
};

class lllnkcmd final : public command {
public:
	virtual
	~lllnkcmd(){}

	lllnkcmd(
		const std::vector<jlm::filepath> & ifiles,
		const jlm::filepath & ofile)
	: ofile_(ofile)
	, ifiles_(ifiles)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const std::vector<jlm::filepath> &
	ifiles() const noexcept
	{
		return ifiles_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const std::vector<jlm::filepath> & ifiles,
		const jlm::filepath & ofile)
	{
		std::unique_ptr<lllnkcmd> cmd(new lllnkcmd(ifiles, ofile));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	std::vector<jlm::filepath> ifiles_;
};

class firrtlcmd final : public command {
public:
	virtual
	~firrtlcmd(){}

	firrtlcmd(
		const jlm::filepath & ifile,
		const jlm::filepath & ofile)
	: ofile_(ofile)
	, ifile_(ifile)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath &
	ofile() const noexcept
	{
		return ofile_;
	}

	inline const jlm::filepath &
	ifile() const noexcept
	{
		return ifile_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const jlm::filepath & ofile)
	{
		std::unique_ptr<firrtlcmd> cmd(new firrtlcmd(ifile, ofile));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ofile_;
	jlm::filepath ifile_;
};

std::basic_string<char>
gcd();

class hlscmd final : public command {
public:
	virtual
	~hlscmd(){}

	hlscmd(
		const jlm::filepath & ifile,
		const std::string & outfolder,
	        const bool &circt)
	: ifile_(ifile)
	, outfolder_(outfolder)
	, circt_(circt)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath
	firfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls.fir");
	}

	inline const jlm::filepath
	llfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls.rest.ll");
	}

	inline const jlm::filepath
	harnessfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls_harness.cpp");
	}

	inline const jlm::filepath &
	ifile() const noexcept
	{
		return ifile_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const std::string & outfolder,
	        const bool &circt)
	{
		std::unique_ptr<hlscmd> cmd(new hlscmd(ifile, outfolder, circt));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ifile_;
	std::string outfolder_;
	bool circt_;
};

class extractcmd final : public command {
public:
	virtual
	~extractcmd(){}

    extractcmd(
		const jlm::filepath & ifile,
		const std::string & function,
		const std::string & outfolder)
	: ifile_(ifile)
	, function_(function)
	, outfolder_(outfolder)
	{}

	virtual std::string
	to_str() const override;

	virtual void
	run() const override;

	inline const jlm::filepath
	functionfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls.function.ll");
	}

	inline const jlm::filepath
	llfile() const noexcept
	{
		return jlm::filepath(outfolder_+"jlm_hls.rest.ll");
	}

	inline const jlm::filepath &
	ifile() const noexcept
	{
		return ifile_;
	}
	inline const std::string &
	function() const noexcept
	{
		return function_;
	}

	static passgraph_node *
	create(
		passgraph * pgraph,
		const jlm::filepath & ifile,
		const std::string & function,
		const std::string & outfolder)
	{
		std::unique_ptr<extractcmd> cmd(new extractcmd(ifile, function, outfolder));
		return passgraph_node::create(pgraph, std::move(cmd));
	}

private:
	jlm::filepath ifile_;
	std::string function_;
	std::string outfolder_;
};

}

#endif
