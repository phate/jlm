/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_RVSDG_MODULE_HPP
#define JLM_IR_RVSDG_MODULE_HPP

#include <jive/rvsdg/graph.h>

#include <jlm/ir/linkage.hpp>
#include <jlm/util/file.hpp>

namespace jlm {

/* impport class */

class impport final : public jive::impport {
public:
	virtual
	~impport();

	impport(
		const jive::type & type
	, const std::string & name
	, const jlm::linkage & lnk)
	: jive::impport(type, name)
	, linkage_(lnk)
	{}

	impport(const impport & other)
	: jive::impport(other)
	, linkage_(other.linkage_)
	{}

	impport(impport && other)
	: jive::impport(other)
	, linkage_(std::move(other.linkage_))
	{}

	impport&
	operator=(const impport&) = delete;

	impport&
	operator=(impport&&) = delete;

	const jlm::linkage &
	linkage() const noexcept
	{
		return linkage_;
	}

	virtual bool
	operator==(const port&) const noexcept override;

	virtual std::unique_ptr<port>
	copy() const override;

private:
	jlm::linkage linkage_;
};

static inline bool
is_import(const jive::output * output)
{
	auto graph = output->region()->graph();

	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && argument->region() == graph->root();
}

/* rvsdg module class */

class rvsdg_module final {
public:
	inline
	rvsdg_module(
		const jlm::filepath & source_filename,
		const std::string & target_triple,
		const std::string & data_layout)
	: data_layout_(data_layout)
	, target_triple_(target_triple)
	, source_filename_(source_filename)
	{}

	rvsdg_module(const rvsdg_module &) = delete;

	rvsdg_module(rvsdg_module &&) = delete;

	rvsdg_module &
	operator=(const rvsdg_module &) = delete;

	rvsdg_module &
	operator=(rvsdg_module &&) = delete;

	inline jive::graph *
	graph() noexcept
	{
		return &graph_;
	}

	inline const jive::graph *
	graph() const noexcept
	{
		return &graph_;
	}

	const jlm::filepath &
	source_filename() const noexcept
	{
		return source_filename_;
	}

	inline const std::string &
	target_triple() const noexcept
	{
		return target_triple_;
	}

	inline const std::string &
	data_layout() const noexcept
	{
		return data_layout_;
	}

	static std::unique_ptr<rvsdg_module>
	create(
		const jlm::filepath & source_filename,
		const std::string & target_triple,
		const std::string & data_layout)
	{
		return std::make_unique<rvsdg_module>(source_filename, target_triple, data_layout);
	}

private:
	jive::graph graph_;
	std::string data_layout_;
	std::string target_triple_;
	const jlm::filepath source_filename_;
};

}

#endif
