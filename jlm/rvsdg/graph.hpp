/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GRAPH_HPP
#define JLM_RVSDG_GRAPH_HPP

#include <stdbool.h>
#include <stdlib.h>

#include <typeindex>

#include <jlm/rvsdg/node-normal-form.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/tracker.hpp>

#include <jlm/util/common.hpp>

namespace jive {

/* impport class */

class impport : public port {
public:
	virtual
	~impport();

	impport(
		const jive::type & type,
		const std::string & name)
	: port(type)
	, name_(name)
	{}

	impport(const impport & other)
	: port(other)
	, name_(other.name_)
	{}

	impport(impport && other)
	: port(other)
	, name_(std::move(other.name_))
	{}

	impport&
	operator=(const impport&) = delete;

	impport&
	operator=(impport&&) = delete;

	const std::string &
	name() const noexcept
	{
		return name_;
	}

	virtual bool
	operator==(const port&) const noexcept override;

	virtual std::unique_ptr<port>
	copy() const override;

private:
	std::string name_;
};

/* expport class */

class expport : public port {
public:
	virtual
	~expport();

	expport(
		const jive::type & type,
		const std::string & name)
	: port(type)
	, name_(name)
	{}

	expport(const expport & other)
	: port(other)
	, name_(other.name_)
	{}

	expport(expport && other)
	: port(other)
	, name_(std::move(other.name_))
	{}

	expport&
	operator=(const expport&) = delete;

	expport&
	operator=(expport&&) = delete;

	const std::string &
	name() const noexcept
	{
		return name_;
	}

	virtual bool
	operator==(const port&) const noexcept override;

	virtual std::unique_ptr<port>
	copy() const override;

private:
	std::string name_;
};

/* graph */

class graph {
public:
	~graph();

	graph();

	inline jive::region *
	root() const noexcept
	{
		return root_;
	}

	inline void
	mark_denormalized() noexcept
	{
		normalized_ = false;
	}

	inline void
	normalize()
	{
		root()->normalize(true);
		normalized_ = true;
	}

	std::unique_ptr<jive::graph>
	copy() const;

	jive::node_normal_form *
	node_normal_form(const std::type_info & type) noexcept;

	inline jive::argument *
	add_import(const impport & port)
	{
		return argument::create(root(), nullptr, port);
	}

	inline jive::input *
	add_export(jive::output * operand, const expport & port)
	{
		return result::create(root(), operand, nullptr, port);
	}

	inline void
	prune()
	{
		root()->prune(true);
	}

private:
	bool normalized_;
	jive::region * root_;
	jive::node_normal_form_hash node_normal_forms_;
};

}

#endif
