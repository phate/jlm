/*
 * Copyright 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_RVSDG_PHI_HPP
#define JIVE_RVSDG_PHI_HPP

#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/node.hpp>
#include <jive/rvsdg/region.hpp>
#include <jive/rvsdg/structural-node.hpp>

namespace jive {
namespace phi {

/* phi operation class  */

class operation final : public structural_op {
public:
	~operation() override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

/* phi node class */

class builder;
class cvargument;
class cvinput;
class rvoutput;

class node final : public structural_node {
	friend phi::builder;

	class cvconstiterator final : public std::iterator<std::forward_iterator_tag,
		const cvinput*, ptrdiff_t> {

		friend phi::node;

		cvconstiterator(const cvinput * input)
		: input_(input)
		{}

	public:
		const cvinput *
		input() const noexcept
		{
			return input_;
		}

		const cvinput &
		operator*() const
		{
			JIVE_DEBUG_ASSERT(input_ != nullptr);
			return *input_;
		}

		const cvinput *
		operator->() const
		{
			return input_;
		}

		cvconstiterator &
		operator++();

		cvconstiterator
		operator++(int)
		{
			cvconstiterator tmp = *this;
			++*this;
			return tmp;
		}

		bool
		operator==(const cvconstiterator & other) const
		{
			return input_ == other.input_;
		}

		bool
		operator!=(const cvconstiterator & other) const
		{
			return !operator==(other);
		}

	private:
		const cvinput * input_;
	};

	class cviterator final : public std::iterator<std::forward_iterator_tag,
		cvinput*, ptrdiff_t> {

		friend phi::node;

		cviterator(cvinput * input)
		: input_(input)
		{}

	public:
		cvinput *
		input() const noexcept
		{
			return input_;
		}

		cvinput &
		operator*() const
		{
			JIVE_DEBUG_ASSERT(input_ != nullptr);
			return *input_;
		}

		cvinput *
		operator->() const
		{
			return input_;
		}

		cviterator &
		operator++();

		cviterator
		operator++(int)
		{
			cviterator tmp = *this;
			++*this;
			return tmp;
		}

		bool
		operator==(const cviterator & other) const
		{
			return input_ == other.input_;
		}

		bool
		operator!=(const cviterator & other) const
		{
			return !operator==(other);
		}

	private:
		cvinput * input_;
	};

	class rvconstiterator final : public std::iterator<std::forward_iterator_tag,
		const rvoutput*, ptrdiff_t> {

		friend phi::node;

		rvconstiterator(const rvoutput * output)
		: output_(output)
		{}

	public:
		const rvoutput *
		output() const noexcept
		{
			return output_;
		}

		const rvoutput &
		operator*() const
		{
			JIVE_DEBUG_ASSERT(output_ != nullptr);
			return *output_;
		}

		const rvoutput *
		operator->() const
		{
			return output_;
		}

		rvconstiterator &
		operator++();

		rvconstiterator
		operator++(int)
		{
			rvconstiterator tmp = *this;
			++*this;
			return tmp;
		}

		bool
		operator==(const rvconstiterator & other) const
		{
			return output_ == other.output_;
		}

		bool
		operator!=(const rvconstiterator & other) const
		{
			return !operator==(other);
		}

	private:
		const rvoutput * output_;
	};

	class rviterator final : public std::iterator<std::forward_iterator_tag,
		rvoutput*, ptrdiff_t> {

		friend phi::node;

		rviterator(rvoutput * output)
		: output_(output)
		{}

	public:
		rvoutput *
		output() const noexcept
		{
			return output_;
		}

		rvoutput &
		operator*() const
		{
			JIVE_DEBUG_ASSERT(output_ != nullptr);
			return *output_;
		}

		rvoutput *
		operator->() const
		{
			return output_;
		}

		rviterator &
		operator++();

		rviterator
		operator++(int)
		{
			rviterator tmp = *this;
			++*this;
			return tmp;
		}

		bool
		operator==(const rviterator & other) const
		{
			return output_ == other.output_;
		}

		bool
		operator!=(const rviterator & other) const
		{
			return !operator==(other);
		}

	private:
		rvoutput * output_;
	};

public:
	~node() override;

private:
	node(
		jive::region * parent,
		const phi::operation & op)
	: structural_node(op, parent, 1)
	{}

	static phi::node *
	create(
		jive::region * parent,
		const phi::operation & op)
	{
		return new phi::node(parent, op);
	}

public:
	cvconstiterator
	begin_cv() const
	{
		if (ninputs() == 0)
			return end_cv();

		return cvconstiterator(input(0));
	}

	cviterator
	begin_cv()
	{
		if (ninputs() == 0)
			return end_cv();

		return cviterator(input(0));
	}

	cvconstiterator
	end_cv() const
	{
		return cvconstiterator(nullptr);
	}

	cviterator
	end_cv()
	{
		return cviterator(nullptr);
	}

	rvconstiterator
	begin_rv() const
	{
		if (noutputs() == 0)
			return end_rv();

		return rvconstiterator(output(0));
	}

	rviterator
	begin_rv()
	{
		if (noutputs() == 0)
			return end_rv();

		return rviterator(output(0));
	}

	rvconstiterator
	end_rv() const
	{
		return rvconstiterator(nullptr);
	}

	rviterator
	end_rv()
	{
		return rviterator(nullptr);
	}

	jive::region *
	subregion() const noexcept
	{
		return structural_node::subregion(0);
	}

	const phi::operation &
	operation() const noexcept
	{
		return *static_cast<const phi::operation*>(&jive::node::operation());
	}

	cvargument *
	add_ctxvar(jive::output * origin);

	cvinput *
	input(size_t n) const noexcept;

	rvoutput *
	output(size_t n) const noexcept;

	virtual phi::node *
	copy(jive::region * region, jive::substitution_map & smap) const override;
};

/* phi builder class */

class rvoutput;

class builder final {
public:
	constexpr
	builder() noexcept
	: node_(nullptr)
	{}

	jive::region *
	subregion() const noexcept
	{
		return node_ ? node_->subregion() : nullptr;
	}

	void
	begin(jive::region * parent)
	{
		if (node_)
			return;

		node_ = phi::node::create(parent, phi::operation());
	}

	phi::cvargument *
	add_ctxvar(jive::output * origin)
	{
		if (!node_)
			return nullptr;

		return node_->add_ctxvar(origin);
	}

	phi::rvoutput *
	add_recvar(const jive::type & type);

	phi::node *
	end();

private:
	phi::node * node_;
};

/* phi context variable input class */

class cvinput final : public structural_input {
	friend phi::node;

public:
	~cvinput() override;

private:
	cvinput(
		phi::node * node,
		jive::output * origin,
		const jive::port & port)
	: structural_input(node, origin, port)
	{}

	cvinput(const cvinput&) = delete;

	cvinput(cvinput&&) = delete;

	cvinput&
	operator=(const cvinput&) = delete;

	cvinput&
	operator=(cvinput&&) = delete;

	static cvinput *
	create(
		phi::node * node,
		jive::output * origin,
		const jive::port & port)
	{
		auto input = std::unique_ptr<cvinput>(new cvinput(node, origin, port));
		return static_cast<cvinput*>(node->append_input(std::move(input)));
	}

public:
	cvargument *
	argument() const noexcept;

	phi::node *
	node() const noexcept
	{
		return static_cast<phi::node*>(structural_input::node());
	}
};

/* phi recursion variable output class */

class rvargument;
class rvresult;

class rvoutput final : public structural_output {
 friend phi::builder;

public:
	~rvoutput() override;

private:
	rvoutput(
		phi::node * node,
		rvargument * argument,
		const jive::port & port)
	: structural_output(node, port)
	, argument_(argument)
	{}

	rvoutput(const rvoutput&) = delete;

	rvoutput(rvoutput&&) = delete;

	rvoutput&
	operator=(const rvoutput&) = delete;

	rvoutput&
	operator=(rvoutput&&) = delete;

	static rvoutput *
	create(
		phi::node * node,
		rvargument * argument,
		const jive::port & port);

public:
	rvargument *
	argument() const noexcept
	{
		return argument_;
	}

	rvresult *
	result() const noexcept;

	void
	set_rvorigin(jive::output * origin);

	phi::node *
	node() const noexcept
	{
		return static_cast<phi::node*>(structural_output::node());
	}

private:
	rvargument * argument_;
};

/* phi recursion variable argument class */

class rvresult;

class rvargument final : public argument {
	friend phi::builder;
	friend phi::rvoutput;

public:
	~rvargument() override;

private:
	rvargument(
		jive::region * region,
		const jive::port & port)
	: argument(region, nullptr, port)
	, output_(nullptr)
	{}

	rvargument(const rvargument&) = delete;

	rvargument(rvargument&&) = delete;

	rvargument&
	operator=(const rvargument&) = delete;

	rvargument&
	operator=(rvargument&&) = delete;

	static rvargument *
	create(
		jive::region * region,
		const jive::port & port)
	{
		auto argument = new rvargument(region, port);
		region->append_argument(argument);
		return argument;
	}

public:
	rvoutput *
	output() const noexcept
	{
		JIVE_DEBUG_ASSERT(output_ != nullptr);
		return output_;
	}

	rvresult *
	result() const noexcept
	{
		return output()->result();
	}

private:
	rvoutput * output_;
};

/* phi context variable argument class */

class cvinput;
class node;

class cvargument final : public argument {
	friend phi::node;

public:
	~cvargument() override;

private:
	cvargument(
		jive::region * region,
		phi::cvinput * input,
		const jive::port & port)
	: jive::argument(region, input, port)
	{}

	cvargument(const cvargument&) = delete;

	cvargument(cvargument&&) = delete;

	cvargument&
	operator=(const cvargument&) = delete;

	cvargument&
	operator=(cvargument&&) = delete;

	static cvargument *
	create(
		jive::region * region,
		phi::cvinput * input,
		const jive::port & port)
	{
		auto argument = new cvargument(region, input, port);
		region->append_argument(argument);
		return argument;
	}

public:
	cvinput *
	input() const noexcept
	{
		return static_cast<cvinput*>(argument::input());
	}
};

/* phi recursion variable result class */

class rvresult final : public result {
	friend phi::builder;

public:
	~rvresult() override;

private:
	rvresult(
		jive::region * region,
		jive::output * origin,
		rvoutput * output,
		const jive::port & port)
	: jive::result(region, origin, output, port)
	{}

	rvresult(const rvresult&) = delete;

	rvresult(rvresult&&) = delete;

	rvresult&
	operator=(const rvresult&) = delete;

	rvresult&
	operator=(rvresult&&) = delete;

	static rvresult *
	create(
		jive::region * region,
		jive::output * origin,
		rvoutput * output,
		const jive::port & port)
	{
		auto result = new rvresult(region, origin, output, port);
		region->append_result(result);
		return result;
	}

public:
	rvoutput *
	output() const noexcept
	{
		return static_cast<rvoutput*>(result::output());
	}

	rvargument *
	argument() const noexcept
	{
		return output()->argument();
	}
};

/* method definitions */

inline node::cvconstiterator &
node::cvconstiterator::operator++()
{
	if (input_ == nullptr)
		return *this;

	auto node = input_->node();
	auto index = input_->index();
	JIVE_DEBUG_ASSERT(node->ninputs() != 0);
	input_ = node->ninputs()-1 == index ? nullptr : node->input(index+1);

	return *this;
}

inline node::cviterator &
node::cviterator::operator++()
{
	if (input_ == nullptr)
		return *this;

	auto node = input_->node();
	auto index = input_->index();
	JIVE_DEBUG_ASSERT(node->ninputs() != 0);
	input_ = node->ninputs()-1 == index ? nullptr : node->input(index+1);

	return *this;
}

inline node::rvconstiterator &
node::rvconstiterator::operator++()
{
	if (output_ == nullptr)
		return *this;

	auto index = output_->index();
	auto node = output_->node();
	JIVE_DEBUG_ASSERT(node->noutputs() != 0);
	output_ = node->noutputs()-1 == index ? nullptr : node->output(index+1);

	return *this;
}

inline node::rviterator &
node::rviterator::operator++()
{
	if (output_ == nullptr)
		return *this;

	auto index = output_->index();
	auto node = output_->node();
	JIVE_DEBUG_ASSERT(node->noutputs() != 0);
	output_ = node->noutputs()-1 == index ? nullptr : node->output(index+1);

	return *this;
}

inline cvargument *
cvinput::argument() const noexcept
{
	JIVE_DEBUG_ASSERT(arguments.size() == 1);
	return static_cast<cvargument*>(arguments.first());
}

inline rvoutput *
rvoutput::create(
	phi::node * node,
	rvargument * argument,
	const jive::port & port)
{
	JIVE_DEBUG_ASSERT(argument->type() == port.type());
	auto output = std::unique_ptr<rvoutput>(new rvoutput(node, argument, port));
	return static_cast<rvoutput*>(node->append_output(std::move(output)));
}

inline rvresult *
rvoutput::result() const noexcept
{
	JIVE_DEBUG_ASSERT(results.size() == 1);
	return static_cast<rvresult*>(results.first());
}

inline void
rvoutput::set_rvorigin(jive::output * origin)
{
	JIVE_DEBUG_ASSERT(result()->origin() == argument());
	result()->divert_to(origin);
}

}}

#endif
