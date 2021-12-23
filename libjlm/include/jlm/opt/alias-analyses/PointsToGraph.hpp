/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP
#define JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP

#include <jlm/common.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/ir/rvsdg-module.hpp>

#include <jive/rvsdg/node.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace jive {
	class argument;
	class node;
	class output;
}

namespace jlm {

class rvsdg_module;

namespace aa {

/** /brief Points-to graph
*
* FIXME: write documentation
*/
class ptg final {
	class constiterator;
	class iterator;

public:
	class allocator;
	class edge;
	class impnode;
	class memnode;
	class node;
	class regnode;
	class unknown;

	using allocnodemap = std::unordered_map<const jive::node*, std::unique_ptr<ptg::allocator>>;
	using impnodemap = std::unordered_map<const jive::argument*, std::unique_ptr<ptg::impnode>>;
	using regnodemap = std::unordered_map<const jive::output*, std::unique_ptr<ptg::regnode>>;

	using allocnode_range = iterator_range<allocnodemap::iterator>;
	using allocnode_constrange = iterator_range<allocnodemap::const_iterator>;

	using impnode_range = iterator_range<impnodemap::iterator>;
	using impnode_constrange = iterator_range<impnodemap::const_iterator>;

	using regnode_range = iterator_range<regnodemap::iterator>;
	using regnode_constrange = iterator_range<regnodemap::const_iterator>;

private:
	ptg();

	ptg(const jlm::aa::ptg&) = delete;

	ptg(jlm::aa::ptg&&) = delete;

	jlm::aa::ptg &
	operator=(const jlm::aa::ptg&) = delete;

	jlm::aa::ptg &
	operator=(jlm::aa::ptg&&) = delete;

public:
	allocnode_range
	allocnodes();

	allocnode_constrange
	allocnodes() const;

	impnode_range
	impnodes();

	impnode_constrange
	impnodes() const;

	regnode_range
	regnodes();

	regnode_constrange
	regnodes() const;

	iterator
	begin();

	constiterator
	begin() const;

	iterator
	end();

	constiterator
	end() const;

	size_t
	nallocnodes() const noexcept
	{
		return allocnodes_.size();
	}

	size_t
	nimpnodes() const noexcept
	{
		return impnodes_.size();
	}

	size_t
	nregnodes() const noexcept
	{
		return regnodes_.size();
	}

	size_t
	nnodes() const noexcept
	{
		return nallocnodes() + nimpnodes() + nregnodes();
	}

	jlm::aa::ptg::unknown &
	memunknown() const noexcept
	{
		return *memunknown_;
	}

	/*
		FIXME: I would like to call this function memnode() or node().
	*/
	const ptg::allocator &
	find(const jive::node * node) const
	{
		auto it = allocnodes_.find(node);
		if (it == allocnodes_.end())
			throw error("Cannot find memory node in points-to graph.");

		return *it->second;
	}

	const ptg::impnode &
	find(const jive::argument * argument) const
	{
		auto it = impnodes_.find(argument);
		if (it == impnodes_.end())
			throw error("Cannot find import node in points-to graph.");

		return *it->second;
	}

	/*
		FIXME: I would like to call this function regnode() or node().
	*/
	const ptg::regnode &
	find_regnode(const jive::output * output) const
	{
		auto it = regnodes_.find(output);
		if (it == regnodes_.end())
			throw error("Cannot find register node in points-to graph.");

		return *it->second;
	}

	/*
		FIXME: change return value to ptg::node &
	*/
	jlm::aa::ptg::node *
	add(std::unique_ptr<ptg::allocator> node);

	/*
		FIXME: change return value to ptg::node &
	*/
	jlm::aa::ptg::node *
	add(std::unique_ptr<ptg::regnode> node);

	ptg::node *
	add(std::unique_ptr<ptg::impnode> node);

	static std::string
	to_dot(const jlm::aa::ptg & ptg);

	static std::unique_ptr<ptg>
	create()
	{
		return std::unique_ptr<jlm::aa::ptg>(new ptg());
	}

private:
	impnodemap impnodes_;
	regnodemap regnodes_;
	allocnodemap allocnodes_;
	std::unique_ptr<jlm::aa::ptg::unknown> memunknown_;
};


/** \brief Points-to graph node
*
* FIXME: write documentation
*/
class ptg::node {
	class constiterator;
	class iterator;

	using node_range = iterator_range<ptg::node::iterator>;
	using node_constrange = iterator_range<ptg::node::constiterator>;

public:
	virtual
	~node();

	/*
		FIXME: change to ptg &
	*/
	node(jlm::aa::ptg * ptg)
	: ptg_(ptg)
	{}

	node(const node&) = delete;

	node(node&&) = delete;

	node&
	operator=(const node&) = delete;

	node&
	operator=(node&&) = delete;

	node_range
	targets();

	node_constrange
	targets() const;

	node_range
	sources();

	node_constrange
	sources() const;

	/*
		FIXME: change to ptg &
	*/
	jlm::aa::ptg *
	Graph() const noexcept
	{
		return ptg_;
	}

	size_t
	ntargets() const noexcept
	{
		return targets_.size();
	}

	size_t
	nsources() const noexcept
	{
		return sources_.size();
	}

	virtual std::string
	debug_string() const = 0;

	/*
		FIXME: change to ptg::node &
		FIXME: I believe that this can only be a memnode. If so, make it explicit in the type.
	*/
	void
	add_edge(ptg::node * target);

	void
	remove_edge(ptg::node * target);

private:
	jlm::aa::ptg * ptg_;
	std::unordered_set<ptg::node*> targets_;
	std::unordered_set<ptg::node*> sources_;
};


/** \brief Points-to graph register node
*
* FIXME: write documentation
*/
class ptg::regnode final : public ptg::node {
public:
	~regnode() override;

private:
	regnode(
		jlm::aa::ptg * ptg,
		const jive::output * output)
	: node(ptg)
	, output_(output)
	{}

public:
	const jive::output *
	output() const noexcept
	{
		return output_;
	}

	virtual std::string
	debug_string() const override;

	/**
		FIXME: write documentation
	*/
	static std::vector<const ptg::memnode*>
	allocators(const ptg::regnode & node);

	static ptg::regnode *
	create(jlm::aa::ptg * ptg, const jive::output * output)
	{
		auto node = std::unique_ptr<ptg::regnode>(new regnode(ptg, output));
		return static_cast<regnode*>(ptg->add(std::move(node)));
	}

private:
	const jive::output * output_;
};


/** \brief Points-to graph memory node
*
* FIXME: write documentation
*
* FIXME: Add final and convert protected to private after unknown inheritance is resolved.
*/
class ptg::memnode : public ptg::node {
public:
	~memnode() override;

protected:
	memnode(jlm::aa::ptg * ptg)
	: node(ptg)
	{}
};


/**
* FIXME: write documentation
*/
class ptg::allocator final : public ptg::memnode {
public:
	~allocator() override;

private:
	allocator(
		jlm::aa::ptg * ptg
	, const jive::node * node)
	: memnode(ptg)
	, node_(node)
	{}

public:
	const jive::node *
	node() const noexcept
	{
		return node_;
	}

	virtual std::string
	debug_string() const override;

	static jlm::aa::ptg::allocator *
	create(
		jlm::aa::ptg * ptg
	, const jive::node * node)
	{
		auto n = std::unique_ptr<ptg::allocator>(new allocator(ptg, node));
		return static_cast<jlm::aa::ptg::allocator*>(ptg->add(std::move(n)));
	}

private:
	const jive::node * node_;
};

/** \brief FIXME: write documentation
*
*/
class ptg::impnode final : public ptg::memnode {
public:
	~impnode() override;

private:
	impnode(
		jlm::aa::ptg * ptg
	, const jive::argument * argument)
	: memnode(ptg)
	, argument_(argument)
	{
		JLM_ASSERT(dynamic_cast<const jlm::impport*>(&argument->port()));
	}

public:
	const jive::argument *
	argument() const noexcept
	{
		return argument_;
	}

	virtual std::string
	debug_string() const override;

	static jlm::aa::ptg::impnode *
	create(
		jlm::aa::ptg * ptg
	, const jive::argument * argument)
	{
		auto n = std::unique_ptr<ptg::impnode>(new impnode(ptg, argument));
		return static_cast<jlm::aa::ptg::impnode*>(ptg->add(std::move(n)));
	}

private:
	const jive::argument * argument_;
};

/**
*
* FIXME: write documentation
*/
class ptg::unknown final : public ptg::memnode {
	friend jlm::aa::ptg;

public:
	~unknown() override;

private:
	unknown(jlm::aa::ptg * ptg)
	: memnode(ptg)
	{}

	virtual std::string
	debug_string() const override;
};


/** \brief Points-to graph node iterator
*/
class ptg::iterator final : public std::iterator<std::forward_iterator_tag,
	ptg::node*, ptrdiff_t> {

	friend jlm::aa::ptg;

	iterator(
		allocnodemap::iterator anit,
		impnodemap::iterator init,
		regnodemap::iterator rnit,
		/* FIXME: the full ranges are unnecessary here */
		const allocnode_range & anrange,
		const impnode_range & inrange,
		const regnode_range & rnrange)
	: anit_(anit)
	, init_(init)
	, rnit_(rnit)
	, anrange_(anrange)
	, inrange_(inrange)
	, rnrange_(rnrange)
	{}

public:
	ptg::node *
	node() const noexcept
	{
		if (anit_ != anrange_.end())
			return anit_->second.get();

		if (init_ != inrange_.end())
			return init_->second.get();

		return rnit_->second.get();
	}

	ptg::node &
	operator*() const
	{
		JLM_ASSERT(node() != nullptr);
		return *node();
	}

	ptg::node *
	operator->() const
	{
		return node();
	}

	iterator &
	operator++()
	{
		if (anit_ != anrange_.end()) {
			++anit_;
			return *this;
		}

		if (init_ != inrange_.end()) {
			++init_;
			return *this;
		}

		++rnit_;
		return *this;
	}

	iterator
	operator++(int)
	{
		iterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const iterator & other) const
	{
		return anit_ == other.anit_
		    && init_ == other.init_
		    && rnit_ == other.rnit_;
	}

	bool
	operator!=(const iterator & other) const
	{
		return !operator==(other);
	}

private:
	allocnodemap::iterator anit_;
	impnodemap::iterator init_;
	regnodemap::iterator rnit_;

	allocnode_range anrange_;
	impnode_range inrange_;
	regnode_range rnrange_;
};

/** \brief Points-to graph node const iterator
*/
class ptg::constiterator final : public std::iterator<std::forward_iterator_tag,
	const ptg::node*, ptrdiff_t> {

	friend jlm::aa::ptg;

	constiterator(
		allocnodemap::const_iterator anit,
		impnodemap::const_iterator init,
		regnodemap::const_iterator rnit,
		/* FIXME: the full ranges are unnecessary here */
		const allocnode_constrange & anrange,
		const impnode_constrange & inrange,
		const regnode_constrange & rnrange)
	: anit_(anit)
	, init_(init)
	, rnit_(rnit)
	, anrange_(anrange)
	, inrange_(inrange)
	, rnrange_(rnrange)
	{}

public:
	const ptg::node *
	node() const noexcept
	{
		if (anit_ != anrange_.end())
			return anit_->second.get();

		if (init_ != inrange_.end())
			return init_->second.get();

		return rnit_->second.get();
	}

	const ptg::node &
	operator*() const
	{
		JLM_ASSERT(node() != nullptr);
		return *node();
	}

	const ptg::node *
	operator->() const
	{
		return node();
	}

	constiterator &
	operator++()
	{
		if (anit_ != anrange_.end()) {
			++anit_;
			return *this;
		}

		if (init_ != inrange_.end()) {
			++init_;
			return *this;
		}

		++rnit_;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return anit_ == other.anit_
		    && init_ == other.init_
		    && rnit_ == other.rnit_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	allocnodemap::const_iterator anit_;
	impnodemap::const_iterator init_;
	regnodemap::const_iterator rnit_;

	allocnode_constrange anrange_;
	impnode_constrange inrange_;
	regnode_constrange rnrange_;
};


/** \brief Points-to graph edge iterator
*/
class ptg::node::iterator final : public std::iterator<std::forward_iterator_tag,
	ptg::node*, ptrdiff_t> {

	friend ptg::node;

	iterator(const std::unordered_set<ptg::node*>::iterator & it)
	: it_(it)
	{}

public:
	ptg::node *
	target() const noexcept
	{
		return *it_;
	}

	ptg::node &
	operator*() const
	{
		JLM_ASSERT(target() != nullptr);
		return *target();
	}

	ptg::node *
	operator->() const
	{
		return target();
	}

	iterator &
	operator++()
	{
		++it_;
		return *this;
	}

	iterator
	operator++(int)
	{
		iterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const iterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const iterator & other) const
	{
		return !operator==(other);
	}

private:
	std::unordered_set<ptg::node*>::iterator it_;
};


/** \brief Points-to graph edge const iterator
*/
class ptg::node::constiterator final : public std::iterator<std::forward_iterator_tag,
	const ptg::node*, ptrdiff_t> {

	friend jlm::aa::ptg;

	constiterator(const std::unordered_set<ptg::node*>::const_iterator & it)
	: it_(it)
	{}

public:
	const ptg::node *
	target() const noexcept
	{
		return *it_;
	}

	const ptg::node &
	operator*() const
	{
		JLM_ASSERT(target() != nullptr);
		return *target();
	}

	const ptg::node *
	operator->() const
	{
		return target();
	}

	constiterator &
	operator++()
	{
		++it_;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	std::unordered_set<ptg::node*>::const_iterator it_;
};

}}

#endif
