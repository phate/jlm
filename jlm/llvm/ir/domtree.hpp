/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_DOMTREE_HPP
#define JLM_LLVM_IR_DOMTREE_HPP

#include <jlm/util/common.hpp>

#include <memory>
#include <vector>

namespace jlm {

class cfg;
class cfg_node;

class domnode final {
	typedef std::vector<std::unique_ptr<domnode>>::const_iterator const_iterator;
public:
	domnode(cfg_node * node)
	: depth_(0)
	, node_(node)
	, parent_(nullptr)
	{}

	domnode(const domnode&) = delete;

	domnode(domnode&&) = delete;

	domnode &
	operator=(const domnode&) = delete;

	domnode &
	operator=(domnode&&) = delete;

	const_iterator
	begin() const
	{
		return children_.begin();
	}

	const_iterator
	end() const
	{
		return children_.end();
	}

	domnode *
	add_child(std::unique_ptr<domnode> child);

	size_t
	nchildren() const noexcept
	{
		return children_.size();
	}

	domnode *
	child(size_t index) const noexcept
	{
		JLM_ASSERT(index < nchildren());
		return children_[index].get();
	}

	cfg_node *
	node() const noexcept
	{
		return node_;
	}

	domnode *
	parent() const noexcept
	{
		return parent_;
	}

	size_t
	depth() const noexcept
	{
		return depth_;
	}

	static std::unique_ptr<domnode>
	create(cfg_node * node)
	{
		return std::unique_ptr<domnode>(new domnode(node));
	}

private:
	size_t depth_;
	cfg_node * node_;
	domnode * parent_;
	std::vector<std::unique_ptr<domnode>> children_;
};

std::unique_ptr<domnode>
domtree(jlm::cfg & cfg);

}

#endif
