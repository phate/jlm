/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NODE_NORMAL_FORM_HPP
#define JLM_RVSDG_NODE_NORMAL_FORM_HPP

#include <stddef.h>

#include <typeindex>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include <jlm/util/common.hpp>
#include <jlm/util/intrusive-hash.hpp>

/* normal forms */

namespace jive {

class graph;
class node;
class operation;
class output;
class region;

class node_normal_form {
public:
	virtual
	~node_normal_form() noexcept;

	inline
	node_normal_form(
		const std::type_info & operator_class,
		jive::node_normal_form * parent,
		jive::graph * graph) noexcept
		: operator_class_(operator_class)
		, parent_(parent)
		, graph_(graph)
		, enable_mutable_(true)
	{
		if (parent) {
			enable_mutable_ = parent->enable_mutable_;
			parent->subclasses_.insert(this);
		}
	}

	virtual bool
	normalize_node(jive::node * node) const;

	inline node_normal_form *
	parent() const noexcept { return parent_; }
	inline jive::graph *
	graph() const noexcept { return graph_; }

	virtual void
	set_mutable(bool enable);
	inline bool
	get_mutable() const noexcept { return enable_mutable_; }

	static void
	register_factory(
		const std::type_info & type,
		jive::node_normal_form *(*fn)(
			const std::type_info & operator_class,
			jive::node_normal_form * parent,
			jive::graph * graph));

	static node_normal_form *
	create(
		const std::type_info & operator_class,
		jive::node_normal_form * parent,
		jive::graph * graph);

	class opclass_hash_accessor {
	public:
		std::type_index
		get_key(const node_normal_form * obj) const noexcept
		{
			return std::type_index(obj->operator_class_);
		}
		node_normal_form *
		get_prev(const node_normal_form * obj) const noexcept
		{
			return obj->new_hash_chain.prev;
		}
		void
		set_prev(node_normal_form * obj, node_normal_form * prev) const noexcept
		{
			obj->new_hash_chain.prev = prev;
		}
		node_normal_form *
		get_next(const node_normal_form * obj) const noexcept
		{
			return obj->new_hash_chain.next;
		}
		void
		set_next(node_normal_form * obj, node_normal_form * next) const noexcept
		{
			obj->new_hash_chain.next = next;
		}
	};

protected:
	template<typename X, void(X::*fn)(bool)>
	void
	children_set(bool enable)
	{
		for (const auto & subclass : subclasses_) {
			if (auto nf = dynamic_cast<X*>(subclass))
				(nf->*fn)(enable);
		}
	}

private:
	const std::type_info & operator_class_;
	node_normal_form * parent_;
	jive::graph * graph_;

	struct {
		node_normal_form * prev;
		node_normal_form * next;
	} new_hash_chain;

	bool enable_mutable_;
	std::unordered_set<node_normal_form*> subclasses_;
};

typedef jive::detail::owner_intrusive_hash<
	std::type_index,
	jive::node_normal_form,
	jive::node_normal_form::opclass_hash_accessor> node_normal_form_hash;

}

#endif
