/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <unordered_map>

#include <jive/rvsdg/equivalence.hpp>

class jive_node_equiv_entry {
public:
	const jive::node * first;
	const jive::node * second;
	bool pending;
	struct {
		jive_node_equiv_entry * prev;
		jive_node_equiv_entry * next;
	} hash_chain;

private:
	jive::detail::intrusive_list_anchor<
		jive_node_equiv_entry
	> pending_anchor;

public:
	typedef jive::detail::intrusive_list_accessor<
		jive_node_equiv_entry,
		&jive_node_equiv_entry::pending_anchor
	> pending_accessor;
};

typedef jive::detail::intrusive_list<
	jive_node_equiv_entry,
	jive_node_equiv_entry::pending_accessor
> pending_list;

class jive_equiv_state {
public:
	pending_list pending;
	std::unordered_map<const jive::node *, jive_node_equiv_entry*> node_mapping;
};

static void
jive_equiv_state_fini(jive_equiv_state * self)
{
	while (self->node_mapping.size()) {
		jive_node_equiv_entry * entry = self->node_mapping.begin()->second;
		self->node_mapping.erase(self->node_mapping.begin());
		delete entry;
	}
}

static jive_node_equiv_entry *
jive_equiv_state_lookup(jive_equiv_state * self, const jive::node * node)
{
	jive_node_equiv_entry * entry;
	auto i = self->node_mapping.find(node);
	if (i == self->node_mapping.end()) {
		entry = new jive_node_equiv_entry;
		entry->first = node;
		entry->second = 0;
		entry->pending = true;
		self->node_mapping[node] = entry;
		self->pending.push_back(entry);
	} else
		entry = i->second;

	return entry;
}

static void
jive_equiv_state_mark_verified(jive_equiv_state * self, jive_node_equiv_entry * entry)
{
	JIVE_DEBUG_ASSERT(entry->pending);
	entry->pending = false;
	self->pending.erase(entry);
}

static bool
jive_equiv_state_check_node(jive_equiv_state * self, const jive::node * n1, const jive::node * n2)
{
	if (n1->noutputs() != n2->noutputs()) {
		return false;
	}
	if (n1->ninputs() != n2->ninputs()) {
		return false;
	}
	if (n1->ninputs() != n2->ninputs()) {
		return false;
	}
	if (n1->operation() != n2->operation()) {
		return false;
	}

	size_t n = 0;
	for (n = 0; n < n1->ninputs(); ++n) {
		auto o1 = jive::node_output::node(n1->input(n)->origin());
		auto o2 = jive::node_output::node(n2->input(n)->origin());

		auto entry = jive_equiv_state_lookup(self, o1);
		if (entry->second && entry->second != o2)
			return false;
		entry->second = o2;
	}
	
	return true;
}

bool
jive_graphs_equivalent(
	jive::graph * graph1, jive::graph * graph2,
	size_t ncheck, jive::node * const check1[], jive::node * const check2[],
	size_t nassumed, jive::node * const ass1[], jive::node * const ass2[])
{
	jive_equiv_state state;

	bool satisfied = true;
	size_t n;
	
	for (n = 0; n < nassumed; ++n) {
		jive_node_equiv_entry * entry = jive_equiv_state_lookup(&state, ass1[n]);
		if (entry->second != 0 && entry->second != ass2[n]) {
			satisfied = false;
			break;
		}
		entry->second = ass2[n];
		jive_equiv_state_mark_verified(&state, entry);
	}
	
	for (n = 0; n < ncheck; ++n) {
		jive_node_equiv_entry * entry = jive_equiv_state_lookup(&state, check1[n]);
		if (entry->second != 0 && entry->second != check2[n]) {
			satisfied = false;
			break;
		}
		entry->second = check2[n];
	}
	
	while (satisfied && state.pending.first()) {
		auto entry = state.pending.first();
		satisfied = jive_equiv_state_check_node(&state, entry->first, entry->second);
		jive_equiv_state_mark_verified(&state, entry);
	}
	
	jive_equiv_state_fini(&state);
	
	return satisfied;
}
