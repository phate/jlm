/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ANNOTATION_HPP
#define JLM_IR_ANNOTATION_HPP

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

class aggnode;
class variable;

class VariableSet final {
	using constiterator = std::unordered_set<const variable*>::const_iterator;

public:
	constiterator
	begin() const
	{
		return set_.begin();
	}

	constiterator
	end() const
	{
		return set_.end();
	}

	bool
	contains(const variable * v) const
	{
		return set_.find(v) != set_.end();
	}

	size_t
	size() const noexcept
	{
		return set_.size();
	}

	void
	insert(const variable * v)
	{
		set_.insert(v);
	}

	void
	insert(const VariableSet & vs)
	{
		set_.insert(vs.set_.begin(), vs.set_.end());
	}

	void
	remove(const variable * v)
	{
		set_.erase(v);
	}

	void
	remove(const VariableSet & vs)
	{
		for (const auto & v : vs)
			remove(v);
	}

	void
	intersect(const VariableSet & vs)
	{
		std::unordered_set<const variable*> intersect;
		for (const auto & v : vs) {
			if (contains(v))
				intersect.insert(v);
		}

		set_ = intersect;
	}

	void
	subtract(const VariableSet & vs)
	{
		for (auto & v : vs)
			remove(v);
	}

	bool
	operator==(const VariableSet & other) const
	{
		if (size() != other.size())
			return false;

		for (const auto & v : other) {
			if (!contains(v))
				return false;
		}

		return true;
	}

	bool
	operator!=(const VariableSet & other) const
	{
		return !(*this == other);
	}

private:
	std::unordered_set<const variable*> set_;
};

class demandset {
public:
	virtual
	~demandset();

	inline
	demandset()
	{}

	static inline std::unique_ptr<demandset>
	create()
	{
		return std::make_unique<demandset>();
	}

	VariableSet top;
	VariableSet bottom;

	VariableSet reads;
	VariableSet allwrites;
	VariableSet fullwrites;
};

typedef std::unordered_map<const aggnode*, std::unique_ptr<demandset>> DemandMap;

DemandMap
Annotate(const jlm::aggnode & root);

}

#endif
