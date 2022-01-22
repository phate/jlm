/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_RVSDG_RESOURCE_HPP
#define JIVE_RVSDG_RESOURCE_HPP

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <jive/common.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jive {

class resource;
class resource_class;
class type;

/* resource_type */

class resource_type {
public:
	virtual
	~resource_type();

	inline
	resource_type(
		bool is_abstract,
		const std::string & name,
		const resource_type * parent)
	: name_(name)
	, is_abstract_(is_abstract)
	, parent_(parent)
	{}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline const resource_type *
	parent() const noexcept
	{
		return parent_;
	}

	inline bool
	is_abstract() const noexcept
	{
		return is_abstract_;
	}

private:
	std::string name_;
	bool is_abstract_;
	const resource_type * parent_;
};

extern const jive::resource_type root_resource;

/* resource_class_demotion */

class resource_class_demotion final {
public:
	inline
	resource_class_demotion(
		const resource_class * target,
		const std::vector<const resource_class*> & path)
	: target_(target)
	, path_(path)
	{}

	inline const resource_class *
	target() const noexcept
	{
		return target_;
	}

	inline const std::vector<const resource_class*>
	path() const noexcept
	{
		return path_;
	}

private:
	const resource_class * target_;
	std::vector<const resource_class*> path_;
};

class resource_class {
public:
	enum class priority {
		  invalid = 0
		, control = 1
		, reg_implicit = 2
		, mem_unique = 3
		, reg_high = 4
		, reg_low = 5
		, mem_generic = 6
		, lowest = 7
	};

	virtual
	~resource_class();

	inline
	resource_class(
		const jive::resource_type * restype,
		const std::string & name,
		const std::unordered_set<const jive::resource*> resources,
		const jive::resource_class * parent,
		priority pr,
		const std::vector<resource_class_demotion> & demotions,
		const jive::type * type)
	: priority(pr)
	, restype_(restype)
	, depth_(parent ? parent->depth()+1 : 0)
	, name_(name)
	, type_(type)
	, parent_(parent)
	, resources_(resources)
	, demotions_(demotions)
	{}

	inline size_t
	depth() const noexcept
	{
		return depth_;
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline const jive::type &
	type() const noexcept
	{
		JIVE_DEBUG_ASSERT(type_ != nullptr);
		return *type_;
	}

	inline const jive::resource_class *
	parent() const noexcept
	{
		return parent_;
	}

	inline size_t
	nresources() const noexcept
	{
		return resources_.size();
	}

	inline const std::unordered_set<const jive::resource*> &
	resources() const noexcept
	{
		return resources_;
	}

	inline const std::vector<resource_class_demotion> &
	demotions() const noexcept
	{
		return demotions_;
	}

	inline const jive::resource_type *
	resource_type() const noexcept
	{
		return restype_;
	}

	inline bool
	is_resource(const jive::resource_type * restype) const noexcept
	{
		auto tmp = resource_type();
		while (restype) {
			if (tmp == restype)
				return true;
			tmp = tmp->parent();
		}

		return false;
	}

	/** \brief Priority for register allocator */
	resource_class::priority priority;
	
private:
	const jive::resource_type * restype_;

	/** \brief Number of steps from root resource class */
	size_t depth_;
	std::string name_;

	/** \brief Port type corresponding to this resource */
	const jive::type * type_;

	/** \brief Parent resource class */
	const jive::resource_class * parent_;

	/** \brief Available resources */
	std::unordered_set<const jive::resource*> resources_;

	/** \brief Paths for "demoting" this resource to a different one */
	std::vector<resource_class_demotion> demotions_;
};

static inline const jive::resource_class *
find_union(
	const jive::resource_class * rescls1,
	const jive::resource_class * rescls2) noexcept
{
	while (true) {
		if (rescls1 == rescls2)
			return rescls1;

		if (rescls1->depth() > rescls2->depth())
			rescls1 = rescls1->parent();
		else
			rescls2 = rescls2->parent();
	}

	JIVE_ASSERT(0);
}

static inline const jive::resource_class *
find_intersection(
	const jive::resource_class * rescls1,
	const jive::resource_class * rescls2) noexcept
{
	auto u = find_union(rescls1, rescls2);
	if (u == rescls1)
		return rescls2;

	if (u == rescls2)
		return rescls1;

	return nullptr;
}

static inline const jive::resource_class *
relax(const jive::resource_class * rescls)
{
	/*
		FIXME: hopefully this function is transitionary --
		currently everything that is needed is the
		class directly below the root
	*/
	while (rescls->parent() && !rescls->parent()->resource_type()->is_abstract())
		rescls = rescls->parent();

	return rescls;
}

}

extern const jive::resource_class jive_root_resource_class;

namespace jive {

class resource {
public:
	virtual
	~resource();

	inline
	resource(const std::string & name, const jive::resource_class * rescls)
	: resource_class(rescls)
	, name_(name)
	{}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	const jive::resource_class * resource_class;
private:
	std::string name_;
};

}

#endif
