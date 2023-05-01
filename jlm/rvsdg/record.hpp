/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TYPES_RECORD_HPP
#define JLM_TYPES_RECORD_HPP

#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jive {

/* declaration */

class rcddeclaration final {
public:
	inline
	~rcddeclaration()
	{}

private:
	inline
	rcddeclaration()
	{}

	rcddeclaration(const rcddeclaration &) = delete;

	rcddeclaration &
	operator=(const rcddeclaration &) = delete;

public:
	inline size_t
	nelements() const noexcept
	{
		return types_.size();
	}

	const valuetype &
	element(size_t index) const noexcept
	{
		JIVE_DEBUG_ASSERT(index < nelements());
		return *static_cast<const valuetype*>(types_[index].get());
	}

	void
	append(const jive::valuetype & type)
	{
		types_.push_back(type.copy());
	}

	static inline std::unique_ptr<rcddeclaration>
	create()
	{
		return std::unique_ptr<rcddeclaration>(new rcddeclaration());
	}

	static inline std::unique_ptr<rcddeclaration>
	create(const std::vector<const valuetype*> & types)
	{
		auto dcl = create();
		for (const auto & type : types)
			dcl->append(*type);

		return dcl;
	}

private:
	std::vector<std::unique_ptr<jive::type>> types_;
};

void
unregister_rcddeclarations(const jive::graph * graph);

/* record type */

class rcdtype final : public jive::valuetype {
public:
	virtual
	~rcdtype() noexcept;

	inline
	rcdtype(const rcddeclaration * dcl) noexcept
	: dcl_(dcl)
	{}

	inline const rcddeclaration *
	declaration() const noexcept
	{
		return dcl_;
	}

	virtual
	std::string debug_string() const override;

	virtual bool
	operator==(const jive::type & type) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

private:
	const rcddeclaration * dcl_;
};

}

#endif
