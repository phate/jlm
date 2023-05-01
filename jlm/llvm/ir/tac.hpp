/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_TAC_HPP
#define JLM_LLVM_IR_TAC_HPP

#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>

#include <list>
#include <memory>
#include <vector>

namespace jive {

namespace base {
	class type;
}
}

namespace jlm {

class tac;

/* tacvariable */

class tacvariable final : public variable {
public:
	virtual
	~tacvariable();

	tacvariable(
		jlm::tac * tac,
		const jive::type & type,
		const std::string & name)
	: variable(type, name)
	, tac_(tac)
	{}

	inline jlm::tac *
	tac() const noexcept
	{
		return tac_;
	}

	static std::unique_ptr<tacvariable>
	create(
		jlm::tac * tac,
		const jive::type & type,
		const std::string & name)
	{
		return std::make_unique<tacvariable>(tac, type, name);
	}

private:
	jlm::tac * tac_;
};

/* tac */

class tac final {
public:
	inline
	~tac() noexcept
	{}

	tac(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands);

	tac(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands,
		const std::vector<std::string> & names);

	tac(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands,
		std::vector<std::unique_ptr<tacvariable>> results);

	tac(const jlm::tac &) = delete;

	tac(jlm::tac &&) = delete;

	tac &
	operator=(const jlm::tac &) = delete;

	tac &
	operator=(jlm::tac &&) = delete;

	inline const jive::simple_op &
	operation() const noexcept
	{
		return *static_cast<const jive::simple_op*>(operation_.get());
	}

	inline size_t
	noperands() const noexcept
	{
		return operands_.size();
	}

	inline const variable *
	operand(size_t index) const noexcept
	{
		JLM_ASSERT(index < operands_.size());
		return operands_[index];
	}

	inline size_t
	nresults() const noexcept
	{
		return results_.size();
	}

	const tacvariable *
	result(size_t index) const noexcept
	{
		JLM_ASSERT(index < results_.size());
		return results_[index].get();
	}

	/*
		FIXME: I am really not happy with this function exposing
		the results, but we need these results for the SSA destruction.
	*/
	std::vector<std::unique_ptr<tacvariable>>
	results()
	{
		return std::move(results_);
	}

	void
	replace(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands);

	void
	convert(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands);

	static std::unique_ptr<jlm::tac>
	create(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands)
	{
		return std::make_unique<jlm::tac>(operation, operands);
	}

	static std::unique_ptr<jlm::tac>
	create(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands,
		const std::vector<std::string> & names)
	{
		return std::make_unique<jlm::tac>(operation, operands, names);
	}

	static std::unique_ptr<jlm::tac>
	create(
		const jive::simple_op & operation,
		const std::vector<const variable*> & operands,
		std::vector<std::unique_ptr<tacvariable>> results)
	{
		return std::make_unique<jlm::tac>(operation, operands, std::move(results));
	}

private:
	void
	create_results(
		const jive::simple_op & operation,
		const std::vector<std::string> & names)
	{
		JLM_ASSERT(names.size() == operation.nresults());

		for (size_t n = 0; n < operation.nresults(); n++) {
			auto & type = operation.result(n).type();
			results_.push_back(tacvariable::create(this, type, names[n]));
		}
	}

	static std::vector<std::string>
	create_names(size_t nnames)
	{
		static size_t c = 0;
		std::vector<std::string> names;
		for (size_t n = 0; n < nnames; n++)
			names.push_back(strfmt("tv", c++));

		return names;
	}

	std::vector<const variable*> operands_;
	std::unique_ptr<jive::operation> operation_;
	std::vector<std::unique_ptr<tacvariable>> results_;
};

template <class T> static inline bool
is(const jlm::tac * tac)
{
	return tac && is<T>(tac->operation());
}

/* FIXME: Replace all occurences of tacsvector_t with taclist
	and then remove tacsvector_t.
*/
typedef std::vector<std::unique_ptr<jlm::tac>> tacsvector_t;

/* taclist */

class taclist final {
public:
	typedef std::list<tac*>::const_iterator const_iterator;
	typedef std::list<tac*>::const_reverse_iterator const_reverse_iterator;

	~taclist();

	inline
	taclist()
	{}

	taclist(const taclist&) = delete;

	taclist(taclist && other)
	: tacs_(std::move(other.tacs_))
	{}

	taclist &
	operator=(const taclist &) = delete;

	taclist &
	operator=(taclist && other)
	{
		if (this == &other)
			return *this;

		for (const auto & tac : tacs_)
			delete tac;

		tacs_.clear();
		tacs_ = std::move(other.tacs_);

		return *this;
	}

	inline const_iterator
	begin() const noexcept
	{
		return tacs_.begin();
	}

	inline const_reverse_iterator
	rbegin() const noexcept
	{
		return tacs_.rbegin();
	}

	inline const_iterator
	end() const noexcept
	{
		return tacs_.end();
	}

	inline const_reverse_iterator
	rend() const noexcept
	{
		return tacs_.rend();
	}

	inline tac *
	insert_before(const const_iterator & it, std::unique_ptr<jlm::tac> tac)
	{
		return *tacs_.insert(it, tac.release());
	}

	inline void
	insert_before(const const_iterator & it, taclist & tl)
	{
		tacs_.insert(it, tl.begin(), tl.end());
	}

	inline void
	append_last(std::unique_ptr<jlm::tac> tac)
	{
		tacs_.push_back(tac.release());
	}

	inline void
	append_first(std::unique_ptr<jlm::tac> tac)
	{
		tacs_.push_front(tac.release());
	}

	inline void
	append_first(taclist & tl)
	{
		tacs_.insert(tacs_.begin(), tl.begin(), tl.end());
		tl.tacs_.clear();
	}

	inline size_t
	ntacs() const noexcept
	{
		return tacs_.size();
	}

	inline tac *
	first() const noexcept
	{
		return ntacs() != 0 ? tacs_.front() : nullptr;
	}

	inline tac *
	last() const noexcept
	{
		return ntacs() != 0 ? tacs_.back() : nullptr;
	}

	std::unique_ptr<tac>
	pop_first() noexcept
	{
		std::unique_ptr<tac> tac(tacs_.front());
		tacs_.pop_front();
		return tac;
	}

	std::unique_ptr<tac>
	pop_last() noexcept
	{
		std::unique_ptr<tac> tac(tacs_.back());
		tacs_.pop_back();
		return tac;
	}

	inline void
	drop_first()
	{
		delete tacs_.front();
		tacs_.pop_front();
	}

	inline void
	drop_last()
	{
		delete tacs_.back();
		tacs_.pop_back();
	}

private:
	std::list<tac*> tacs_;
};

}

#endif
