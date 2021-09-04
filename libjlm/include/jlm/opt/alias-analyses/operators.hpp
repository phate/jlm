/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_OPERATORS_HPP
#define JLM_OPT_ALIAS_ANALYSES_OPERATORS_HPP

#include <jlm/ir/operators.hpp>

namespace jlm {
namespace aa {

/** \brief FIXME: write documentation
*/
class lambda_aamux_op final : public MemStateOperator {
public:
	~lambda_aamux_op() override;

private:
	lambda_aamux_op(
		size_t noperands,
		size_t nresults,
		bool is_entry,
		const std::vector<std::string> & dbgstrs)
	: MemStateOperator(noperands, nresults)
	, is_entry_(is_entry)
	, dbgstrs_(dbgstrs)
	{}

public:
	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	/** \brief FIXME: write documentation
	*/
	static std::vector<jive::output*>
	create_entry(
		jive::output * output,
		size_t nresults,
		const std::vector<std::string> & dbgstrs)
	{
		if (nresults != dbgstrs.size())
			throw error("Insufficient number of state debug strings.");

		auto region = output->region();
		lambda_aamux_op op(1, nresults, true, dbgstrs);
		return jive::simple_node::create_normalized(region, op, {output});
	}

	/** \brief FIXME: write documentation
	*/
	static jive::output *
	create_exit(
		jive::region * region,
		const std::vector<jive::output*> & operands,
		const std::vector<std::string> & dbgstrs)
	{
		if (operands.size() != dbgstrs.size())
			throw error("Insufficient number of state debug strings.");

		lambda_aamux_op op(operands.size(), 1, false, dbgstrs);
		return jive::simple_node::create_normalized(region, op, operands)[0];
	}

private:
	bool is_entry_;
	std::vector<std::string> dbgstrs_;
};

/** \brief FIXME: write documentation
*/
class call_aamux_op final : public MemStateOperator {
public:
	~call_aamux_op() override;

private:
	call_aamux_op(
		size_t noperands,
		size_t nresults,
		bool is_entry,
		const std::vector<std::string> & dbgstrs)
	: MemStateOperator(noperands, nresults)
	, is_entry_(is_entry)
	, dbgstrs_(dbgstrs)
	{}

public:
	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static jive::output *
	create_entry(
		jive::region * region,
		const std::vector<jive::output*> & operands,
		const std::vector<std::string> & dbgstrs)
	{
		call_aamux_op op(operands.size(), 1, true, dbgstrs);
		return jive::simple_node::create_normalized(region, op, operands)[0];
	}

	static std::vector<jive::output*>
	create_exit(
		jive::output * output,
		size_t nresults,
		const std::vector<std::string> & dbgstrs)
	{
		auto region = output->region();
		call_aamux_op op(1, nresults, false, dbgstrs);
		return jive::simple_node::create_normalized(region, op, {output});
	}

private:
	bool is_entry_;
	std::vector<std::string> dbgstrs_;
};

}
}

#endif
