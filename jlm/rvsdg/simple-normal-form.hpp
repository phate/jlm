/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_SIMPLE_NORMAL_FORM_HPP
#define JLM_RVSDG_SIMPLE_NORMAL_FORM_HPP

#include <jlm/rvsdg/node-normal-form.hpp>

namespace jive {

class simple_op;

class simple_normal_form : public node_normal_form {
public:
	virtual
	~simple_normal_form() noexcept;

	simple_normal_form(
		const std::type_info & operator_class,
		jive::node_normal_form * parent,
		jive::graph * graph) noexcept;

	virtual bool
	normalize_node(jive::node * node) const override;

	virtual std::vector<jive::output*>
	normalized_create(
		jive::region * region,
		const jive::simple_op & op,
		const std::vector<jive::output*> & arguments) const;

	virtual void
	set_cse(bool enable);

	inline bool
	get_cse() const noexcept
	{
		return enable_cse_;
	}

private:
	bool enable_cse_;
};

}

#endif
