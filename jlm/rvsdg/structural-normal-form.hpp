/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_STRUCTURAL_NORMAL_FORM_HPP
#define JLM_RVSDG_STRUCTURAL_NORMAL_FORM_HPP

#include <jlm/rvsdg/node-normal-form.hpp>

namespace jive {

class structural_normal_form : public node_normal_form {
public:
	virtual
	~structural_normal_form() noexcept;

	structural_normal_form(
		const std::type_info & operator_class,
		jive::node_normal_form * parent,
		jive::graph * graph) noexcept;
};

}

#endif
