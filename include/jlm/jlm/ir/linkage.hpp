/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM_IR_LINKAGE_HPP
#define JLM_JLM_IR_LINKAGE_HPP

namespace jlm {

enum class linkage {
	  external_linkage
	, available_externally_linkage
	, link_once_any_linkage
	, link_once_odr_linkage
	, weak_any_linkage
	, weak_odr_linkage
	, appending_linkage
	, internal_linkage
	, private_linkage
	, external_weak_linkage
	, common_linkage
};

}

#endif
