/*
 * Copyright 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_RVSDG_SECTION_HPP
#define JIVE_RVSDG_SECTION_HPP

/**
	\brief Enumerate standard sections
	
	Enumerate the standard sections that are generally available on
	all targets.
*/
typedef enum jive_stdsectionid {
	jive_stdsectionid_invalid = -1,
	jive_stdsectionid_external = 0,
	jive_stdsectionid_code = 1,
	jive_stdsectionid_data = 2,
	jive_stdsectionid_rodata = 3,
	jive_stdsectionid_bss = 4
} jive_stdsectionid;

#endif
