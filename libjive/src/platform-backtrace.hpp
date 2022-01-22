/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_PLATFORM_BACKTRACE
#define JIVE_PLATFORM_BACKTRACE

typedef struct jive_platform_backtrace jive_platform_backtrace;

#ifdef __linux__

#include <execinfo.hpp>

struct jive_platform_backtrace {
	void * chain[64];
	int size;
};

static inline void
jive_platform_backtrace_collect(jive_platform_backtrace * self)
{
	self->size = backtrace(self->chain, sizeof(self->chain) / sizeof(self->chain[0]));
}

static inline void
jive_platform_backtrace_print(jive_platform_backtrace * self)
{
	backtrace_symbols_fd(self->chain, self->size, 2);
}

#else

struct jive_platform_backtrace {
};

static inline void
jive_platform_backtrace_collect(jive_platform_backtrace * self)
{
}

static inline void
jive_platform_backtrace_print(jive_platform_backtrace * self)
{
}

#endif

#endif
