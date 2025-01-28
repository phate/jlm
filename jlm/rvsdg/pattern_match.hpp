/*
 * Copyright 2015 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_PATTERN_MATCH_HPP
#define JLM_RVSDG_PATTERN_MATCH_HPP

#include <tuple>

namespace jlm::rvsdg
{

// Template helper to deduce the first argument type of a member
// function pointer.
template<typename func>
struct method_argtype;

template<typename T, typename Ret, typename... Args>
struct method_argtype<Ret (T::*)(Args...)> {
	// Deduce for non-const member functions.
	using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

template<typename T, typename Ret, typename... Args>
struct method_argtype<Ret (T::*)(Args...) const> {
	// Deduce for const member functions.
	using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

// Template helper to deduce the first argument type of a callable
// object (function (pointer) type of member function (pointer) type).
template<typename T>
struct callable_argtype {
	// Deduce for object types.
	using type = typename method_argtype<decltype(&T::operator())>::type;
};

template<typename Ret, typename... Args>
struct callable_argtype<Ret (Args...)> {
	// Specialization to deduce for function pointer types.
	using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

template<typename T, typename... Fns>
void pattern_match(const T&, const Fns&...);

template<typename T>
void pattern_match(const T& x) {}

template<typename T, typename Fn, typename... Fns>
void pattern_match(const T& x, const Fn& fn, const Fns&... fns) {
	using S = typename std::remove_reference<typename callable_argtype<Fn>::type>::type;
	if (auto i = dynamic_cast<const S*>(&x)) {
		fn(*i);
	} else {
		pattern_match(x, fns...);
	}
};

}

#endif
