/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_MATCH_VARIANT_HPP
#define JLM_RVSDG_MATCH_VARIANT_HPP

#include <variant>

namespace jlm::rvsdg
{

namespace
{
// The usual C++ "overloaded" pattern & deduction pattern.
template<class... Ts>
struct Overloaded : Ts...
{
  using Ts::operator()...;
};
template<class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;
}

/**
 * \brief Pattern match over variant
 *
 * \param obj
 *   Object to be matched over
 *
 * \param fns
 *   Functions to be attempted, in order.
 *
 * \pre
 *   \p obj must be a variant, and the given
 *   functions must be a complete set of handlers
 *   for every member of the variant.
 *
 * Determines the active alternative of the variant and calls the
 * user-specified handler function for it.
 *
 * All handler functions must produce the same return values,
 * and its result is passed back to the caller.
 */
template<typename T, typename... Fns>
decltype(auto)
MatchVariant(T && obj, Fns &&... fns)
{
  return std::visit(Overloaded{ std::forward<Fns>(fns)... }, std::forward<T>(obj));
}

}

#endif
