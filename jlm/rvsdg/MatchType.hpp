/*
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_MATCH_TYPE_HPP
#define JLM_RVSDG_MATCH_TYPE_HPP

#include <stdexcept>
#include <tuple>

namespace jlm::rvsdg
{

/**
 * \brief Template helper to deduce argument type of member function pointer.
 *
 * \param MFP
 *   Member function pointer to deduce first argument type of.
 */
template<typename MFP>
struct member_function_pointer_argtype;

// Partial specialization to deduce type from non-const member function pointers.
template<typename MFP, typename Ret, typename... Args>
struct member_function_pointer_argtype<Ret (MFP::*)(Args...)>
{
  // Deduce for non-const member functions.
  using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

// Partial specialization to deduce type from const member function pointers.
template<typename MFP, typename Ret, typename... Args>
struct member_function_pointer_argtype<Ret (MFP::*)(Args...) const>
{
  // Deduce for const member functions.
  using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

/**
 * \brief Template helper to deduce result type of member function pointer.
 *
 * \param MFP
 *   Member function pointer to deduce first argument from.
 */
template<typename MFP>
struct member_function_pointer_restype;

// Partial specialization to deduce type from non-const member function pointers.
template<typename MFP, typename Ret, typename... Args>
struct member_function_pointer_restype<Ret (MFP::*)(Args...)>
{
  // Deduce for non-const member functions.
  using type = Ret;
};

// Partial specialization to deduce type from const member function pointers.
template<typename MFP, typename Ret, typename... Args>
struct member_function_pointer_restype<Ret (MFP::*)(Args...) const>
{
  // Deduce for const member functions.
  using type = Ret;
};

/**
 * \brief Template helper to deduce first argument of a callable object.
 *
 * \param Callable
 *   Callable object to deduce result type of.
 */
template<typename Callable>
struct callable_argtype
{
  // Deduce for object types.
  using type = typename member_function_pointer_argtype<decltype(&Callable::operator())>::type;
};

// Partial specialization to deduce if callabale object is a function pointer type.
template<typename Ret, typename... Args>
struct callable_argtype<Ret(Args...)>
{
  // Specialization to deduce for function pointer types.
  using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

/**
 * \brief Template helper to deduce result of a callable object.
 *
 * \param Callable
 *   Callable object to deduce first parameter of.
 */
template<typename Callable>
struct callable_restype
{
  // Deduce for object types.
  using type = typename member_function_pointer_restype<decltype(&Callable::operator())>::type;
};

// Partial specialization to deduce if callable object is a function pointer type.
template<typename Ret, typename... Args>
struct callable_restype<Ret(Args...)>
{
  // Specialization to deduce for function pointer types.
  using type = Ret;
};

/**
 * \brief Pattern match over subclass type of given object.
 *
 * \param obj
 *   Object to be matched over
 *
 * \param fns
 *   Functions to be attempted, in order.
 *
 * Each of the callable function objects must take a reference to
 * a subclass of the given object. Pattern matching tries to cast
 * the object to each given subclass in turn, and calls the first
 * applicable handler function
 */
template<typename T, typename... Fns>
void
MatchType(T & obj, const Fns &... fns);

// Specialization to handle the termination (empty handlers) case.
template<typename T>
void
MatchType(T & x)
{}

// Specialization to handle the head case.
template<typename T, typename Fn, typename... Fns>
void
MatchType(T & x, const Fn & fn, const Fns &... fns)
{
  using S = std::remove_reference_t<typename callable_argtype<Fn>::type>;
  if (auto i = dynamic_cast<S *>(&x))
  {
    fn(*i);
  }
  else
  {
    MatchType(x, fns...);
  }
};

/**
 * \brief Pattern match over subclass type of given object with default handler.
 *
 * \param obj
 *   Object to be matched over
 *
 * \param fns
 *   Functions to be attempted, in order. Last callable must
 *   not take any parameters and is called as failure fallback.
 *
 * Each of the callable function objects must take a reference to
 * a subclass of the given object. Pattern matching tries to cast
 * the object to each given subclass in turn, and calls the first
 * applicable handler function. The last callable must not take
 * any parameters and will be called when neither of the previous
 * handlers matched.
 */
template<typename T, typename... Fns>
void
MatchTypeWithDefault(T & obj, const Fns &... fns);

// Specialization to handle the termination (empty handlers) case.
template<typename T, typename Fn>
typename callable_restype<Fn>::type
MatchTypeWithDefault(T & /* x */, const Fn & fn)
{
  return fn();
}

// Specialization to handle the head case.
template<typename T, typename Fn, typename... Fns>
typename callable_restype<Fn>::type
MatchTypeWithDefault(T & x, const Fn & fn, const Fns &... fns)
{
  using S = std::remove_reference_t<typename callable_argtype<Fn>::type>;
  if (auto i = dynamic_cast<S *>(&x))
  {
    return fn(*i);
  }
  else
  {
    return MatchTypeWithDefault(x, fns...);
  }
};

/**
 * \brief Pattern match over subclass type of given object.
 *
 * \param obj
 *   Object to be matched over
 *
 * \param fns
 *   Functions to be attempted, in order.
 *
 * Each of the callable function objects must take a reference to
 * a subclass of the given object. Pattern matching tries to cast
 * the object to each given subclass in turn, and calls the first
 * applicable handler function. If no handler function applies,
 * this is considered a logic error (it should never occur).
 */
template<typename T, typename... Fns>
void
MatchTypeOrFail(T & obj, const Fns &... fns);

// Specialization to handle the termination (last handler) case.
template<typename T, typename Fn>
typename callable_restype<Fn>::type
MatchTypeOrFail(T & x, const Fn & fn)
{
  using S = std::remove_reference_t<typename callable_argtype<Fn>::type>;
  if (auto i = dynamic_cast<S *>(&x))
  {
    return fn(*i);
  }
  else
  {
    throw std::logic_error(std::string("Incomplete pattern matching on ") + typeid(T).name());
  }
}

// Specialization to handle the head case.
template<typename T, typename Fn, typename... Fns>
typename callable_restype<Fn>::type
MatchTypeOrFail(T & x, const Fn & fn, const Fns &... fns)
{
  using S = std::remove_reference_t<typename callable_argtype<Fn>::type>;
  if (auto i = dynamic_cast<S *>(&x))
  {
    return fn(*i);
  }
  else
  {
    return MatchTypeOrFail(x, fns...);
  }
};

}

#endif
