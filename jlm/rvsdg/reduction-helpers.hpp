/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_REDUCTION_HELPERS_HPP
#define JLM_RVSDG_REDUCTION_HELPERS_HPP

#include <jlm/rvsdg/node.hpp>

#include <algorithm>

namespace jlm::rvsdg
{
namespace base
{
namespace detail
{

/* Test whether any pair of adjacent elements of "args" can be reduced according
 * to "reduction_tester". */
template<typename Container, typename ReductionTester>
bool
pairwise_test_reduce(const Container & args, const ReductionTester & reduction_tester) noexcept
{
  if (args.empty())
  {
    return false;
  }

  auto left = args.begin();
  auto right = std::next(left);

  while (right != args.end())
  {
    if (reduction_tester(*left, *right))
    {
      return true;
    }
    left = right;
    ++right;
  }

  return false;
}

/* Apply "reductor" to each pair of adjacent elements of "args", replace the two
 * with the result if not nullptr. */
template<typename Container, typename Reductor>
Container
pairwise_reduce(Container args, const Reductor & reductor)
{
  if (args.empty())
  {
    return args;
  }

  auto left = args.begin();
  auto right = std::next(left);

  while (right != args.end())
  {
    auto res = reductor(*left, *right);
    if (res)
    {
      *left = res;
      ++right;
    }
    else
    {
      ++left;
      *left = *right;
      ++right;
    }
  }
  args.erase(std::next(left), args.end());

  return args;
}

/* Test whether any pair of elements of "args" can be reduced according
 * to "reduction_tester". */
template<typename Container, typename ReductionTester>
bool
commutative_pairwise_test_reduce(
    const Container & args,
    const ReductionTester & reduction_tester) noexcept
{
  auto left = args.begin();
  while (left != args.end())
  {
    auto right = std::next(left);
    while (right != args.end())
    {
      if (reduction_tester(*left, *right))
      {
        return true;
      }
      else
      {
        ++right;
      }
    }
    ++left;
  }

  return false;
}

/* Apply "reductor" to each pair of elements of "args", replace the two
 * with the result if not nullptr. */
template<typename Container, typename Reductor>
Container
commutative_pairwise_reduce(Container args, const Reductor & reductor)
{
  auto left = args.begin();
  while (left != args.end())
  {
    auto right = std::next(left);
    while (right != args.end())
    {
      auto result = reductor(*left, *right);
      if (result)
      {
        *left = result;
        *right = args.back();
        args.pop_back();
        /* Start over and compare with everything */
        left = args.begin();
        right = std::next(left);
      }
      else
      {
        ++right;
      }
    }
    ++left;
  }

  return args;
}

/* Test whether "flatten_tester" applies to any element of "args". */
template<typename Container, typename FlattenTester>
bool
associative_test_flatten(const Container & args, const FlattenTester & flatten_tester)
{
  return std::any_of(args.begin(), args.end(), flatten_tester);
}

/* Replace each argument of "args" with the arguments of its defining node
 * for each where "flatten_tester" returns true. */
template<typename FlattenTester>
std::vector<jlm::rvsdg::Output *>
associative_flatten(std::vector<jlm::rvsdg::Output *> args, const FlattenTester & flatten_tester)
{
  size_t n = 0;
  while (n < args.size())
  {
    if (flatten_tester(args[n]))
    {
      auto arg = args[n];
      JLM_ASSERT(is<NodeOutput>(arg));
      auto sub_args = jlm::rvsdg::operands(TryGetOwnerNode<Node>(*arg));
      args[n] = sub_args[0];
      args.insert(args.begin() + n + 1, sub_args.begin() + 1, sub_args.end());
    }
    else
    {
      ++n;
    }
  }
  return args;
}

}
}
}

#endif
