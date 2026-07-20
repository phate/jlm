/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_UNARY_HPP
#define JLM_RVSDG_UNARY_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>

#include <optional>

namespace jlm::rvsdg
{

/**
  \brief Unary operator

  Operator taking a single argument.
*/
class UnaryOperation : public SimpleOperation
{
public:
  ~UnaryOperation() noexcept override;

  UnaryOperation(std::shared_ptr<const Type> operand, std::shared_ptr<const Type> result)
      : SimpleOperation({ std::move(operand) }, { std::move(result) })
  {}
};

}

#endif
