/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NULLARY_HPP
#define JLM_RVSDG_NULLARY_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::rvsdg
{

class Output;

/**
  \brief Nullary operator (operator taking no formal arguments)
*/
class NullaryOperation : public SimpleOperation
{
public:
  ~NullaryOperation() noexcept override;

  explicit NullaryOperation(std::shared_ptr<const Type> resultType)
      : SimpleOperation({}, { std::move(resultType) })
  {}
};

}

#endif
