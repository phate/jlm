/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jlm::rvsdg
{

// explicit instantiation
template class DomainConstOperation<
    BitType,
    BitValueRepresentation,
    BitValueRepresentationFormatValue,
    type_of_value>;

}
