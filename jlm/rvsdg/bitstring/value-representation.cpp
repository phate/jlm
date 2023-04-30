/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/value-representation.hpp>

#include <stdexcept>

namespace jive {

uint64_t
bitvalue_repr::to_uint() const
{
	size_t limit = std::min(nbits(), size_t(64));
	/* bits beyond 64 must be zero, else value is not representable as uint64_t */
	for (size_t n = limit; n < nbits(); ++n) {
		if (data_[n] != '0')
			throw std::range_error("Bit constant value exceeds uint64 range");
	}

	uint64_t result = 0;
	uint64_t pos_value = 1;
	for (size_t n = 0; n < limit; ++n) {
		switch (data_[n]) {
			case '0': {
				break;
			}
			case '1': {
				result |= pos_value;
				break;
			}
			default: {
				throw std::range_error("Undetermined bit constant");
			}
		}
		pos_value = pos_value << 1;
	}
	return result;
}

int64_t
bitvalue_repr::to_int() const
{
	/* all bits from 63 on must be identical, else value is not representable as int64_t */
	char sign_bit = data_[nbits()-1];
	size_t limit = std::min(nbits(), size_t(63));
	for (size_t n = limit; n < nbits(); ++n) {
		if (data_[n] != sign_bit)
			throw std::range_error("Bit constant value exceeds int64 range");
	}

	int64_t result = 0;
	uint64_t pos_value = 1;
	for (size_t n = 0; n < 64; ++n) {
		switch (n < nbits() ? data_[n] : sign_bit) {
			case '0': {
				break;
			}
			case '1': {
				result |= pos_value;
				break;
			}
			default: {
				throw std::range_error("Undetermined bit constant");
			}
		}
		pos_value = pos_value << 1;
	}
	return result;
}

}
