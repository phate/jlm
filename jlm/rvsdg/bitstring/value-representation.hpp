/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_VALUE_REPRESENTATION_HPP
#define JLM_RVSDG_BITSTRING_VALUE_REPRESENTATION_HPP

#include <jlm/util/common.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace jive {

/**
 Value representation used for compile-time evaluation of bitstring
 expressions. A bit is either:
	- '0' : zero
	- '1' : one
	- 'D' : defined, but unknown
	- 'X' : undefined and unknown
*/

class bitvalue_repr {
public:
	inline
	bitvalue_repr(size_t nbits, int64_t value)
	{
		if (nbits == 0)
			throw compiler_error("Number of bits is zero.");

		if (nbits < 64 && (value >> nbits) != 0 && (value >> nbits != -1))
			throw compiler_error("Value cannot be represented with the given number of bits.");

		for (size_t n = 0; n < nbits; ++n) {
			data_.push_back('0' + (value & 1));
			value = value >> 1;
		}
	}

	inline
	bitvalue_repr(const char * s)
	{
		if (strlen(s) == 0)
			throw compiler_error("Number of bits is zero.");

		for (size_t n = 0; n < strlen(s); n++) {
			if (s[n] != '0' && s[n] != '1' && s[n] != 'X' && s[n] != 'D')
				throw compiler_error("Not a valid bit.");
			data_.push_back(s[n]);
		}
	}

	bitvalue_repr(const bitvalue_repr & other)
	: data_(other.data_)
	{}

	bitvalue_repr(bitvalue_repr && other)
	: data_(std::move(other.data_))
	{}

	inline static bitvalue_repr
	repeat(size_t nbits, char bit)
	{
		return bitvalue_repr(std::string(nbits, bit).c_str());
	}

private:
	inline char
	lor(char a, char b) const noexcept
	{
		switch (a) {
			case '0':
				return b;
			case '1':
				return '1';
			case 'X':
				if (b == '1')
					return '1';
				return 'X';
			case 'D':
				if (b == '1')
					return '1';
				if (b == 'X')
					return 'X';
				return 'D';
			default:
				return 'X';
		}
	}

	inline char
	lxor(char a, char b) const noexcept
	{
		switch (a) {
			case '0':
				return b;
			case '1':
				if (b == '1')
					return '0';
				if (b == '0')
					return '1';
				return b;
			case 'X':
				return 'X';
			case 'D':
				if (b == 'X')
					return 'X';
				return a;
			default:
				return 'X';
		}
	}

	inline char
	lnot(char a) const noexcept
	{
		return lxor('1', a);
	}

	inline char
	land(char a, char b) const noexcept
	{
		switch (a) {
			case '0':
				return '0';
			case '1':
				return b;
			case 'X':
				if (b == '0')
					return '0';
				return 'X';
			case 'D':
				if (b == '0')
					return '0';
				if (b == 'X')
					return 'X';
				return 'D';
			default:
				return 'X';
		}
	}

	inline char
	carry(char a, char b, char c) const noexcept
	{
		return lor(lor(land(a,b), land(a,c)), land(b,c));
	}

	inline char
	add(char a, char b, char c) const noexcept
	{
		return lxor(lxor(a,b), c);
	}

	inline void
	udiv(
		const bitvalue_repr & divisor,
		bitvalue_repr & quotient,
		bitvalue_repr & remainder) const
	{
		JIVE_DEBUG_ASSERT(quotient == 0);
		JIVE_DEBUG_ASSERT(remainder == 0);

		if (divisor.nbits() != nbits())
			throw compiler_error("Unequal number of bits.");

		/*
			FIXME: This should check whether divisor is zero, not whether nbits() is zero.
		*/
		if (divisor.nbits() == 0)
			throw compiler_error("Division by zero.");

		for (size_t n = 0; n < nbits(); n++) {
			remainder = remainder.shl(1);
			remainder[0] = data_[nbits()-n-1];
			if (remainder.uge(divisor) == '1') {
				remainder = remainder.sub(divisor);
				quotient[nbits()-n-1] = '1';
			}
		}
	}

	inline void
	mul(const bitvalue_repr & factor1, const bitvalue_repr & factor2, bitvalue_repr & product) const
	{
		JIVE_DEBUG_ASSERT(product.nbits() == factor1.nbits() + factor2.nbits());

		for (size_t i = 0; i < factor1.nbits(); i++) {
			char c = '0';
			for (size_t j = 0; j < factor2.nbits(); j++) {
				char s = land(factor1[i], factor2[j]);
				char nc = carry(s, product[i+j], c);
				product[i+j] = add(s, product[i+j], c);
				c = nc;
			}
		}
	}

public:
	/*
		FIXME: add <, <=, >, >= operator for uint64_t and int64_t
	*/
	inline bitvalue_repr &
	operator=(const bitvalue_repr & other)
	{
		data_ = other.data_;
		return *this;
	}

	bitvalue_repr &
	operator=(bitvalue_repr && other)
	{
		if (this == &other)
			return *this;

		data_ = std::move(other.data_);
		return *this;
	}

	inline char &
	operator[](size_t n)
	{
		JIVE_DEBUG_ASSERT(n < nbits());
		return data_[n];
	}

	inline const char &
	operator[](size_t n) const
	{
		JIVE_DEBUG_ASSERT(n < nbits());
		return data_[n];
	}

	inline bool
	operator==(const bitvalue_repr & other) const noexcept
	{
		return data_ == other.data_;
	}

	inline bool
	operator!=(const bitvalue_repr & other) const noexcept
	{
		return !(*this == other);
	}

	inline bool
	operator==(int64_t value) const
	{
		return *this == bitvalue_repr(nbits(), value);
	}

	inline bool
	operator!=(int64_t value) const
	{
		return !(*this == bitvalue_repr(nbits(), value));
	}

	inline bool
	operator==(const std::string & other) const noexcept
	{
		if (nbits() != other.size())
			return false;

		for (size_t n = 0; n < other.size(); n++) {
			if (data_[n] != other[n])
				return false;
		}

		return true;
	}

	inline bool
	operator!=(const std::string & other) const noexcept
	{
		return !(*this == other);
	}

	inline char
	sign() const noexcept
	{
		return data_[nbits()-1];
	}

	inline bool
	is_defined() const noexcept
	{
		for (auto bit : data_) {
			if (bit == 'X')
				return false;
		}

		return true;
	}

	inline bool
	is_known() const noexcept
	{
		for (auto bit : data_) {
			if (bit == 'X' || bit == 'D')
				return false;
		}

		return true;
	}

	inline bool
	is_negative() const noexcept
	{
		return sign() == '1';
	}

	inline bitvalue_repr
	concat(const bitvalue_repr & other) const
	{
		bitvalue_repr result(*this);
		result.data_.insert(result.data_.end(), other.data_.begin(), other.data_.end());
		return result;
	}

	inline bitvalue_repr
	slice(size_t low, size_t high) const
	{
		if (high <= low || high > nbits()) {
			throw compiler_error("Slice is out of bound.");
		}

		return bitvalue_repr(std::string(&data_[low], high - low).c_str());
	}

	inline bitvalue_repr
	zext(size_t nbits) const
	{
		if (nbits == 0)
			return *this;

		return concat(bitvalue_repr(nbits, 0));
	}

	inline bitvalue_repr
	sext(size_t nbits) const
	{
		if (nbits == 0)
			return *this;

		return concat(bitvalue_repr::repeat(nbits, sign()));
	}

	inline size_t
	nbits() const noexcept
	{
		return data_.size();
	}

	inline std::string
	str() const
	{
		return std::string(data_.begin(), data_.end());
	}

	uint64_t
	to_uint() const;

	int64_t
	to_int() const;

	inline char
	ult(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		char v = land(lnot(data_[0]), other[0]);
		for (size_t n = 1; n < nbits(); n++)
			v = land(lor(lnot(data_[n]), other[n]), lor(land(lnot(data_[n]), other[n]), v));

		return v;
	}

	inline char
	slt(const bitvalue_repr & other) const
	{
		bitvalue_repr t1(*this), t2(other);
		t1[t1.nbits()-1] = lnot(t1.sign());
		t2[t2.nbits()-1] = lnot(t2.sign());
		return t1.ult(t2);
	}

	inline char
	ule(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		char v = '1';
		for (size_t n = 0; n < nbits(); n++)
			v = land(land(lor(lnot(data_[n]), other[n]), lor(lnot(data_[n]), v)), lor(v, other[n]));

		return v;
	}

	inline char
	sle(const bitvalue_repr & other) const
	{
		bitvalue_repr t1(*this), t2(other);
		t1[t1.nbits()-1] = lnot(t1.sign());
		t2[t2.nbits()-1] = lnot(t2.sign());
		return t1.ule(t2);
	}

	inline char
	ne(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		char v = '0';
		for (size_t n = 0; n < nbits(); n++)
			v = lor(v, lxor(data_[n], other[n]));
		return v;
	}

	inline char
	eq(const bitvalue_repr & other) const
	{
		return lnot(ne(other));
	}

	inline char
	sge(const bitvalue_repr & other) const
	{
		return lnot(slt(other));
	}

	inline char
	uge(const bitvalue_repr & other) const
	{
		return lnot(ult(other));
	}

	inline char
	sgt(const bitvalue_repr & other) const
	{
		return lnot(sle(other));
	}

	inline char
	ugt(const bitvalue_repr & other) const
	{
		return lnot(ule(other));
	}

	inline bitvalue_repr
	add(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		char c = '0';
		bitvalue_repr sum = repeat(nbits(), 'X');
		for (size_t n = 0; n < nbits(); n++) {
			sum[n] = add(data_[n], other[n], c);
			c = carry(data_[n], other[n], c);
		}

		return sum;
	}

	inline bitvalue_repr
	land(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		bitvalue_repr result = repeat(nbits(), 'X');
		for (size_t n = 0; n < nbits(); n++)
			result[n] = land(data_[n], other[n]);

		return result;
	}

	inline bitvalue_repr
	lor(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		bitvalue_repr result = repeat(nbits(), 'X');
		for (size_t n = 0; n < nbits(); n++)
			result[n] = lor(data_[n], other[n]);

		return result;
	}

	inline bitvalue_repr
	lxor(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		bitvalue_repr result = repeat(nbits(), 'X');
		for (size_t n = 0; n < nbits(); n++)
			result[n] = lxor(data_[n], other[n]);

		return result;
	}

	inline bitvalue_repr
	lnot() const
	{
		return lxor(repeat(nbits(), '1'));
	}

	inline bitvalue_repr
	neg() const
	{
		char c = '1';
		bitvalue_repr result = repeat(nbits(), 'X');
		for (size_t n = 0; n < nbits(); n++) {
			char tmp = lxor(data_[n], '1');
			result[n] = add(tmp, '0', c);
			c = carry(tmp, '0', c);
		}

		return result;
	}

	inline bitvalue_repr
	sub(const bitvalue_repr & other) const
	{
		return add(other.neg());
	}

	inline bitvalue_repr
	shr(size_t shift) const
	{
		if (shift >= nbits())
			return repeat(nbits(), '0');

		bitvalue_repr result(std::string(&data_[shift], nbits()-shift).c_str());
		return result.zext(shift);
	}

	inline bitvalue_repr
	ashr(size_t shift) const
	{
		if (shift >= nbits())
			return repeat(nbits(), sign());

		bitvalue_repr result(std::string(&data_[shift], nbits()-shift).c_str());
		return result.sext(shift);
	}

	inline bitvalue_repr
	shl(size_t shift) const
	{
		if (shift >= nbits())
			return repeat(nbits(), '0');

		return repeat(shift, '0').concat(slice(0, nbits()-shift));
	}

	inline bitvalue_repr
	udiv(const bitvalue_repr & other) const
	{
		bitvalue_repr quotient(nbits(), 0);
		bitvalue_repr remainder(nbits(), 0);
		udiv(other, quotient, remainder);
		return quotient;
	}

	inline bitvalue_repr
	umod(const bitvalue_repr & other) const
	{
		bitvalue_repr quotient(nbits(), 0);
		bitvalue_repr remainder(nbits(), 0);
		udiv(other, quotient, remainder);
		return remainder;
	}

	inline bitvalue_repr
	sdiv(const bitvalue_repr & other) const
	{
		bitvalue_repr dividend(*this), divisor(other);

		if (dividend.is_negative())
			dividend = dividend.neg();

		if (divisor.is_negative())
			divisor = divisor.neg();

		bitvalue_repr quotient(nbits(), 0), remainder(nbits(), 0);
		dividend.udiv(divisor, quotient, remainder);

		if (is_negative())
			remainder = remainder.neg();

		if (is_negative() ^ other.is_negative())
			quotient = quotient.neg();

		return quotient;
	}

	inline bitvalue_repr
	smod(const bitvalue_repr & other) const
	{
		bitvalue_repr dividend(*this), divisor(other);

		if (dividend.is_negative())
			dividend = dividend.neg();

		if (divisor.is_negative())
			divisor = divisor.neg();

		bitvalue_repr quotient(nbits(), 0), remainder(nbits(), 0);
		dividend.udiv(divisor, quotient, remainder);

		if (is_negative())
			remainder = remainder.neg();

		if (is_negative() ^ other.is_negative())
			quotient = quotient.neg();

		return remainder;
	}

	inline bitvalue_repr
	mul(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		bitvalue_repr product(2*nbits(), 0);
		mul(*this, other, product);
		return product.slice(0, nbits());
	}

	inline bitvalue_repr
	umulh(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		bitvalue_repr product(4*nbits(), 0);
		bitvalue_repr factor1 = this->zext(nbits());
		bitvalue_repr factor2 = other.zext(nbits());
		mul(factor1, factor2, product);
		return product.slice(nbits(), 2*nbits());
	}

	inline bitvalue_repr
	smulh(const bitvalue_repr & other) const
	{
		if (nbits() != other.nbits())
			throw compiler_error("Unequal number of bits.");

		bitvalue_repr product(4*nbits(), 0);
		bitvalue_repr factor1 = this->sext(nbits());
		bitvalue_repr factor2 = other.sext(nbits());
		mul(factor1, factor2, product);
		return product.slice(nbits(), 2*nbits());
	}

private:
	/* [lsb ... msb] */
	std::vector<char> data_;
};

}

#endif
