/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_VALUE_REPRESENTATION_HPP
#define JLM_RVSDG_BITSTRING_VALUE_REPRESENTATION_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/strfmt.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace jlm::rvsdg
{

/**
 Value representation used for compile-time evaluation of bitstring
 expressions. A bit is either:
  - '0' : zero
  - '1' : one
  - 'D' : defined, but unknown
  - 'X' : undefined and unknown
*/

class BitValueRepresentation
{
public:
  BitValueRepresentation(size_t nbits, int64_t value)
  {
    if (nbits == 0)
      throw util::Error("Number of bits is zero.");

    if (nbits < 64 && (value >> nbits) != 0 && (value >> nbits != -1))
      throw util::Error("Value cannot be represented with the given number of bits.");

    for (size_t n = 0; n < nbits; ++n)
    {
      data_.push_back('0' + (value & 1));
      value = value >> 1;
    }
  }

  BitValueRepresentation(const char * s)
  {
    if (strlen(s) == 0)
      throw util::Error("Number of bits is zero.");

    for (size_t n = 0; n < strlen(s); n++)
    {
      if (s[n] != '0' && s[n] != '1' && s[n] != 'X' && s[n] != 'D')
        throw util::Error("Not a valid bit.");
      data_.push_back(s[n]);
    }
  }

  BitValueRepresentation(const BitValueRepresentation & other)
      : data_(other.data_)
  {}

  BitValueRepresentation(BitValueRepresentation && other)
      : data_(std::move(other.data_))
  {}

  static BitValueRepresentation
  repeat(size_t nbits, char bit)
  {
    return BitValueRepresentation(std::string(nbits, bit).c_str());
  }

private:
  inline char
  lor(char a, char b) const noexcept
  {
    switch (a)
    {
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
    switch (a)
    {
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
    switch (a)
    {
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
    return lor(lor(land(a, b), land(a, c)), land(b, c));
  }

  inline char
  add(char a, char b, char c) const noexcept
  {
    return lxor(lxor(a, b), c);
  }

  inline void
  udiv(
      const BitValueRepresentation & divisor,
      BitValueRepresentation & quotient,
      BitValueRepresentation & remainder) const
  {
    JLM_ASSERT(quotient == 0);
    JLM_ASSERT(remainder == 0);

    if (divisor.nbits() != nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in udiv, ", divisor.nbits(), " != ", nbits()));

    /*
      FIXME: This should check whether divisor is zero, not whether nbits() is zero.
    */
    if (divisor.nbits() == 0)
      throw util::Error("Division by zero.");

    for (size_t n = 0; n < nbits(); n++)
    {
      remainder = remainder.shl(1);
      remainder[0] = data_[nbits() - n - 1];
      if (remainder.uge(divisor) == '1')
      {
        remainder = remainder.sub(divisor);
        quotient[nbits() - n - 1] = '1';
      }
    }
  }

  inline void
  mul(const BitValueRepresentation & factor1,
      const BitValueRepresentation & factor2,
      BitValueRepresentation & product) const
  {
    JLM_ASSERT(product.nbits() == factor1.nbits() + factor2.nbits());

    for (size_t i = 0; i < factor1.nbits(); i++)
    {
      char c = '0';
      for (size_t j = 0; j < factor2.nbits(); j++)
      {
        char s = land(factor1[i], factor2[j]);
        char nc = carry(s, product[i + j], c);
        product[i + j] = add(s, product[i + j], c);
        c = nc;
      }
    }
  }

public:
  /*
    FIXME: add <, <=, >, >= operator for uint64_t and int64_t
  */
  BitValueRepresentation &
  operator=(const BitValueRepresentation & other)
  {
    data_ = other.data_;
    return *this;
  }

  BitValueRepresentation &
  operator=(BitValueRepresentation && other)
  {
    if (this == &other)
      return *this;

    data_ = std::move(other.data_);
    return *this;
  }

  inline char &
  operator[](size_t n)
  {
    JLM_ASSERT(n < nbits());
    return data_[n];
  }

  inline const char &
  operator[](size_t n) const
  {
    JLM_ASSERT(n < nbits());
    return data_[n];
  }

  inline bool
  operator==(const BitValueRepresentation & other) const noexcept
  {
    return data_ == other.data_;
  }

  inline bool
  operator!=(const BitValueRepresentation & other) const noexcept
  {
    return !(*this == other);
  }

  inline bool
  operator==(int64_t value) const
  {
    return *this == BitValueRepresentation(nbits(), value);
  }

  inline bool
  operator!=(int64_t value) const
  {
    return !(*this == BitValueRepresentation(nbits(), value));
  }

  inline bool
  operator==(const std::string & other) const noexcept
  {
    if (nbits() != other.size())
      return false;

    for (size_t n = 0; n < other.size(); n++)
    {
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
    return data_[nbits() - 1];
  }

  inline bool
  is_defined() const noexcept
  {
    for (auto bit : data_)
    {
      if (bit == 'X')
        return false;
    }

    return true;
  }

  inline bool
  is_known() const noexcept
  {
    for (auto bit : data_)
    {
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

  BitValueRepresentation
  concat(const BitValueRepresentation & other) const
  {
    BitValueRepresentation result(*this);
    result.data_.insert(result.data_.end(), other.data_.begin(), other.data_.end());
    return result;
  }

  BitValueRepresentation
  slice(size_t low, size_t high) const
  {
    if (high <= low || high > nbits())
    {
      throw util::Error("Slice is out of bound.");
    }

    return BitValueRepresentation(std::string(&data_[low], high - low).c_str());
  }

  BitValueRepresentation
  zext(size_t nbits) const
  {
    if (nbits == 0)
      return *this;

    return concat(BitValueRepresentation(nbits, 0));
  }

  BitValueRepresentation
  sext(size_t nbits) const
  {
    if (nbits == 0)
      return *this;

    return concat(BitValueRepresentation::repeat(nbits, sign()));
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
  ult(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in ult, ", nbits(), " != ", other.nbits()));

    char v = land(lnot(data_[0]), other[0]);
    for (size_t n = 1; n < nbits(); n++)
      v = land(lor(lnot(data_[n]), other[n]), lor(land(lnot(data_[n]), other[n]), v));

    return v;
  }

  inline char
  slt(const BitValueRepresentation & other) const
  {
    BitValueRepresentation t1(*this), t2(other);
    t1[t1.nbits() - 1] = lnot(t1.sign());
    t2[t2.nbits() - 1] = lnot(t2.sign());
    return t1.ult(t2);
  }

  inline char
  ule(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in ule, ", nbits(), " != ", other.nbits()));

    char v = '1';
    for (size_t n = 0; n < nbits(); n++)
      v = land(land(lor(lnot(data_[n]), other[n]), lor(lnot(data_[n]), v)), lor(v, other[n]));

    return v;
  }

  inline char
  sle(const BitValueRepresentation & other) const
  {
    BitValueRepresentation t1(*this), t2(other);
    t1[t1.nbits() - 1] = lnot(t1.sign());
    t2[t2.nbits() - 1] = lnot(t2.sign());
    return t1.ule(t2);
  }

  inline char
  ne(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in ne, ", nbits(), " != ", other.nbits()));

    char v = '0';
    for (size_t n = 0; n < nbits(); n++)
      v = lor(v, lxor(data_[n], other[n]));
    return v;
  }

  inline char
  eq(const BitValueRepresentation & other) const
  {
    return lnot(ne(other));
  }

  inline char
  sge(const BitValueRepresentation & other) const
  {
    return lnot(slt(other));
  }

  inline char
  uge(const BitValueRepresentation & other) const
  {
    return lnot(ult(other));
  }

  inline char
  sgt(const BitValueRepresentation & other) const
  {
    return lnot(sle(other));
  }

  inline char
  ugt(const BitValueRepresentation & other) const
  {
    return lnot(ule(other));
  }

  BitValueRepresentation
  add(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in add, ", nbits(), " != ", other.nbits()));

    char c = '0';
    BitValueRepresentation sum = repeat(nbits(), 'X');
    for (size_t n = 0; n < nbits(); n++)
    {
      sum[n] = add(data_[n], other[n], c);
      c = carry(data_[n], other[n], c);
    }

    return sum;
  }

  BitValueRepresentation
  land(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in land, ", nbits(), " != ", other.nbits()));

    BitValueRepresentation result = repeat(nbits(), 'X');
    for (size_t n = 0; n < nbits(); n++)
      result[n] = land(data_[n], other[n]);

    return result;
  }

  BitValueRepresentation
  lor(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in lor, ", nbits(), " != ", other.nbits()));

    BitValueRepresentation result = repeat(nbits(), 'X');
    for (size_t n = 0; n < nbits(); n++)
      result[n] = lor(data_[n], other[n]);

    return result;
  }

  BitValueRepresentation
  lxor(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in lxor, ", nbits(), " != ", other.nbits()));

    BitValueRepresentation result = repeat(nbits(), 'X');
    for (size_t n = 0; n < nbits(); n++)
      result[n] = lxor(data_[n], other[n]);

    return result;
  }

  BitValueRepresentation
  lnot() const
  {
    return lxor(repeat(nbits(), '1'));
  }

  BitValueRepresentation
  neg() const
  {
    char c = '1';
    BitValueRepresentation result = repeat(nbits(), 'X');
    for (size_t n = 0; n < nbits(); n++)
    {
      char tmp = lxor(data_[n], '1');
      result[n] = add(tmp, '0', c);
      c = carry(tmp, '0', c);
    }

    return result;
  }

  BitValueRepresentation
  sub(const BitValueRepresentation & other) const
  {
    return add(other.neg());
  }

  BitValueRepresentation
  shr(size_t shift) const
  {
    if (shift >= nbits())
      return repeat(nbits(), '0');

    BitValueRepresentation result(std::string(&data_[shift], nbits() - shift).c_str());
    return result.zext(shift);
  }

  BitValueRepresentation
  ashr(size_t shift) const
  {
    if (shift >= nbits())
      return repeat(nbits(), sign());

    BitValueRepresentation result(std::string(&data_[shift], nbits() - shift).c_str());
    return result.sext(shift);
  }

  BitValueRepresentation
  shl(size_t shift) const
  {
    if (shift >= nbits())
      return repeat(nbits(), '0');

    return repeat(shift, '0').concat(slice(0, nbits() - shift));
  }

  BitValueRepresentation
  udiv(const BitValueRepresentation & other) const
  {
    BitValueRepresentation quotient(nbits(), 0);
    BitValueRepresentation remainder(nbits(), 0);
    udiv(other, quotient, remainder);
    return quotient;
  }

  BitValueRepresentation
  umod(const BitValueRepresentation & other) const
  {
    BitValueRepresentation quotient(nbits(), 0);
    BitValueRepresentation remainder(nbits(), 0);
    udiv(other, quotient, remainder);
    return remainder;
  }

  BitValueRepresentation
  sdiv(const BitValueRepresentation & other) const
  {
    BitValueRepresentation dividend(*this), divisor(other);

    if (dividend.is_negative())
      dividend = dividend.neg();

    if (divisor.is_negative())
      divisor = divisor.neg();

    BitValueRepresentation quotient(nbits(), 0), remainder(nbits(), 0);
    dividend.udiv(divisor, quotient, remainder);

    if (is_negative())
      remainder = remainder.neg();

    if (is_negative() ^ other.is_negative())
      quotient = quotient.neg();

    return quotient;
  }

  BitValueRepresentation
  smod(const BitValueRepresentation & other) const
  {
    BitValueRepresentation dividend(*this), divisor(other);

    if (dividend.is_negative())
      dividend = dividend.neg();

    if (divisor.is_negative())
      divisor = divisor.neg();

    BitValueRepresentation quotient(nbits(), 0), remainder(nbits(), 0);
    dividend.udiv(divisor, quotient, remainder);

    if (is_negative())
      remainder = remainder.neg();

    if (is_negative() ^ other.is_negative())
      quotient = quotient.neg();

    return remainder;
  }

  BitValueRepresentation
  mul(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in mul, ", nbits(), " != ", other.nbits()));

    BitValueRepresentation product(2 * nbits(), 0);
    mul(*this, other, product);
    return product.slice(0, nbits());
  }

  BitValueRepresentation
  umulh(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in umulh, ", nbits(), " != ", other.nbits()));

    BitValueRepresentation product(4 * nbits(), 0);
    BitValueRepresentation factor1 = this->zext(nbits());
    BitValueRepresentation factor2 = other.zext(nbits());
    mul(factor1, factor2, product);
    return product.slice(nbits(), 2 * nbits());
  }

  BitValueRepresentation
  smulh(const BitValueRepresentation & other) const
  {
    if (nbits() != other.nbits())
      throw util::Error(
          jlm::util::strfmt("Unequal number of bits in smulh, ", nbits(), " != ", other.nbits()));

    BitValueRepresentation product(4 * nbits(), 0);
    BitValueRepresentation factor1 = this->sext(nbits());
    BitValueRepresentation factor2 = other.sext(nbits());
    mul(factor1, factor2, product);
    return product.slice(nbits(), 2 * nbits());
  }

  void
  Append(const BitValueRepresentation & other)
  {
    data_.insert(data_.end(), other.data_.begin(), other.data_.end());
  }

private:
  /* [lsb ... msb] */
  std::vector<char> data_;
};

}

#endif
