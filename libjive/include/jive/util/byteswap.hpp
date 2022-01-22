/*
 * Copyright 2013 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_UTIL_BYTESWAP_HPP
#define JIVE_UTIL_BYTESWAP_HPP

#include <byteswap.hpp>
#include <endian.hpp>
#include <stdint.hpp>

#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint16_t jive_cpu_to_be16(uint16_t v) { return bswap_16(v); }
static inline uint32_t jive_cpu_to_be32(uint32_t v) { return bswap_32(v); }
static inline uint64_t jive_cpu_to_be64(uint64_t v) { return bswap_64(v); }
static inline uint16_t jive_cpu_to_le16(uint16_t v) { return v; }
static inline uint32_t jive_cpu_to_le32(uint32_t v) { return v; }
static inline uint64_t jive_cpu_to_le64(uint64_t v) { return v; }
static inline uint16_t jive_be16_to_cpu(uint16_t v) { return bswap_16(v); }
static inline uint32_t jive_be32_to_cpu(uint32_t v) { return bswap_32(v); }
static inline uint64_t jive_be64_to_cpu(uint64_t v) { return bswap_64(v); }
static inline uint16_t jive_le16_to_cpu(uint16_t v) { return v; }
static inline uint32_t jive_le32_to_cpu(uint32_t v) { return v; }
static inline uint64_t jive_le64_to_cpu(uint64_t v) { return v; }
#elif __BYTE_ORDER == __BIG_ENDIAN
static inline uint16_t jive_cpu_to_be16(uint16_t v) { return v; }
static inline uint32_t jive_cpu_to_be32(uint32_t v) { return v; }
static inline uint64_t jive_cpu_to_be64(uint64_t v) { return v; }
static inline uint16_t jive_cpu_to_le16(uint16_t v) { return bswap_16(v); }
static inline uint32_t jive_cpu_to_le32(uint32_t v) { return bswap_32(v); }
static inline uint64_t jive_cpu_to_le64(uint64_t v) { return bswap_64(v); }
static inline uint16_t jive_be16_to_cpu(uint16_t v) { return v; }
static inline uint32_t jive_be32_to_cpu(uint32_t v) { return v; }
static inline uint64_t jive_be64_to_cpu(uint64_t v) { return v; }
static inline uint16_t jive_le16_to_cpu(uint16_t v) { return bswap_16(v); }
static inline uint32_t jive_le32_to_cpu(uint32_t v) { return bswap_32(v); }
static inline uint64_t jive_le64_to_cpu(uint64_t v) { return bswap_64(v); }
#else
#error Unknown endian
#endif

#endif
