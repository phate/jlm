/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_UTIL_PTR_COLLECTION_HPP
#define JIVE_UTIL_PTR_COLLECTION_HPP

#include <memory>
#include <vector>

/*
 * Utilities for managing collections of "pointer-like things" objects.
 *
 * Utility functions to assist in managing containers (mostly vectors
 * and arrays) that contain pointers (mostly std::unique_ptr) to objects
 * to allow polymorphism of elements.
 */

namespace jive {
namespace detail {

/* Compare for element-wise equality. Assumes an operator== defined on
 * the container elements */
template<typename Container1, typename Container2>
static inline bool
ptr_container_equals(const Container1 & first, const Container2 & second)
{
	typename Container1::const_iterator i1 = first.begin();
	typename Container2::const_iterator i2 = second.begin();

	while (i1 != first.end()) {
		if (i2 == second.end()) {
			return false;
		}
		if (**i1 != **i2) {
			return false;
		}
		++i1;
		++i2;
	}
	return i2 == second.end();
}

/* Derive type of "pointed to" object for plain pointers/unique_ptr/
 * shared_ptr
 */
template<typename T>
struct pointee_type { typedef typename T::element_type type; };
template<typename T>
struct pointee_type<T *> { typedef T type; };
template<typename T>
struct pointee_type<T * const> { typedef T type; };

/* Derive type of "pointed to" object for containers of plain
 * pointers/unique_ptr/shared_ptr
 */
template<typename T>
struct container_pointee_type { typedef typename pointee_type<typename T::value_type>::type type; };
template<typename T>
struct container_pointee_type<T *> { typedef typename pointee_type<T>::type type; };

/* Perform element-wise copy into a vector. Assumes that there is a
 * "copy" member function on the elements */
template<typename Container>
static inline
std::vector<std::unique_ptr<typename container_pointee_type<Container>::type>>
unique_ptr_vector_copy(const Container & container)
{
	std::vector<std::unique_ptr<typename container_pointee_type<Container>::type>> result;
	result.reserve(container.size());
	for (const auto & t : container) {
		result.emplace_back(t->copy());
	}
	return result;
}

/* Adapt an array specified as base + size as a container */
template<typename T>
class array_slice {
public:
	typedef T * iterator;
	typedef T value_type;

	inline constexpr
	array_slice(T * begin, T * end) noexcept
		: begin_(begin), end_(end)
	{
	}

	inline iterator begin() const noexcept { return begin_; }
	inline iterator end() const noexcept { return end_; }

	inline size_t size() const noexcept { return end_ - begin_; }

private:
	iterator begin_;
	iterator end_;
};

template<typename T>
inline constexpr array_slice<T>
make_array_slice(T * begin, size_t size) noexcept
{
	return array_slice<T>(begin, begin + size);
}

}
}

#endif
