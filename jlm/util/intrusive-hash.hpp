/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_INTRUSIVE_HASH_HPP
#define JLM_UTIL_INTRUSIVE_HASH_HPP

#include <functional>
#include <memory>
#include <utility>
#include <vector>

/*
 * Implementation of intrusive_hash data structure
 *
 * An intrusive_hash is a data structure linking multiple objects such that
 * can be looked up efficiently by a specific key. The intrusive_hash is not
 * a container: It does not "own" the objects referenced. Additionally it
 * also does not manage memory to represent the linkage: The anchor data
 * structure for forming the linkage are required to be members of the
 * objects themselves. Any object can be member of an arbitrary number of
 * such intrusive_hash collections.
 *
 * Usage:
 *
 * For a class declared as the following:
 *
 * class X {
 * private:
 *   int num;
 *   jive::detail::intrusive_hash_anchor<X> num_hash_anchor;
 * public:
 *   typedef jive::detail::intrusive_hash_accessor<
 *     int,                // key type
 *     X,                  // element type
 *     &X::num,            // key member
 *     &X::num_hash_anchor // anchor member
 *   > num_hash_accessor;
 * };
 *
 * an intrusive_hash data structure can be declared in the following way:
 *
 * typedef jive::detail::intrusive_hash<
 *   int,
 *   X,
 *   X::num_hash_accessor
 * > num_hash;
 *
 * It is possible to implement a custom accessor instead of using the
 * template-generated one. In this case the get_key, get_prev, get_next,
 * set_prev and set_next members must be implemented appropriately.
 *
 * An object h of num_hash class then supports STL-style operations
 * - num_hash::iterator, num_hash::const_iterator for iteration
 * - h.begin(), h.end() and const-qualified variants
 * - h.find(key) performs lookup and yields an iterator
 * - h.insert(element) links an object into the data structure
 * - h.erase(element) or h.erase(iterator) unlinks an object from the data
 *   structure
 * - h.size() and h.empty() testing
 *
 * The implementation requires the following guarantees:
 * - the accessor to key and anchor are noexcept
 * - the hash function is noexcept
 * - while linked to the data structure, an objects key may not change
 *
 * The implementation provides the following guarantees
 * - insert is O(1) amortized
 * - erase, empty, size are O(1) and noexcept
 * - find is O(1) average and noexcept
 * - end is O(1)
 * - begin is O(n)
 * - inserting a new object does not invalidate iterators, but may
 *   invalidate traversal order
 * - erase an object does not invalidate iterators except those pointing
 *   to the object removed and does not invalidate traversal order
 *
 * An additional template "owner_intrusive_hash" implements the same
 * interface, but in addition assumes "ownership" of the objects it contains.
 * This means that upon destruction of the container, the elements will
 * be deleted as well. In particular, it differs in the following ways:
 *
 * - insert expects a std::unique_ptr as argument
 * - erase is only supported on keys and iterators and will cause the
 *   element in question to be deleted
 * - a new method "unlink" removes an element from the hash and returns
 *   a std::unique_ptr to it
 */

namespace jive {
namespace detail {

// FIXME: for some weird reason, std::equal_to does not specify noexcept, so
// define our own equality comparison operator here
template<typename T>
struct safe_equal {
	inline bool
	operator()(const T & a, const T & b) const
		noexcept(
			noexcept(std::declval<const T&>() == std::declval<const T&>()))
	{
		return a == b;
	}
};

template<>
struct safe_equal<std::string> {
	inline bool
	operator()(const std::string & a, const std::string & b) const noexcept
	{
		// C++11 lacks "noexcept" specification for
		// std::string::operator==, but implementations lacking
		// this would be a weird breed. So declare it noexcept
		// by "fiat".
		return a == b;
	}
};

template<
	typename KeyType,
	typename ElementType,
	typename Accessor,
	typename KeyHash = std::hash<KeyType>,
	typename KeyEqual = safe_equal<KeyType>>
class intrusive_hash {
private:
	struct bucket_type {
		constexpr inline bucket_type() noexcept
			: first(nullptr)
			, last(nullptr)
		{
		}

		ElementType * first;
		ElementType * last;
	};

public:
	static_assert(
		noexcept(KeyHash()(std::declval<KeyType &>())),
		"require noexcept hash");
	static_assert(
		noexcept(Accessor().get_key(nullptr)),
		"require noexcept get_key");
	static_assert(
		noexcept(Accessor().get_prev(nullptr)),
		"require noexcept get_prev");
	static_assert(
		noexcept(Accessor().get_next(nullptr)),
		"require noexcept get_next");
	static_assert(
		noexcept(Accessor().set_prev(nullptr, nullptr)),
		"require noexcept set_prev");
	static_assert(
		noexcept(Accessor().set_next(nullptr, nullptr)),
		"require noexcept set_next");
	static_assert(
		noexcept(KeyEqual()(std::declval<KeyType &>(), std::declval<KeyType &>())),
		"require noexcept key equality");
	class const_iterator;
	class iterator {
	public:
		typedef ElementType value_type;
		typedef ElementType * pointer;
		typedef ElementType & reference;
		typedef std::forward_iterator_tag iterator_category;
		typedef size_t size_type;
		typedef ssize_t difference_type;

		constexpr iterator() noexcept
			: map_(nullptr)
			, element_(nullptr)
		{
		}

		constexpr iterator(
			const intrusive_hash * map,
			ElementType * object)
			: map_(map)
			, element_(object)
		{
		}
	
		inline const iterator &
		operator++(void) noexcept
		{
			ElementType * next = map_->accessor_.get_next(element_);
			if (next == nullptr) {
				size_t hash = map_->hash_(map_->accessor_.get_key(element_));
				size_t index = (hash & map_->mask_) + 1;
				while (next == nullptr && index < map_->buckets_.size()) {
					next = map_->buckets_[index].first;
					++index;
				}
			}
			element_ = next;
			return *this;
		}

		inline iterator
		operator++(int) noexcept
		{
			iterator i = *this;
			++*this;
			return i;
		}

		inline ElementType &
		operator*() const noexcept
		{
			return *element_;
		}

		inline ElementType *
		operator->() const noexcept
		{
			return element_;
		}

		inline bool
		operator==(const iterator & other) const noexcept
		{
			return element_ == other.element_;
		}

		inline bool
		operator!=(const iterator & other) const noexcept
		{
			return element_ != other.element_;
		}

		inline ElementType *
		ptr() const noexcept
		{
			return element_;
		}

	private:
		const intrusive_hash * map_;
		ElementType * element_;
		friend class const_iterator;
	};

	class const_iterator {
	public:
		typedef const ElementType value_type;
		typedef const ElementType * pointer;
		typedef const ElementType & reference;
		typedef std::forward_iterator_tag iterator_category;
		typedef size_t size_type;
		typedef ssize_t difference_type;

		constexpr const_iterator(const const_iterator & other) noexcept = default;
		constexpr const_iterator(const iterator & other) noexcept
			: map_(other.map_)
			, element_(other.element_)
		{
		}

		constexpr const_iterator() noexcept
			: map_(nullptr)
			, element_(nullptr)
		{
		}

		constexpr const_iterator(
			const intrusive_hash * map,
			const ElementType * object)
			: map_(map)
			, element_(object)
		{
		}
	
		inline const const_iterator &
		operator++(void) noexcept
		{
			ElementType * next = map_->accessor_.get_next(element_);
			if (next == nullptr) {
				size_t hash = map_->hash_(map_->accessor_.get_key(element_));
				size_t index = (hash & map_->mask_) + 1;
				while (next == nullptr && index < map_->buckets_.size()) {
					next = map_->buckets_[index].first;
					++index;
				}
			}
			element_ = next;
			return *this;
		}

		inline const_iterator
		operator++(int) noexcept
		{
			const_iterator i = *this;
			++*this;
			return i;
		}

		inline const ElementType &
		operator*() const noexcept
		{
			return *element_;
		}

		inline const ElementType *
		operator->() const noexcept
		{
			return element_;
		}

		inline bool
		operator==(const const_iterator & other) const noexcept
		{
			return element_ == other.element_;
		}

		inline bool
		operator!=(const const_iterator & other) const noexcept
		{
			return element_ != other.element_;
		}

		inline const ElementType *
		ptr() const noexcept
		{
			return element_;
		}

	private:
		const intrusive_hash * map_;
		const ElementType * element_;
	};

	typedef ElementType value_type;
	typedef ElementType mapped_type;
	typedef KeyType key_type;
	typedef size_t size_type;

	inline constexpr
	intrusive_hash() noexcept
		: size_(0)
		, mask_(0)
	{
	}

	intrusive_hash(const intrusive_hash & other) = delete;

	void operator=(const intrusive_hash & other) = delete;

	intrusive_hash(intrusive_hash && other) noexcept
		: intrusive_hash()
	{
		swap(other);
	}

	void swap(intrusive_hash & other) noexcept
	{
		buckets_.swap(other.buckets_);
		std::swap(size_, other.size_);
		std::swap(mask_, other.mask_);
	}

	void
	clear() noexcept
	{
		for (bucket_type & bucket : buckets_) {
			bucket.first = nullptr;
			bucket.last = nullptr;
			size_ = 0;
		}
	}

	inline iterator
	insert(ElementType * element)
	{
		++size_;
		if (size_ > buckets_.size()) {
			rehash();
		}
		private_insert_into(buckets_, mask_, element);
		
		return iterator(this, element);
	}

	inline void
	erase(ElementType * element) noexcept
	{
		size_t index = hash_(accessor_.get_key(element)) & mask_;
		bucket_type & b = buckets_[index];
		ElementType * prev = accessor_.get_prev(element);
		ElementType * next = accessor_.get_next(element);
		if (prev) {
			accessor_.set_next(prev, next);
		} else {
			b.first = next;
		}
		if (next) {
			accessor_.set_prev(next, prev);
		} else {
			b.last = prev;
		}
		--size_;
	}

	inline void
	erase(iterator i) noexcept
	{
		erase(i.ptr());
	}

	inline void
	erase(const KeyType & key) noexcept
	{
		iterator i = find(key);
		if (i != end()) {
			erase(i);
		}
	}

	inline void
	erase(iterator begin, iterator end) noexcept
	{
		while (begin != end) {
			ElementType * element = begin.ptr();
			++begin;
			erase(element);
		}
	}

	inline size_type
	size() const noexcept
	{
		return size_;
	}

	inline bool
	empty() const noexcept
	{
		return size() == 0;
	}

	iterator
	begin() noexcept
	{
		return iterator(this, first_object());
	}

	iterator
	end() noexcept
	{
		return iterator(this, nullptr);
	}

	const_iterator
	cbegin() const noexcept
	{
		return const_iterator(this, first_object());
	}

	const_iterator
	cend() const noexcept
	{
		return const_iterator(this, nullptr);
	}

	const_iterator
	begin() const noexcept
	{
		return cbegin();
	}

	const_iterator
	end() const noexcept
	{
		return cend();
	}

	inline iterator
	find(const KeyType & key) noexcept
	{
		return iterator(this, lookup(key));
	}

	inline const_iterator
	find(const KeyType & key) const noexcept
	{
		return const_iterator(this, lookup(key));
	}

private:
	ElementType *
	first_object() const noexcept
	{
		for (const bucket_type & bucket : buckets_) {
			if (bucket.first) {
				return bucket.first;
			}
		}
		return nullptr;
	}

	size_t size_;
	size_t mask_;
	std::vector<bucket_type> buckets_;

	inline void
	private_insert_into(
		std::vector<bucket_type> & bucket_types,
		size_t mask,
		ElementType * element) noexcept
	{
		size_t index = hash_(accessor_.get_key(element)) & mask;
		accessor_.set_prev(element, bucket_types[index].last);
		accessor_.set_next(element, nullptr);
		if (bucket_types[index].last) {
			accessor_.set_next(bucket_types[index].last, element);
		} else {
			bucket_types[index].first = element;
		}
		bucket_types[index].last = element;
	}

	void
	rehash()
	{
		std::vector<bucket_type> new_buckets(
			std::max(
				typename decltype(buckets_)::size_type(1),
				buckets_.size() * 2),
			bucket_type());
		size_t new_mask = new_buckets.size() - 1;
		
		for (bucket_type & old_bucket_type : buckets_) {
			ElementType * element = old_bucket_type.first;
			while (element) {
				ElementType * next = accessor_.get_next(element);
				private_insert_into(new_buckets, new_mask, element);
				element = next;
			}
		}
		
		buckets_.swap(new_buckets);
		mask_ = new_mask;
	}
	
	inline ElementType *
	lookup(const KeyType & key) const noexcept
	{
		if (empty()) {
			return nullptr;
		}

		size_t index = hash_(key) & mask_;
		ElementType * element = buckets_[index].first;
		while (element && !equal_(key, accessor_.get_key(element))) {
			element = accessor_.get_next(element);
		}

		return element;
	}

	Accessor accessor_;
	KeyHash hash_;
	KeyEqual equal_;
};

template<
	typename ElementType>
class intrusive_hash_anchor {
public:
	ElementType * prev;
	ElementType * next;
};

template<
	typename KeyType,
	typename ElementType,
	KeyType ElementType::*key_member,
	intrusive_hash_anchor<ElementType> ElementType::*anchor_member>
class intrusive_hash_accessor {
public:
	inline KeyType
	get_key(const ElementType * element) const noexcept
	{
		return element->*key_member;
	}
	inline ElementType *
	get_prev(const ElementType * element) const noexcept
	{
		return (element->*anchor_member).prev;
	}
	inline void
	set_prev(ElementType * element, ElementType * prev) const noexcept
	{
		(element->*anchor_member).prev = prev;
	}
	inline ElementType *
	get_next(const ElementType * element) const noexcept
	{
		return (element->*anchor_member).next;
	}
	inline void
	set_next(ElementType * element, ElementType * next) const noexcept
	{
		(element->*anchor_member).next = next;
	}
};

template<
	typename KeyType,
	typename ElementType,
	typename Accessor,
	typename KeyHash = std::hash<KeyType>,
	typename KeyEqual = safe_equal<KeyType>>
class owner_intrusive_hash {
private:
	typedef intrusive_hash<KeyType, ElementType, Accessor, KeyHash, KeyEqual>
		internal_hash_type;
public:
	static_assert(
		noexcept(std::declval<ElementType&>().~ElementType()),
		"Require noexcept destructor for ElementType");
	typedef typename internal_hash_type::const_iterator const_iterator;
	typedef typename internal_hash_type::iterator iterator;
	typedef typename internal_hash_type::value_type value_type;
	typedef typename internal_hash_type::mapped_type mapped_type;
	typedef typename internal_hash_type::key_type key_type;
	typedef typename internal_hash_type::size_type size_type;

	~owner_intrusive_hash() noexcept
	{
		clear();
	}

	inline constexpr
	owner_intrusive_hash() noexcept
	{
	}

	owner_intrusive_hash(const owner_intrusive_hash & other) = delete;

	void operator=(const owner_intrusive_hash & other) = delete;

	owner_intrusive_hash(owner_intrusive_hash && other) noexcept
		: internal_hash_(std::move(other.internal_hash_))
	{
	}

	void swap(owner_intrusive_hash & other) noexcept
	{
		internal_hash_.swap(other.internal_hash_);
	}

	void
	clear() noexcept
	{
		iterator i = begin();
		while (i != end()) {
			iterator j = i;
			++i;
			erase(j);
		}
	}

	inline iterator
	insert(std::unique_ptr<ElementType> element)
	{
		iterator i = internal_hash_.insert(element.get());
		element.release();
		return i;
	}

	inline void
	erase(ElementType * element) noexcept
	{
		internal_hash_.erase(element);
		delete element;
	}

	inline void
	erase(iterator i) noexcept
	{
		erase(i.ptr());
	}

	inline void
	erase(const KeyType & key) noexcept
	{
		iterator i = find(key);
		if (i != end()) {
			erase(i);
		}
	}

	inline std::unique_ptr<ElementType>
	unlink(iterator i) noexcept
	{
		std::unique_ptr<ElementType> e = i.ptr();
		internal_hash_.erase(i);
		return e;
	}

	inline void
	erase(iterator begin, iterator end) noexcept
	{
		internal_hash_.erase(begin, end);
	}

	inline size_type
	size() const noexcept
	{
		return internal_hash_.size();
	}

	inline bool
	empty() const noexcept
	{
		return internal_hash_.empty();
	}

	iterator
	begin() noexcept
	{
		return internal_hash_.begin();
	}

	iterator
	end() noexcept
	{
		return internal_hash_.end();
	}

	const_iterator
	cbegin() const noexcept
	{
		return internal_hash_.cbegin();
	}

	const_iterator
	cend() const noexcept
	{
		return internal_hash_.cend();
	}

	const_iterator
	begin() const noexcept
	{
		return internal_hash_.begin();
	}

	const_iterator
	end() const noexcept
	{
		return internal_hash_.end();
	}

	inline iterator
	find(const KeyType & key) noexcept
	{
		return internal_hash_.find(key);
	}

	inline const_iterator
	find(const KeyType & key) const noexcept
	{
		return internal_hash_.find(key);
	}

private:
	internal_hash_type internal_hash_;
};

}
}

#endif
