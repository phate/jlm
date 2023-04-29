/*
 * Copyright 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_INTRUSIVE_LIST_HPP
#define JLM_UTIL_INTRUSIVE_LIST_HPP

#include <iterator>
#include <memory>

/*
 * Implementation of intrusive_list data structure
 *
 * An intrusive_list is a data structure linking multiple objects into a
 * list. The difference to std::list is that the linkage pointers are part
 * of the objects linked into the data structure itself. As a result, the
 * intrusive_list does not manage memory for the objects it contains, but
 * links objects allocated elsewhere. Any object can be member of an arbitrary
 * number of such intrusive_list collections.
 *
 * Usage:
 *
 * For a class declared as the following:
 *
 * class X {
 * private:
 *   int num;
 *   jive::detail::intrusive_list_anchor<X> num_list_anchor;
 * public:
 *   typedef jive::detail::intrusive_list_accessor<
 *     X,                  // element type
 *     &X::num_list_anchor // anchor member
 *   > num_list_accessor;
 * };
 *
 * an intrusive_list data structure can be declared in the following way:
 *
 * typedef jive::detail::intrusive_list<
 *   X,
 *   X::num_list_accessor
 * > num_list;
 *
 * It is possible to implement a custom accessor instead of using the
 * template-generated one. In this case the get_prev, get_next, set_prev and
 * set_next members must be implemented appropriately.
 *
 * An object h of num_list class then supports STL-style operations
 * - num_list::iterator, num_list::const_iterator for iteration
 * - h.begin(), h.end() and const-qualified variants
 * - h.insert(element) h.push_front(element), h.push_back(element) links an
 *   object into the data structure
 * - h.erase(element) or h.erase(iterator) unlinks an object from the data
 *   structure
 * - h.size() and h.empty() testing
 *
 * The implementation requires the following guarantees:
 * - the accessor to key and anchor are noexcept
 *
 * The implementation provides the following guarantees
 * - insert is O(1) amortized
 * - erase, empty, are O(1) and noexcept
 * - end is O(1)
 * - begin is O(n)
 * - inserting a new object does not invalidate iterators or change order
 *   of previously inserted objects
 * - erasing an object does not invalidate iterators except those pointing
 *   to the object removed and does not change order of previously inserted
 *   objects
 *
 * An additional template "owner_intrusive_list" implements the same
 * interface, but in addition assumes "ownership" of the objects it contains.
 * This means that upon destruction of the container, the elements will
 * be deleted as well. This variant differs in the following ways:
 *
 * - insert, push_back, push_front expect to be given a std::unique_ptr
 *   to the object to be inserted
 * - a new method "unlink" removes an object from the list and returns
 *   a std::unique_ptr to the element
 * - "erase" will actually cause an element to be deleted; likewise,
 *   all elements will be deleted on destruction of the owner_intrusive_list
 */

namespace jive {
namespace detail {

template<
	typename ElementType,
	typename Accessor>
class intrusive_list {
public:
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
	class const_iterator;
	class iterator {
	public:
		typedef ElementType value_type;
		typedef ElementType * pointer;
		typedef ElementType & reference;
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef size_t size_type;
		typedef ssize_t difference_type;

		constexpr iterator() noexcept
			: list_(nullptr)
			, element_(nullptr)
		{
		}

		constexpr iterator(
			const intrusive_list * list,
			ElementType * object)
			: list_(list)
			, element_(object)
		{
		}
	
		inline const iterator &
		operator++(void) noexcept
		{
			element_ = list_->accessor_.get_next(element_);
			return *this;
		}

		inline iterator
		operator++(int) noexcept
		{
			iterator i = *this;
			++*this;
			return i;
		}

		inline const iterator &
		operator--(void) noexcept
		{
			if (element_) {
				element_ = list_->accessor_.get_prev(element_);
			} else {
				element_ = list_->last_;
			}
			return *this;
		}

		inline iterator
		operator--(int) noexcept
		{
			iterator i = *this;
			--*this;
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
		const intrusive_list * list_;
		ElementType * element_;
		friend class const_iterator;
	};

	class const_iterator {
	public:
		typedef const ElementType value_type;
		typedef const ElementType * pointer;
		typedef const ElementType & reference;
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef size_t size_type;
		typedef ssize_t difference_type;

		constexpr const_iterator(const const_iterator & other) noexcept = default;
		constexpr const_iterator(const iterator & other) noexcept
			: list_(other.list_)
			, element_(other.element_)
		{
		}

		constexpr const_iterator() noexcept
			: list_(nullptr)
			, element_(nullptr)
		{
		}

		constexpr const_iterator(
			const intrusive_list * list,
			const ElementType * object)
			: list_(list)
			, element_(object)
		{
		}
	
		inline const const_iterator &
		operator++(void) noexcept
		{
			element_ = list_->accessor_.get_next(element_);
			return *this;
		}

		inline const_iterator
		operator++(int) noexcept
		{
			iterator i = *this;
			++*this;
			return i;
		}

		inline const const_iterator &
		operator--(void) noexcept
		{
			if (element_) {
				element_ = list_->accessor_.get_prev(element_);
			} else {
				element_ = list_->last_;
			}
			return *this;
		}

		inline const_iterator
		operator--(int) noexcept
		{
			iterator i = *this;
			--*this;
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
		const intrusive_list * list_;
		const ElementType * element_;
	};

	typedef ElementType value_type;
	typedef size_t size_type;

	inline constexpr
	intrusive_list() noexcept
		: first_(nullptr)
		, last_(nullptr)
	{
	}

	intrusive_list(const intrusive_list & other) = delete;

	void operator=(const intrusive_list & other) = delete;

	intrusive_list(intrusive_list && other) noexcept
		: intrusive_list()
	{
		swap(other);
	}

	void clear() noexcept
	{
		first_ = nullptr;
		last_ = nullptr;
	}

	void swap(intrusive_list & other) noexcept
	{
		std::swap(first_, other.first_);
		std::swap(last_, other.last_);
	}

	inline void
	push_back(ElementType * element) noexcept
	{
		accessor_.set_prev(element, last_);
		accessor_.set_next(element, nullptr);
		if (last_) {
			accessor_.set_next(last_, element);
		} else {
			first_ = element;
		}
		last_ = element;
	}

	inline void
	push_front(ElementType * element) noexcept
	{
		accessor_.set_prev(element, nullptr);
		accessor_.set_next(element, first_);
		if (first_) {
			accessor_.set_prev(first_, element);
		} else {
			last_ = element;
		}
		first_ = element;
	}

	inline iterator
	insert(iterator i, ElementType * element) noexcept
	{
		ElementType * next = i.ptr();
		ElementType * prev = next ? accessor_.get_prev(next) : last_;
		accessor_.set_prev(element, prev);
		accessor_.set_next(element, next);
		if (prev) {
			accessor_.set_next(prev, element);
		} else {
			first_ = element;
		}
		if (next) {
			accessor_.set_prev(next, element);
		} else {
			last_ = element;
		}
		return iterator(this, element);
	}

	inline void
	erase(ElementType * element) noexcept
	{
		ElementType * prev = accessor_.get_prev(element);
		ElementType * next = accessor_.get_next(element);
		if (prev) {
			accessor_.set_next(prev, next);
		} else {
			first_ = next;
		}
		if (next) {
			accessor_.set_prev(next, prev);
		} else {
			last_ = prev;
		}
	}

	inline iterator
	erase(iterator i) noexcept
	{
		auto element = i.ptr();
		++i;
		erase(element);
		return i;
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

	inline void
	splice(iterator position, intrusive_list & other) noexcept
	{
		splice(position, other, other.begin(), other.end());
	}

	inline void
	splice(iterator position, intrusive_list & other, iterator i) noexcept
	{
		iterator j = i;
		++j;
		splice(position, other, i, j);
	}

	inline void
	splice(iterator position, intrusive_list & other, iterator begin, iterator end) noexcept
	{
		if (begin == end) {
			return;
		}

		ElementType * first = begin.ptr();
		ElementType * before = accessor_.get_prev(first);
		ElementType * after = end.ptr();
		ElementType * last = after ? accessor_.get_prev(after) : other.last_;

		/* unlink from source */
		if (before) {
			accessor_.set_next(before, after);
		} else {
			other.first_ = after;
		}
		if (after) {
			accessor_.set_prev(after, before);
		} else {
			other.last_ = before;
		}

		/* link to destination */
		ElementType * dst_next = position.ptr();
		ElementType * dst_prev = dst_next ? accessor_.get_prev(dst_next) : last_;
		accessor_.set_prev(first, dst_prev);
		accessor_.set_next(last, dst_next);
		if (dst_prev) {
			accessor_.set_next(dst_prev, first);
		} else {
			first_ = first;
		}
		if (dst_next) {
			accessor_.set_prev(dst_next, last);
		} else {
			last_ = last;
		}
	}

	inline size_type
	size() const noexcept
	{
		size_type count = 0;
		for (const_iterator i = begin(); i != end(); ++i) {
			++count;
		}
		return count;
	}

	inline bool
	empty() const noexcept
	{
		return first_ == nullptr;
	}

	iterator
	begin() noexcept
	{
		return iterator(this, first_);
	}

	iterator
	end() noexcept
	{
		return iterator(this, nullptr);
	}

	const_iterator
	cbegin() const noexcept
	{
		return const_iterator(this, first_);
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

	/* create iterator for element */
	iterator
	make_element_iterator(ElementType * element) const noexcept
	{
		return iterator(this, element);
	}

	/* create iterator for element */
	const_iterator
	make_element_iterator(const ElementType * element) const noexcept
	{
		return const_iterator(this, element);
	}

	ElementType *
	first() const noexcept
	{
		return first_;
	}

	ElementType *
	last() const noexcept
	{
		return last_;
	}

private:
	ElementType * first_;
	ElementType * last_;

	Accessor accessor_;
};

template<typename ElementType>
class intrusive_list_anchor {
public:
	ElementType * prev;
	ElementType * next;
};

template<
	typename ElementType,
	intrusive_list_anchor<ElementType> ElementType::*anchor_member>
class intrusive_list_accessor {
public:
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
	typename ElementType,
	typename Accessor>
class owner_intrusive_list {
private:
	typedef intrusive_list<ElementType, Accessor> internal_list_type;
public:
	static_assert(
		noexcept(std::declval<ElementType&>().~ElementType()),
		"Require noexcept destructor for ElementType");
	typedef typename internal_list_type::const_iterator const_iterator;
	typedef typename internal_list_type::iterator iterator;
	typedef typename internal_list_type::value_type value_type;
	typedef typename internal_list_type::size_type size_type;

	~owner_intrusive_list() noexcept
	{
		clear();
	}

	owner_intrusive_list() {}

	owner_intrusive_list(const owner_intrusive_list & other) = delete;

	void operator=(const owner_intrusive_list & other) = delete;

	owner_intrusive_list(owner_intrusive_list && other) noexcept
	{
		swap(other);
	}

	void clear() noexcept
	{
		while (!empty()) {
			ElementType * element = begin().ptr();
			internal_list_.erase(element);
			delete element;
		}
	}

	void swap(owner_intrusive_list & other) noexcept
	{
		internal_list_.swap(other.internal_list_);
	}

	inline void
	push_back(std::unique_ptr<ElementType> element) noexcept
	{
		internal_list_.push_back(element.release());
	}

	inline void
	push_front(std::unique_ptr<ElementType> element) noexcept
	{
		internal_list_.push_front(element.release());
	}

	inline iterator
	insert(iterator i, std::unique_ptr<ElementType> element) noexcept
	{
		return internal_list_.insert(i, element.release());
	}

	inline std::unique_ptr<ElementType>
	unlink(iterator i) noexcept
	{
		ElementType * element = i.ptr();
		internal_list_.erase(i);
		return std::unique_ptr<ElementType>(element);
	}

	inline void
	erase(ElementType * element) noexcept
	{
		erase(make_element_iterator(element));
	}

	inline void
	erase(iterator i) noexcept
	{
		unlink(i);
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

	inline void
	splice(iterator position, owner_intrusive_list & other) noexcept
	{
		splice(position, other, other.begin(), other.end());
	}

	inline void
	splice(iterator position, owner_intrusive_list & other, iterator i) noexcept
	{
		iterator j = i;
		++j;
		splice(position, other, i, j);
	}

	inline void
	splice(iterator position, owner_intrusive_list & other, iterator begin, iterator end) noexcept
	{
		internal_list_.splice(position, other.internal_list_, begin, end);
	}

	inline size_type
	size() const noexcept
	{
		return internal_list_.size();
	}

	inline bool
	empty() const noexcept
	{
		return internal_list_.empty();
	}

	iterator
	begin() noexcept
	{
		return internal_list_.begin();
	}

	iterator
	end() noexcept
	{
		return internal_list_.end();
	}

	const_iterator
	cbegin() const noexcept
	{
		return internal_list_.cbegin();
	}

	const_iterator
	cend() const noexcept
	{
		return internal_list_.cend();
	}

	const_iterator
	begin() const noexcept
	{
		return internal_list_.begin();
	}

	const_iterator
	end() const noexcept
	{
		return internal_list_.end();
	}

	/* create iterator for element */
	iterator
	make_element_iterator(ElementType * element) const noexcept
	{
		return iterator(&internal_list_, element);
	}

	/* create iterator for element */
	const_iterator
	make_element_iterator(const ElementType * element) const noexcept
	{
		return const_iterator(&internal_list_, element);
	}

private:
	internal_list_type internal_list_;
};

}
}

#endif
