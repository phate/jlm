/*
 * Copyright 2013 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_UTIL_HEAP_HPP
#define JIVE_UTIL_HEAP_HPP

#define DEFINE_HEAP(heap_type, obj_type, comparator) \
 \
/* take heap of size 'heap_size', add extend to heap */ \
/* of size 'heap_size + 1' with given obj inserted */ \
static inline void \
heap_type##_push(obj_type * heap, size_t heap_size, obj_type obj) \
{ \
	size_t index = heap_size; \
	while (index) { \
		size_t parent = (index - 1) >> 1; \
		if (!comparator(heap[parent], obj)) \
			break; \
		heap[index] = heap[parent]; \
		index = parent; \
	} \
	heap[index] = obj; \
}; \
 \
/* take heap of size 'heap_size', drop top element and */ \
/* reduce to heap of size 'heap_size - 1' */ \
static inline void \
heap_type##_pop(obj_type * heap, size_t heap_size) \
{ \
	heap[0] = heap[heap_size - 1]; \
	size_t index = 0; \
	for (;;) { \
		size_t current = index; \
		size_t left = (index << 1) + 1; \
		size_t right = left + 1; \
		if (left < heap_size) { \
			if (!comparator(heap[left], heap[current])) \
				current = left; \
			if (right < heap_size && !comparator(heap[right], heap[current])) \
				current = right; \
		} \
		if (current == index) \
			break; \
		obj_type tmp = heap[index]; \
		heap[index] = heap[current]; \
		heap[current] = tmp; \
		index = current; \
	} \
}

#endif
