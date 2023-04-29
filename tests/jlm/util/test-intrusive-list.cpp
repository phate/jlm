/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <assert.h>

#include <jlm/util/intrusive-list.hpp>

namespace {

struct my_item {
	inline my_item() : p(nullptr) {}
	inline my_item(int * ptr) : p(ptr) {}
	inline ~my_item() { if (p) { *p = 0; } }

	int * p;
	jive::detail::intrusive_list_anchor<my_item> anchor;
	typedef jive::detail::intrusive_list_accessor<my_item, &my_item::anchor> accessor;
};

typedef jive::detail::intrusive_list<my_item, my_item::accessor> my_list;
typedef jive::detail::owner_intrusive_list<my_item, my_item::accessor> my_owner_list;

static void test_simple_list(void)
{
	my_list l;

	assert(l.empty());
	my_item i1, i2, i3;

	l.push_back(&i2);
	assert(l.begin().ptr() == &i2);
	assert(std::next(l.begin()) == l.end());
	assert(std::prev(l.end()).ptr() == &i2);

	l.insert(l.begin(), &i1);
	assert(l.begin().ptr() == &i1);
	assert(std::next(l.begin()).ptr() == &i2);
	assert(std::next(l.begin(), 2) == l.end());
	assert(std::prev(l.end()).ptr() == &i2);

	l.insert(l.end(), &i3);
	assert(l.begin().ptr() == &i1);
	assert(std::next(l.begin()).ptr() == &i2);
	assert(std::next(l.begin(), 2).ptr() == &i3);
	assert(std::next(l.begin(), 3) == l.end());
	assert(std::prev(l.end()).ptr() == &i3);
	assert(std::prev(l.end(), 2).ptr() == &i2);
	assert(std::prev(l.end(), 3).ptr() == &i1);

	l.erase(&i2);
	assert(l.begin().ptr() == &i1);
	assert(std::next(l.begin()).ptr() == &i3);
	assert(std::next(l.begin(), 2) == l.end());
	assert(std::prev(l.end()).ptr() == &i3);
	assert(std::prev(l.end(), 2).ptr() == &i1);

	my_list l2;
	l2.splice(l2.begin(), l);
	assert(l.empty());
	assert(l2.size() == 2);
}

static void test_owner_list(void)
{
	int v1 = 1;
	int v2 = 2;
	int v3 = 3;

	{
		my_owner_list l;

		assert(l.empty());

		l.push_back(std::unique_ptr<my_item>(new my_item(&v2)));
		assert(l.begin()->p == &v2);
		assert(std::next(l.begin()) == l.end());
		assert(std::prev(l.end())->p == &v2);

		l.insert(l.begin(), std::unique_ptr<my_item>(new my_item(&v1)));
		assert(l.begin()->p == &v1);
		assert(std::next(l.begin())->p == &v2);
		assert(std::next(l.begin(), 2) == l.end());
		assert(std::prev(l.end())->p == &v2);

		l.insert(l.end(), std::unique_ptr<my_item>(new my_item(&v3)));
		assert(l.begin()->p == &v1);
		assert(std::next(l.begin())->p == &v2);
		assert(std::next(l.begin(), 2)->p == &v3);
		assert(std::next(l.begin(), 3) == l.end());
		assert(std::prev(l.end())->p == &v3);
		assert(std::prev(l.end(), 2)->p == &v2);
		assert(std::prev(l.end(), 3)->p == &v1);

		l.erase(std::next(l.begin()));
		assert(v1 == 1);
		assert(v2 == 0); // destructor should have been called
		assert(v3 == 3);

		assert(l.begin()->p == &v1);
		assert(std::next(l.begin())->p == &v3);
		assert(std::next(l.begin(), 2) == l.end());
		assert(std::prev(l.end())->p == &v3);
		assert(std::prev(l.end(), 2)->p == &v1);

		std::unique_ptr<my_item> i = l.unlink(l.begin());
		assert(v1 == 1); // destructor not invoked, transferred ownership
		assert(v2 == 0);
		assert(v3 == 3);

		i.reset();
		assert(v1 == 0); // destructor called now

		assert(l.begin()->p == &v3);
		assert(std::next(l.begin()) == l.end());
		assert(std::prev(l.end())->p == &v3);

		my_owner_list l2;
		l2.splice(l2.begin(), l);
		assert(l.size() == 0);
		assert(l2.size() == 1);
	}
	assert(v3 == 0);
}

static int test_main(void)
{
	test_simple_list();
	test_owner_list();

	return 0;
}

}

JLM_UNIT_TEST_REGISTER("jlm/util/test-intrusive-list", test_main)
