/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/disjointset.hpp>

#include <assert.h>
#include <iostream>

static void
print(const jlm::disjointset<int>::set & set)
{
	std::cout << "{";
	for (auto & member : set)
		std::cout << member << " ";
	std::cout << "}";
}

static void
print(const jlm::disjointset<int> & djset)
{
	for (auto & set : djset)
		print(set);

	std::cout << "\n";
}

static int
test()
{
	using namespace jlm;

	disjointset<int> djset({1, 2, 3, 4 , 5});
	print(djset);
	assert(djset.nvalues() == 5 && djset.nsets() == 5);

	auto s1 = djset.find_or_insert(6);
	auto s2 = djset.find(6);
	print(djset);
	assert(s1 == s2);
	assert(djset.nvalues() == 6 && djset.nsets() == 6);

	auto root = djset.merge(1, 2);
	print(djset);
	assert(root->is_root() && root->nmembers() == 2);
	assert(djset.nvalues() == 6 && djset.nsets() == 5);

	root = djset.merge(6, 5);
	print(djset);
	assert(root->is_root() && root->nmembers() == 2);
	assert(djset.nvalues() == 6 && djset.nsets() == 4);

	root = djset.merge(1, 6);
	print(djset);
	assert(root->is_root() && root->nmembers() == 4);
	assert(djset.nvalues() == 6 && djset.nsets() == 3);
	assert(djset.find(2) == djset.find(5));

	djset.clear();
	print(djset);
	assert(djset.nvalues() == 0 && djset.nsets() == 0);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/test-disjointset", test)
