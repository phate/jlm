/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ANNOTATION_HPP
#define JLM_IR_ANNOTATION_HPP

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

class aggnode;
class variable;

typedef std::unordered_set<const jlm::variable*> variableset;

class demandset {
public:
	virtual
	~demandset();

	inline
	demandset(const variableset & b)
	: bottom(b)
	{}

	static inline std::unique_ptr<demandset>
	create(const variableset & bottom)
	{
		return std::make_unique<demandset>(bottom);
	}

	variableset top;
	variableset bottom;
};

class branchset final : public demandset {
public:
	virtual
	~branchset();

	inline
	branchset(const variableset & bottom)
	: demandset(bottom)
	{}

	static inline std::unique_ptr<branchset>
	create(const variableset & bottom)
	{
		return std::make_unique<branchset>(bottom);
	}

	variableset cases_top;
};

typedef std::unordered_map<const aggnode*, std::unique_ptr<demandset>> demandmap;

demandmap
annotate(const jlm::aggnode & root);

}

#endif
