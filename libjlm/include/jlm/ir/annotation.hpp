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
	demandset()
	{}

	static inline std::unique_ptr<demandset>
	create()
	{
		return std::make_unique<demandset>();
	}

	variableset top;
	variableset bottom;

	variableset reads;
	variableset writes;
};

typedef std::unordered_map<const aggnode*, std::unique_ptr<demandset>> demandmap;

demandmap
annotate(const jlm::aggnode & root);

}

#endif
