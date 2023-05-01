/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_RVSDGMODULE_HPP
#define JLM_LLVM_IR_RVSDGMODULE_HPP

#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/util/file.hpp>

namespace jlm {

/* impport class */

class impport final : public jive::impport {
public:
	virtual
	~impport();

    impport(
        const jive::valuetype & valueType,
        const std::string & name,
        const jlm::linkage & lnk)
        : jive::impport(PointerType(), name)
        , linkage_(lnk)
        , ValueType_(valueType.copy())
    {}

	impport(const impport & other)
	: jive::impport(other)
	, linkage_(other.linkage_)
    , ValueType_(other.ValueType_->copy())
	{}

	impport(impport && other)
	: jive::impport(other)
	, linkage_(std::move(other.linkage_))
    , ValueType_(std::move(other.ValueType_))
	{}

	impport&
	operator=(const impport&) = delete;

	impport&
	operator=(impport&&) = delete;

	const jlm::linkage &
	linkage() const noexcept
	{
		return linkage_;
	}

    [[nodiscard]] const jive::valuetype &
    GetValueType() const noexcept
    {
        return *AssertedCast<jive::valuetype>(ValueType_.get());
    }

	virtual bool
	operator==(const port&) const noexcept override;

	virtual std::unique_ptr<port>
	copy() const override;

private:
	jlm::linkage linkage_;
    std::unique_ptr<jive::type> ValueType_;
};

static inline bool
is_import(const jive::output * output)
{
	auto graph = output->region()->graph();

	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && argument->region() == graph->root();
}

static inline bool
is_export(const jive::input * input)
{
	auto graph = input->region()->graph();

	auto result = dynamic_cast<const jive::result*>(input);
	return result && result->region() == graph->root();
}

/** \brief RVSDG module class
 *
 */
class RvsdgModule final {
public:
	RvsdgModule(
		jlm::filepath sourceFileName,
		std::string targetTriple,
		std::string dataLayout)
	: DataLayout_(std::move(dataLayout))
	, TargetTriple_(std::move(targetTriple))
	, SourceFileName_(std::move(sourceFileName))
	{}

	RvsdgModule(const RvsdgModule &) = delete;

	RvsdgModule(RvsdgModule &&) = delete;

	RvsdgModule &
	operator=(const RvsdgModule &) = delete;

	RvsdgModule &
	operator=(RvsdgModule &&) = delete;

	jive::graph &
  Rvsdg() noexcept
	{
		return Rvsdg_;
	}

	const jive::graph &
  Rvsdg() const noexcept
	{
		return Rvsdg_;
	}

	const jlm::filepath &
	SourceFileName() const noexcept
	{
		return SourceFileName_;
	}

	const std::string &
	TargetTriple() const noexcept
	{
		return TargetTriple_;
	}

	const std::string &
	DataLayout() const noexcept
	{
		return DataLayout_;
	}

	static std::unique_ptr<RvsdgModule>
	Create(
		const jlm::filepath & sourceFileName,
		const std::string & targetTriple,
		const std::string & dataLayout)
	{
		return std::make_unique<RvsdgModule>(sourceFileName, targetTriple, dataLayout);
	}

private:
	jive::graph Rvsdg_;
	std::string DataLayout_;
	std::string TargetTriple_;
	const jlm::filepath SourceFileName_;
};

}

#endif
