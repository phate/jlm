/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/jlm2llvm/context.hpp>
#include <jlm/llvm/backend/jlm2llvm/type.hpp>

#include <llvm/IR/Module.h>

#include <typeindex>
#include <unordered_map>

namespace jlm {
namespace jlm2llvm {

static llvm::Type *
convert(const jive::bittype & type, context & ctx)
{
	return llvm::Type::getIntNTy(ctx.llvm_module().getContext(), type.nbits());
}

static llvm::Type *
convert(const FunctionType & functionType, context & ctx)
{
	auto & lctx = ctx.llvm_module().getContext();

	using namespace llvm;

	bool isvararg = false;
	std::vector<Type*> argumentTypes;
  for (auto & argumentType : functionType.Arguments()) {
		if (jive::is<varargtype>(argumentType)) {
			isvararg = true;
			continue;
		}

		if (jive::is<iostatetype>(argumentType))
			continue;
		if (jive::is<MemoryStateType>(argumentType))
			continue;
		if (jive::is<loopstatetype>(argumentType))
			continue;

		argumentTypes.push_back(convert_type(argumentType, ctx));
	}

	/*
		The return type can either be (valuetype, statetype, statetype, ...) if the function has
		a return value, or (statetype, statetype, ...) if the function returns void.
	*/
	auto resultType = Type::getVoidTy(lctx);
	if (functionType.NumResults() > 0 && jive::is<jive::valuetype>(functionType.ResultType(0)))
		resultType = convert_type(functionType.ResultType(0), ctx);

	return llvm::FunctionType::get(resultType, argumentTypes, isvararg);
}

static llvm::Type *
convert(const PointerType&, context & ctx)
{
  return llvm::PointerType::get(ctx.llvm_module().getContext(), 0);
}

static llvm::Type *
convert(const arraytype & type, context & ctx)
{
	return llvm::ArrayType::get(convert_type(type.element_type(), ctx), type.nelements());
}

static llvm::Type *
convert(const jive::ctltype & type, context & ctx)
{
	if (type.nalternatives() == 2)
		return llvm::Type::getInt1Ty(ctx.llvm_module().getContext());

	return llvm::Type::getInt32Ty(ctx.llvm_module().getContext());
}

static llvm::Type *
convert(const fptype & type, context & ctx)
{
	static std::unordered_map<
		fpsize
	, llvm::Type*(*)(llvm::LLVMContext&)
	> map({
	  {fpsize::half,    llvm::Type::getHalfTy}
	, {fpsize::flt,     llvm::Type::getFloatTy}
	, {fpsize::dbl,     llvm::Type::getDoubleTy}
	, {fpsize::x86fp80, llvm::Type::getX86_FP80Ty}
	});

	JLM_ASSERT(map.find(type.size()) != map.end());
	return map[type.size()](ctx.llvm_module().getContext());
}

static llvm::Type *
convert(const StructType & type, context & ctx)
{
	auto & decl = type.GetDeclaration();

	if (auto st = ctx.structtype(&decl))
		return st;

	auto st = llvm::StructType::create(ctx.llvm_module().getContext());
	ctx.add_structtype(&decl, st);

	std::vector<llvm::Type*> elements;
	for (size_t n = 0; n < decl.nelements(); n++)
		elements.push_back(convert_type(decl.element(n), ctx));

	if (type.HasName())
		st->setName(type.GetName());
	st->setBody(elements, type.IsPacked());

	return st;
}

static llvm::Type *
convert(const fixedvectortype & type, context & ctx)
{
    return llvm::VectorType::get(convert_type(type.type(), ctx), type.size(), false);
}

static llvm::Type *
convert(const scalablevectortype & type, context & ctx)
{
    return llvm::VectorType::get(convert_type(type.type(), ctx), type.size(), true);
}

template<class T> static llvm::Type *
convert(
	const jive::type & type,
	context & ctx)
{
	JLM_ASSERT(jive::is<T>(type));
	return convert(*static_cast<const T*>(&type), ctx);
}

llvm::Type *
convert_type(const jive::type & type, context & ctx)
{
  static std::unordered_map<
    std::type_index,
    std::function<llvm::Type*(const jive::type&, context&)>
  > map({
          {typeid(jive::bittype),      convert<jive::bittype>},
          {typeid(FunctionType),       convert<FunctionType>},
          {typeid(PointerType),        convert<PointerType>},
          {typeid(arraytype),          convert<arraytype>},
          {typeid(jive::ctltype),      convert<jive::ctltype>},
          {typeid(fptype),             convert<fptype>},
          {typeid(StructType),         convert<StructType>},
          {typeid(fixedvectortype),    convert<fixedvectortype>},
          {typeid(scalablevectortype), convert<scalablevectortype>}
        });

  JLM_ASSERT(map.find(typeid(type)) != map.end());
  return map[typeid(type)](type, ctx);
}

}}
