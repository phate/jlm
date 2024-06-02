/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * Copyright 2023 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/mlir/frontend/MlirToJlmConverter.hpp>

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/TopologicalSortUtils.h>

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>
#include <llvm/Support/raw_os_ostream.h>

namespace jlm::mlir
{

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ReadAndConvertMlir(const util::filepath & filePath)
{
  auto config = ::mlir::ParserConfig(Context_.get());
  std::unique_ptr<::mlir::Block> block = std::make_unique<::mlir::Block>();
  auto result = ::mlir::parseSourceFile(filePath.to_str(), block.get(), config);
  if (result.failed())
  {
    JLM_ASSERT("Parsing MLIR input file failed.");
  }
  return ConvertMlir(block);
}

std::unique_ptr<llvm::RvsdgModule>
MlirToJlmConverter::ConvertMlir(std::unique_ptr<::mlir::Block> & block)
{
  auto rvsdgModule = llvm::RvsdgModule::Create(util::filepath(""), std::string(), std::string());
  ConvertBlock(*block, *rvsdgModule->Rvsdg().root());

  return rvsdgModule;
}

std::vector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertRegion(::mlir::Region & region, rvsdg::region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return ConvertBlock(region.front(), rvsdgRegion);
}

std::vector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertBlock(::mlir::Block & block, rvsdg::region & rvsdgRegion)
{
  ::mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<::mlir::Operation *, rvsdg::node *> operations;
  for (auto & mlirOp : block.getOperations())
  {
    std::vector<jlm::rvsdg::output *> inputs;
    for (auto operand : mlirOp.getOperands())
    {
      if (auto * producer = operand.getDefiningOp())
      {
        size_t index = GetOperandIndex(producer, operand);
        inputs.push_back(operations[producer]->output(index));
      }
      else
      {
        // If there is no defining op, the Value is necessarily a Block argument.
        JLM_ASSERT(::mlir::dyn_cast<::mlir::BlockArgument>(operand));
        inputs.push_back(
            rvsdgRegion.argument(operand.cast<::mlir::BlockArgument>().getArgNumber()));
      }
    }

    if (auto * node = ConvertOperation(mlirOp, rvsdgRegion, inputs))
    {
      operations[&mlirOp] = node;
    }
  }

  // The results of the region/block are encoded in the terminator operation
  auto terminator = block.getTerminator();
  std::vector<jlm::rvsdg::output *> results;
  for (auto operand : terminator->getOperands())
  {
    if (auto * producer = operand.getDefiningOp())
    {
      size_t index = GetOperandIndex(producer, operand);
      results.push_back(operations[producer]->output(index));
    }
    else
    {
      // If there is no defining op, the Value is necessarily a Block argument.
      JLM_ASSERT(::mlir::dyn_cast<::mlir::BlockArgument>(operand));
      results.push_back(rvsdgRegion.argument(operand.cast<::mlir::BlockArgument>().getArgNumber()));
    }
  }

  return results;
}

rvsdg::node *
MlirToJlmConverter::ConvertCmpIOp(
    ::mlir::arith::CmpIOp & CompOp,
    std::vector<rvsdg::output *> & inputs,
    size_t nbits)
{
  if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::eq)
  {
    return rvsdg::node_output::node(rvsdg::biteq_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ne)
  {
    return rvsdg::node_output::node(rvsdg::bitne_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sge)
  {
    return rvsdg::node_output::node(rvsdg::bitsge_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sgt)
  {
    return rvsdg::node_output::node(rvsdg::bitsgt_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::sle)
  {
    return rvsdg::node_output::node(rvsdg::bitsle_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::slt)
  {
    return rvsdg::node_output::node(rvsdg::bitslt_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::uge)
  {
    return rvsdg::node_output::node(rvsdg::bituge_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ugt)
  {
    return rvsdg::node_output::node(rvsdg::bitugt_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ule)
  {
    return rvsdg::node_output::node(rvsdg::bitule_op::create(nbits, inputs[0], inputs[1]));
  }
  else if (CompOp.getPredicate() == ::mlir::arith::CmpIPredicate::ult)
  {
    return rvsdg::node_output::node(rvsdg::bitult_op::create(nbits, inputs[0], inputs[1]));
  }
  else
  {
    JLM_UNREACHABLE("frontend : Unknown comparison predicate.");
  }
}

rvsdg::node *
MlirToJlmConverter::ConvertBitBinaryNode(
    ::mlir::Operation & mlirOperation,
    std::vector<rvsdg::output *> & inputs)
{
  if (auto castedOp = ::mlir::dyn_cast<::mlir::LLVM::AddOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitadd_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::AddIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitadd_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::AndIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitand_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ShRUIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitashr_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::MulIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitmul_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::OrIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitor_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::DivSIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitsdiv_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ShLIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitshl_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ShRUIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitshr_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::RemSIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitsmod_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::SubIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitsub_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::DivUIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitudiv_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::RemUIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitumod_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::XOrIOp>(&mlirOperation))
  {
    return rvsdg::node_output::node(rvsdg::bitxor_op::create(
        static_cast<size_t>(castedOp.getType().cast<::mlir::IntegerType>().getWidth()),
        inputs[0],
        inputs[1]));
  }
  else if (auto castedOp = ::mlir::dyn_cast<::mlir::arith::ExtUIOp>(&mlirOperation))
  {
    auto st = dynamic_cast<const jlm::rvsdg::bittype *>(&inputs[0]->type());
    if (!st)
      JLM_ASSERT("frontend : expected bitstring type for ExtUIOp operation.");
    auto op = llvm::zext_op(st->nbits(), castedOp.getType().cast<::mlir::IntegerType>().getWidth());

    return rvsdg::node_output::node(
        rvsdg::simple_node::create_normalized(inputs[0]->region(), op, inputs)[0]);
  }

  return nullptr;
}

rvsdg::node *
MlirToJlmConverter::ConvertOperation(
    ::mlir::Operation & mlirOperation,
    rvsdg::region & rvsdgRegion,
    std::vector<rvsdg::output *> & inputs)
{
  if (::mlir::isa<::mlir::rvsdg::OmegaNode>(&mlirOperation))
  {
    ConvertOmega(mlirOperation, rvsdgRegion);
    // Omega doesn't have a corresponding RVSDG node so we return nullptr
    return nullptr;
  }
  else if (::mlir::isa<::mlir::rvsdg::LambdaNode>(&mlirOperation))
  {
    return ConvertLambda(mlirOperation, rvsdgRegion);
  }
  else if (auto constant = ::mlir::dyn_cast<::mlir::arith::ConstantIntOp>(&mlirOperation))
  {
    auto type = constant.getType();
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return rvsdg::node_output::node(
        rvsdg::create_bitconstant(&rvsdgRegion, integerType.getWidth(), constant.value()));
  }

  // Binary Comparision operations
  else if (auto ComOp = ::mlir::dyn_cast<::mlir::arith::CmpIOp>(&mlirOperation))
  {
    auto type = ComOp.getOperandTypes()[0];
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return ConvertCmpIOp(ComOp, inputs, integerType.getWidth());
  }

  /* #region Arithmetic Integer Operation*/
  //! Here the LLVM dialect where only implemented for AddOp. Other operation should maybe be
  //! imported Need to choose which one of mlir::arith or mlir::LLVM to use for the MLIR
  //! representation
  rvsdg::node * convertedNode = ConvertBitBinaryNode(mlirOperation, inputs);
  // If the operation was converted it means it has been casted to a bit binary operation
  if (convertedNode)
    return convertedNode;
  /* #endregion */

  if (::mlir::isa<::mlir::rvsdg::LambdaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::OmegaResult>(&mlirOperation))
  {
    // This is a terminating operation that doesn't have a corresponding RVSDG node
    return nullptr;
  }
  else
  {
    auto message = util::strfmt(
        "Operation not implemented:",
        mlirOperation.getName().getStringRef().str(),
        "\n");
    JLM_UNREACHABLE(message.c_str());
  }
}

void
MlirToJlmConverter::ConvertOmega(::mlir::Operation & mlirOmega, rvsdg::region & rvsdgRegion)
{
  JLM_ASSERT(mlirOmega.getRegions().size() == 1);
  ConvertRegion(mlirOmega.getRegion(0), rvsdgRegion);
}

jlm::rvsdg::node *
MlirToJlmConverter::ConvertLambda(::mlir::Operation & mlirLambda, rvsdg::region & rvsdgRegion)
{
  // Get the name of the function
  auto functionNameAttribute = mlirLambda.getAttr(::llvm::StringRef("sym_name"));
  JLM_ASSERT(functionNameAttribute != nullptr);
  auto functionName = ::mlir::cast<::mlir::StringAttr>(functionNameAttribute);

  // A lambda node has only the function signature as the result
  JLM_ASSERT(mlirLambda.getNumResults() == 1);
  auto result = mlirLambda.getResult(0).getType();

  JLM_ASSERT(result.getTypeID() == ::mlir::rvsdg::LambdaRefType::getTypeID());

  // Create the RVSDG function signature
  auto lambdaRefType = ::mlir::cast<::mlir::rvsdg::LambdaRefType>(result);
  std::vector<std::shared_ptr<const rvsdg::type>> argumentTypes;
  for (auto argumentType : lambdaRefType.getParameterTypes())
  {
    argumentTypes.push_back(ConvertType(argumentType)->copy());
  }
  std::vector<std::shared_ptr<const rvsdg::type>> resultTypes;
  for (auto returnType : lambdaRefType.getReturnTypes())
  {
    resultTypes.push_back(ConvertType(returnType)->copy());
  }
  llvm::FunctionType functionType(std::move(argumentTypes), std::move(resultTypes));

  // FIXME
  // The linkage should be part of the MLIR attributes so it can be extracted here
  auto rvsdgLambda = llvm::lambda::node::create(
      &rvsdgRegion,
      functionType,
      functionName.getValue().str(),
      llvm::linkage::external_linkage);

  auto lambdaRegion = rvsdgLambda->subregion();
  auto regionResults = ConvertRegion(mlirLambda.getRegion(0), *lambdaRegion);

  rvsdgLambda->finalize(regionResults);

  return rvsdgLambda;
}

std::unique_ptr<rvsdg::type>
MlirToJlmConverter::ConvertType(::mlir::Type & type)
{
  if (auto intType = ::mlir::dyn_cast<::mlir::IntegerType>(type))
  {
    return std::make_unique<rvsdg::bittype>(intType.getWidth());
  }
  else if (::mlir::isa<::mlir::rvsdg::MemStateEdgeType>(type))
  {
    return std::make_unique<llvm::MemoryStateType>();
  }
  else if (::mlir::isa<::mlir::rvsdg::IOStateEdgeType>(type))
  {
    return std::make_unique<llvm::iostatetype>();
  }
  else
  {
    JLM_UNREACHABLE("Type conversion not implemented\n");
  }
}

// TODO
// Consider tracking outputs instead of operations to avoid the need for this function
size_t
MlirToJlmConverter::GetOperandIndex(::mlir::Operation * producer, ::mlir::Value & operand)
{
  JLM_ASSERT(producer->getNumResults() >= 1);
  if (producer->getNumResults() == 1)
  {
    return 0;
  }

  size_t index = 0;
  for (auto tmp : producer->getResults())
  {
    if (tmp == operand)
    {
      break;
    }
    index++;
  }

  return index;
}

} // jlm::mlirrvsdg
