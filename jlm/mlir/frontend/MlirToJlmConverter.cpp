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

::llvm::SmallVector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertRegion(::mlir::Region & region, rvsdg::region & rvsdgRegion)
{
  // MLIR use blocks as the innermost "container"
  // In the RVSDG Dialect a region should contain one and only one block
  JLM_ASSERT(region.getBlocks().size() == 1);
  return ConvertBlock(region.front(), rvsdgRegion);
}

void
MlirToJlmConverter::getConvertedInputs(
    ::mlir::Operation & mlirOp,
    std::unordered_map<::mlir::Operation *, rvsdg::node *> operationsMap,
    rvsdg::region & rvsdgRegion,
    ::llvm::SmallVector<jlm::rvsdg::output *> & inputs)
{
  for (::mlir::Value operand : mlirOp.getOperands())
  {
    if (::mlir::Operation * producer = operand.getDefiningOp())
    {
      JLM_ASSERT(operationsMap.find(producer) != operationsMap.end());
      JLM_ASSERT(::mlir::dyn_cast<::mlir::OpResult>(operand));
      inputs.push_back(
          operationsMap[producer]->output(operand.cast<::mlir::OpResult>().getResultNumber()));
    }
    else
    {
      // If there is no defining op, the Value is necessarily a Block argument.
      JLM_ASSERT(::mlir::dyn_cast<::mlir::BlockArgument>(operand));
      inputs.push_back(rvsdgRegion.argument(operand.cast<::mlir::BlockArgument>().getArgNumber()));
    }
  }
}

::llvm::SmallVector<jlm::rvsdg::output *>
MlirToJlmConverter::ConvertBlock(::mlir::Block & block, rvsdg::region & rvsdgRegion)
{
  ::mlir::sortTopologically(&block);

  // Create an RVSDG node for each MLIR operation and store each pair in a
  // hash map for easy lookup of corresponding RVSDG nodes
  std::unordered_map<::mlir::Operation *, rvsdg::node *> operationsMap;
  for (::mlir::Operation & mlirOp : block.getOperations())
  {

    ::llvm::SmallVector<jlm::rvsdg::output *> inputs;

    getConvertedInputs(mlirOp, operationsMap, rvsdgRegion, inputs);

    if (auto * node = ConvertOperation(mlirOp, rvsdgRegion, inputs))
    {
      operationsMap[&mlirOp] = node;
    }
  }

  // The results of the region/block are encoded in the terminator operation
  ::mlir::Operation * terminator = block.getTerminator();
  ::llvm::SmallVector<jlm::rvsdg::output *> results;

  getConvertedInputs(*terminator, operationsMap, rvsdgRegion, results);

  return results;
}

rvsdg::node *
MlirToJlmConverter::ConvertCmpIOp(
    ::mlir::arith::CmpIOp & CompOp,
    ::llvm::SmallVector<rvsdg::output *> & inputs,
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
    ::llvm::SmallVector<rvsdg::output *> & inputs)
{
  if (inputs.size() != 2)
    return nullptr;
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

    return rvsdg::node_output::node(rvsdg::simple_node::create_normalized(
        inputs[0]->region(),
        op,
        std::vector<rvsdg::output *>(inputs.begin(), inputs.end()))[0]);
  }

  return nullptr;
}

rvsdg::node *
MlirToJlmConverter::ConvertOperation(
    ::mlir::Operation & mlirOperation,
    rvsdg::region & rvsdgRegion,
    ::llvm::SmallVector<rvsdg::output *> & inputs)
{

  /* #region Arithmetic Integer Operation*/
  //! Here the LLVM dialect where only implemented for AddOp. Other operation should maybe be
  //! imported Need to choose which one of mlir::arith or mlir::LLVM to use for the MLIR
  //! representation
  rvsdg::node * convertedNode = ConvertBitBinaryNode(mlirOperation, inputs);
  // If the operation was converted it means it has been casted to a bit binary operation
  if (convertedNode)
    return convertedNode;
  /* #endregion */

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

  // Binary Integer Comparision operations
  else if (auto ComOp = ::mlir::dyn_cast<::mlir::arith::CmpIOp>(&mlirOperation))
  {
    auto type = ComOp.getOperandTypes()[0];
    JLM_ASSERT(type.getTypeID() == ::mlir::IntegerType::getTypeID());
    auto integerType = ::mlir::cast<::mlir::IntegerType>(type);

    return ConvertCmpIOp(ComOp, inputs, integerType.getWidth());
  }

  /* #region Structural nodes */
  else if (auto MlirCtrlConst = ::mlir::dyn_cast<::mlir::rvsdg::ConstantCtrl>(&mlirOperation))
  {
    JLM_ASSERT(::mlir::isa<::mlir::rvsdg::RVSDG_CTRLType>(MlirCtrlConst.getType()));
    return rvsdg::node_output::node(rvsdg::control_constant(
        &rvsdgRegion,
        ::mlir::cast<::mlir::rvsdg::RVSDG_CTRLType>(MlirCtrlConst.getType()).getNumOptions(),
        MlirCtrlConst.getValue()));
  }
  else if (auto MlirGammaNode = ::mlir::dyn_cast<::mlir::rvsdg::GammaNode>(&mlirOperation))
  {
    jlm::rvsdg::gamma_node * RvsdgGammaNode = rvsdg::gamma_node::create(
        inputs[0],                    // predicate
        MlirGammaNode.getNumRegions() // nalternatives
    );

    // Add inputs to the gamma node and to all it's subregions
    for (size_t i = 1; i < inputs.size(); i++)
    {
      RvsdgGammaNode->add_entryvar(inputs[i]);
    }

    ::llvm::SmallVector<::llvm::SmallVector<jlm::rvsdg::output *>> regionResults;
    for (size_t i = 0; i < MlirGammaNode.getNumRegions(); i++)
    {
      regionResults.push_back(
          ConvertRegion(MlirGammaNode.getRegion(i), *RvsdgGammaNode->subregion(i)));
    }

    // Connect the outputs
    //! Here we connect all subregion result to output of the gamma node
    for (size_t exitvarIndex = 0; exitvarIndex < regionResults[0].size(); exitvarIndex++)
    {
      std::vector<rvsdg::output *> exitvars;
      for (size_t regionIndex = 0; regionIndex < MlirGammaNode.getNumRegions(); regionIndex++)
      {
        JLM_ASSERT(regionResults[regionIndex].size() == regionResults[0].size());
        exitvars.push_back(regionResults[regionIndex][exitvarIndex]);
      }
      RvsdgGammaNode->add_exitvar(exitvars);
    }

    return RvsdgGammaNode;
  }
  else if (auto MlirMatch = ::mlir::dyn_cast<::mlir::rvsdg::Match>(&mlirOperation))
  {
    std::unordered_map<uint64_t, uint64_t> mapping;
    uint64_t defaultAlternative = 0;
    for (auto & attr : MlirMatch.getMapping())
    {
      JLM_ASSERT(attr.isa<::mlir::rvsdg::MatchRuleAttr>());
      auto matchRuleAttr = attr.cast<::mlir::rvsdg::MatchRuleAttr>();
      if (matchRuleAttr.isDefault())
      {
        defaultAlternative = matchRuleAttr.getIndex();
        continue;
      }
      // In our Mlir implementation, an index is associated with a single value
      mapping[matchRuleAttr.getValues().front()] = matchRuleAttr.getIndex();
    }

    return rvsdg::node_output::node(rvsdg::match_op::Create(
        *(inputs[0]),                 // predicate
        mapping,                      // mapping
        defaultAlternative,           // defaultAlternative
        MlirMatch.getMapping().size() // numAlternatives
        ));
  }
  /* #endregion */

  else if (
      ::mlir::isa<::mlir::rvsdg::LambdaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::OmegaResult>(&mlirOperation)
      || ::mlir::isa<::mlir::rvsdg::GammaResult>(&mlirOperation))
  {
    // This is a terminating operation that doesn't have a corresponding RVSDG node
    return nullptr;
  }
  else
  {
    auto message = util::strfmt(
        "Operation not implemented: ",
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

  rvsdgLambda->finalize(std::vector<rvsdg::output *>(regionResults.begin(), regionResults.end()));

  return rvsdgLambda;
}

std::unique_ptr<rvsdg::type>
MlirToJlmConverter::ConvertType(::mlir::Type & type)
{
  if (auto ctrlType = ::mlir::dyn_cast<::mlir::rvsdg::RVSDG_CTRLType>(type))
  {
    return std::make_unique<jlm::rvsdg::ctltype>(ctrlType.getNumOptions());
  }
  else if (auto intType = ::mlir::dyn_cast<::mlir::IntegerType>(type))
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

} // jlm::mlirrvsdg
