/*
 * Copyright 2021 Magnus Sjalander <work@sjalander.com> and
 * David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "jlm/backend/hls/rhls2firrtl/mlirgen.hpp"

#ifdef CIRCT

// Handles nodes with 2 inputs and 1 output
circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenSimpleNode(const jive::simple_node *node) {
	// Only handles nodes with a single output
	if (node->noutputs() != 1) {
		throw std::logic_error(node->operation().debug_string() + " has more than 1 output");
	}

	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	// Get the body of the module such that we can add contents to the module
	auto body = module.getBody();

	llvm::SmallVector<mlir::Value> inBundles;

	// Get input signals
	for (size_t i=0; i<node->ninputs(); i++) {
		// Get the input bundle
		auto bundle = GetInPort(module, i);
		// Get the data signal from the bundle
		GetSubfield(body, bundle, "data");
		inBundles.push_back(bundle);
	}

	// Get the output bundle
	auto outBundle = GetOutPort(module, 0);
	// Get the data signal from the bundle
	auto outData = GetSubfield(body, outBundle, "data");

	if (dynamic_cast<const jive::bitadd_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddAddOp(body, input0, input1);
		// Connect the op to the output data
		PartialConnect(body, outData, op);
	} else if (dynamic_cast<const jive::bitsub_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddSubOp(body, input0, input1);
		// Connect the op to the output data
		PartialConnect(body, outData, op);
	} else if (dynamic_cast<const jive::bitand_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddAndOp(body, input0, input1);
		// Connect the op to the output data
		PartialConnect(body, outData, op);
	} else if (dynamic_cast<const jive::bitxor_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddXorOp(body, input0, input1);
		// Connect the op to the output data
		PartialConnect(body, outData, op);
	} else if (dynamic_cast<const jive::bitor_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddOrOp(body, input0, input1);
		// Connect the op to the output data
		PartialConnect(body, outData, op);
	} else if (dynamic_cast<const jive::bitmul_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddMulOp(body, input0, input1);
		// Connect the op to the output data
		PartialConnect(body, outData, op);
	} else if (dynamic_cast<const jive::bitsdiv_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sIntOp0 = AddAsSIntOp(body, input0);
		auto sIntOp1 = AddAsSIntOp(body, input1);
		auto divOp = AddDivOp(body, sIntOp0, sIntOp1);
		auto uIntOp = AddAsUIntOp(body, divOp);
		// Connect the op to the output data
		PartialConnect(body, outData, uIntOp);
	} else if (dynamic_cast<const jive::bitshr_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddDShrOp(body, input0, input1);
		// Connect the op to the output data
		Connect(body, outData, op);
	} else if (dynamic_cast<const jive::bitashr_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sIntOp0 = AddAsSIntOp(body, input0);
		auto shrOp = AddDShrOp(body, sIntOp0, input1);
		auto uIntOp = AddAsUIntOp(body, shrOp);
		// Connect the op to the output data
		Connect(body, outData, uIntOp);
	} else if (dynamic_cast<const jive::bitshl_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto bitsOp = AddBitsOp(body, input1, 7, 0);
		auto op = AddDShlOp(body, input0, bitsOp);
		int outSize = JlmSize(&node->output(0)->type());
		auto slice = AddBitsOp(body, op, outSize - 1, 0);
		// Connect the op to the output data
		Connect(body, outData, slice);
	} else if (dynamic_cast<const jive::bitsmod_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sIntOp0 = AddAsSIntOp(body, input0);
		auto sIntOp1 = AddAsSIntOp(body, input1);
		auto remOp = AddRemOp(body, sIntOp0, sIntOp1);
		auto uIntOp = AddAsUIntOp(body, remOp);
		Connect(body, outData, uIntOp);
	} else if (dynamic_cast<const jive::biteq_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddEqOp(body, input0, input1);
		// Connect the op to the output data
		Connect(body, outData, op);
	}  else if (dynamic_cast<const jive::bitne_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddNeqOp(body, input0, input1);
		// Connect the op to the output data
		Connect(body, outData, op);
	} else if (dynamic_cast<const jive::bitsgt_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sIntOp0 = AddAsSIntOp(body, input0);
		auto sIntOp1 = AddAsSIntOp(body, input1);
		auto op = AddGtOp(body, sIntOp0, sIntOp1);
		// Connect the op to the output data
		Connect(body, outData, op);
	}  else if (dynamic_cast<const jive::bitult_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto op = AddLtOp(body, input0, input1);
		// Connect the op to the output data
		Connect(body, outData, op);
	} else if (dynamic_cast<const jive::bitsge_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sIntOp0 = AddAsSIntOp(body, input0);
		auto sIntOp1 = AddAsSIntOp(body, input1);
		auto op = AddGeqOp(body, sIntOp0, sIntOp1);
		// Connect the op to the output data
		Connect(body, outData, op);
	} else if (dynamic_cast<const jive::bitsle_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sIntOp0 = AddAsSIntOp(body, input0);
		auto sIntOp1 = AddAsSIntOp(body, input1);
		auto op = AddLeqOp(body, sIntOp0, sIntOp1);
		// Connect the op to the output data
		Connect(body, outData, op);
	} else if (dynamic_cast<const jlm::zext_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		Connect(body, outData, input0);
	} else if (dynamic_cast<const jlm::trunc_op *>(&(node->operation()))) {
		auto inData = GetSubfield(body, inBundles[0], "data");
		int outSize = JlmSize(&node->output(0)->type());
		Connect(body, outData, AddBitsOp(body, inData, outSize-1, 0));
	} else if (auto op = dynamic_cast<const sext_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto sintOp = AddAsSIntOp(body, input0);
		auto padOp = AddPadOp(body, sintOp, op->ndstbits());
		auto uintOp = AddAsUIntOp(body, padOp);
		Connect(body, outData, uintOp);
	} else if (auto op = dynamic_cast<const jive::bitconstant_op *>(&(node->operation()))) {
		auto value = op->value();
		auto size = value.nbits();
		// Create a constant of UInt<size>(value) and connect to output data
		auto constant = GetConstant(body, size, value.to_uint());
		Connect(body, outData, constant);
	} else if (auto op = dynamic_cast<const jive::ctlconstant_op *>(&(node->operation()))) {
		auto value = op->value().alternative();
		auto size = ceil(log2(op->value().nalternatives()));
		auto constant = GetConstant(body, size, value);
		Connect(body, outData, constant);
	} else if (dynamic_cast<const jive::bitslt_op *>(&(node->operation()))) {
		auto input0 = GetSubfield(body, inBundles[0], "data");
		auto input1 = GetSubfield(body, inBundles[1], "data");
		auto sInt0 = AddAsSIntOp(body, input0);
		auto sInt1 = AddAsSIntOp(body, input1);
		auto op = AddLtOp(body, sInt0, sInt1);
		Connect(body, outData, op);
	} else if (auto op = dynamic_cast<const jive::match_op *>(&(node->operation()))) {
		auto inData = GetSubfield(body, inBundles[0], "data");
		auto outData = GetSubfield(body, outBundle, "data");
		int inSize = JlmSize(&node->input(0)->type());
		int outSize = JlmSize(&node->output(0)->type());
		if (IsIdentityMapping(*op)) {
			if(inSize == outSize){
				Connect(body, outData, inData);
			} else{
				Connect(body, outData, AddBitsOp(body, inData, outSize-1, 0));
			}
		} else {
			auto size = op->nbits();
			mlir::Value result = GetConstant(body, size, op->default_alternative());
			for (auto it = op->begin(); it != op->end(); it++) {
				auto comparison = AddEqOp(body, inData, GetConstant(body, size, it->first));
				auto value = GetConstant(body, size, it->second);
				result = AddMuxOp(body, comparison, value, result);
			}
			if((unsigned long) outSize != size){
				result = AddBitsOp(body, result, outSize-1, 0);
			}
			Connect(body,outData,result);
		}
	} else if (auto op = dynamic_cast<const jlm::GetElementPtrOperation *>(&(node->operation()))) {
		// Start of with base pointer
		auto input0 = GetSubfield(body, inBundles[0], "data");
		mlir::Value result = AddCvtOp(body, input0);

		//TODO: support structs
		const jive::type *pointeeType = &op->GetPointeeType();
		for (size_t i = 1; i < node->ninputs(); i++) {
			int bits = JlmSize(pointeeType);
			if (dynamic_cast<const jive::bittype *>(pointeeType)) {
				;
			} else if (auto arrayType = dynamic_cast<const jlm::arraytype *>(pointeeType)) {
				pointeeType = &arrayType->element_type();
			} else {
				throw std::logic_error(pointeeType->debug_string() + " pointer not implemented!");
			}
			// GEP inputs are signed
			auto input = GetSubfield(body, inBundles[i], "data");
			auto asSInt = AddAsSIntOp(body, input);
			int bytes = bits / 8;
			auto constantOp = GetConstant(body, 64, bytes);
			auto cvtOp = AddCvtOp(body, constantOp);
			auto offset = AddMulOp(body, asSInt, cvtOp);
			result = AddAddOp(body, result, offset);
		}
		auto asUInt = AddAsUIntOp(body, result);
		Connect(body, outData, AddBitsOp(body, asUInt, 63, 0));
	} else if (dynamic_cast<const jlm::UndefValueOperation *>(&(node->operation()))) {
		Connect(body, outData, GetConstant(body, 1, 0));
	} else {
		throw std::logic_error("Simple node " + node->operation().debug_string() + " not implemented!");
	}

	// Generate the output valid signal
	auto oneBitValue = GetConstant(body, 1, 1);
	mlir::Value prevAnd = oneBitValue;
	for (size_t i=0; i<node->ninputs(); i++) {
		auto bundle = inBundles[i];
		prevAnd = AddAndOp(body, prevAnd, GetSubfield(body, bundle, "valid"));
	}
	// Connect the valide signal to the output bundle
	auto outValid = GetSubfield(body, outBundle, "valid");
	Connect(body, outValid, prevAnd);

	// Generate the ready signal
	auto outReady = GetSubfield(body, outBundle, "ready");
	auto andReady = AddAndOp(body, outReady, prevAnd);
	// Connect it to the ready signal of the two input bundles
	for (size_t i=0; i<node->ninputs(); i++) {
		auto bundle = inBundles[i];
		auto ready = GetSubfield(body, bundle, "ready");
		Connect(body, ready, andReady);
	}

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenSink(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	// Create a constant of UInt<1>(1)
	auto intType= GetIntType(1);
	auto constant = builder.create<circt::firrtl::ConstantOp>(
				location,
				intType,
				llvm::APInt(1, 1));
	body->push_back(constant);

	// Get the input bundle
	auto bundle = GetInPort(module, 0);
	// Get the ready signal from the bundle (first signal in the bundle)
	auto ready = GetSubfield(body, bundle, "ready");
	// Connect the constant to the ready signal
	Connect(body, ready, constant);

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenFork(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	// Input signals
	auto inBundle = GetInPort(module, 0);
	auto inReady = GetSubfield(body, inBundle, "ready");
	auto inValid = GetSubfield(body, inBundle, "valid");
	auto inData  = GetSubfield(body, inBundle, "data");

	//
	// Output registers
	//
	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);
	llvm::SmallVector<circt::firrtl::RegResetOp> firedRegs;
	llvm::SmallVector<circt::firrtl::AndPrimOp> whenConditions;
	auto oneBitValue = GetConstant(body, 1, 1);
	auto zeroBitValue = GetConstant(body, 1, 0);
	mlir::Value prevAnd = oneBitValue;
	for (size_t i = 0; i < node->noutputs(); ++i) {
		std::string validName("out");
		validName.append(std::to_string(i));
		validName.append("_fired_reg");
		auto firedReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(validName));
		body->push_back(firedReg);
		firedRegs.push_back(firedReg);

		// Get the bundle
		auto port = GetOutPort(module, i);
		auto portReady = GetSubfield(body, port, "ready");
		auto portValid = GetSubfield(body, port, "valid");
		auto portData = GetSubfield(body, port, "data");

		auto notFiredReg = AddNotOp(body, firedReg);
		auto andOp = AddAndOp(body, inValid, notFiredReg);
		Connect(body, portValid, andOp);
		Connect(body, portData, inData);

		auto orOp = AddOrOp(body, portReady, firedReg);
	        prevAnd = AddAndOp(body, prevAnd, orOp);

		// Conditions needed for the when statements
		whenConditions.push_back(AddAndOp(body, portReady, portValid));
	}
	Connect(body, inReady, prevAnd);

	// When statement
	auto condition = AddNotOp(body, prevAnd);
	auto whenOp = AddWhenOp(body, condition, true);
	// getThenBlock() cause an error during commpilation
	// So we first get the builder and then its associated body
	auto thenBody = whenOp.getThenBodyBuilder().getBlock();
	// Then region
	for (size_t i=0; i<node->noutputs(); i++) {
		auto nestedWhen = AddWhenOp(thenBody, whenConditions[i], false);
		auto nestedBody = nestedWhen.getThenBodyBuilder().getBlock();
		Connect(nestedBody, firedRegs[i], oneBitValue);
	}
	// Else region
	auto elseBody = whenOp.getElseBodyBuilder().getBlock();
	for (size_t i=0; i<node->noutputs(); i++) {
		Connect(elseBody, firedRegs[i], zeroBitValue);
	}

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenMem(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node, true);
	auto body = module.getBody();

	// Check if it's a load or store operation
	bool store = dynamic_cast<const jlm::StoreOperation *>(&(node->operation()));

	InitializeMemReq(module);
	// Input signals
	auto inBundle0 = GetInPort(module, 0);
	auto inReady0 = GetSubfield(body, inBundle0, "ready");
	auto inValid0 = GetSubfield(body, inBundle0, "valid");
	auto inData0  = GetSubfield(body, inBundle0, "data");

	auto inBundle1 = GetInPort(module, 1);
	auto inReady1 = GetSubfield(body, inBundle1, "ready");
	auto inValid1 = GetSubfield(body, inBundle1, "valid");
	auto inData1  = GetSubfield(body, inBundle1, "data");

	// Stores also have a data input that needs to be handled
	// The input is not used by loads but code below reference
	// these variables so we need to define them
	mlir::BlockArgument inBundle2 = NULL;
	circt::firrtl::SubfieldOp inReady2 = NULL;
	circt::firrtl::SubfieldOp inValid2 = NULL;
	circt::firrtl::SubfieldOp inData2 = NULL;
	if (store) {
		inBundle2 = GetInPort(module, 2);
		inReady2 = GetSubfield(body, inBundle2, "ready");
		inValid2 = GetSubfield(body, inBundle2, "valid");
		inData2  = GetSubfield(body, inBundle2, "data");
	}

	// Output signals
	auto outBundle0 = GetOutPort(module, 0);
	auto outReady0 = GetSubfield(body, outBundle0, "ready");
	auto outValid0 = GetSubfield(body, outBundle0, "valid");
	auto outData0  = GetSubfield(body, outBundle0, "data");

	// Mem signals
	mlir::BlockArgument memReq = GetPort(module, "mem_req");
	mlir::BlockArgument memRes = GetPort(module, "mem_res");

	auto memReqReady = GetSubfield(body, memReq, "ready");
	auto memReqValid = GetSubfield(body, memReq, "valid");
	auto memReqAddr  = GetSubfield(body, memReq, "addr");
	auto memReqData  = GetSubfield(body, memReq, "data");
	auto memReqWrite = GetSubfield(body, memReq, "write");
	auto memReqWidth = GetSubfield(body, memReq, "width");

	auto memResValid = GetSubfield(body, memRes, "valid");
	auto memResData  = GetSubfield(body, memRes, "data");

	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);
	auto zeroBitValue = GetConstant(body, 1, 0);
	auto oneBitValue = GetConstant(body, 1, 1);

	// Registers
	llvm::SmallVector<circt::firrtl::RegResetOp> oValidRegs;
	llvm::SmallVector<circt::firrtl::RegResetOp> oDataRegs;
	for (size_t i=0; i<node->noutputs(); i++) {
		std::string validName("o");
		validName.append(std::to_string(i));
		validName.append("_valid_reg");
		auto validReg = builder.create<circt::firrtl::RegResetOp>(
					location,
					GetIntType(1),
					clock,
					reset,
					zeroBitValue,
					builder.getStringAttr(validName));
		body->push_back(validReg);
		oValidRegs.push_back(validReg);

		auto zeroValue = GetConstant(body, JlmSize(&node->output(i)->type()), 0);
		std::string dataName("o");
		dataName.append(std::to_string(i));
		dataName.append("_data_reg");
		auto dataReg = builder.create<circt::firrtl::RegResetOp>(
					location,
					GetIntType(&node->output(i)->type()),
					clock,
					reset,
					zeroValue,
					builder.getStringAttr(dataName));
		body->push_back(dataReg);
		oDataRegs.push_back(dataReg);
	}
	auto sentReg = builder.create<circt::firrtl::RegResetOp>(
					location,
					GetIntType(1),
					clock,
					reset,
					zeroBitValue,
					builder.getStringAttr("sent_reg"));
	body->push_back(sentReg);

	mlir::Value canRequest = AddNotOp(body, sentReg);
	canRequest = AddAndOp(body, canRequest, inValid0);
	canRequest = AddAndOp(body, canRequest, inValid1);
	if (store) {
		canRequest = AddAndOp(body, canRequest, inValid2);
	}
	for (size_t i = 0; i < node->noutputs(); i++) {
		canRequest = AddAndOp(body, canRequest, AddNotOp(body, oValidRegs[i]));
	}

	// Block until all inputs and no outputs are valid
	Connect(body, memReqValid, canRequest);
	Connect(body, memReqAddr, inData0);

	int bitWidth;
	if (store) {
		Connect(body, memReqWrite, oneBitValue);
		Connect(body, memReqData, inData1);
		bitWidth = dynamic_cast<const jive::bittype *>(&node->input(1)->type())->nbits();
	} else {
		Connect(body, memReqWrite, zeroBitValue);
		auto invalid = GetInvalid(body, 32);
		Connect(body, memReqData, invalid);
		if (auto bitType = dynamic_cast<const jive::bittype *>(&node->output(0)->type())) {
			bitWidth = bitType->nbits();
		} else if (dynamic_cast<const jlm::PointerType *>(&node->output(0)->type())) {
			bitWidth = 64;
		} else {
			throw jlm::error("unknown width for mem request");
		}
	}

	int log2Bytes = log2(bitWidth / 8);
	Connect(body, memReqWidth, GetConstant(body, 3, log2Bytes));

	// mem_req fire
	auto whenReqFireOp = AddWhenOp(body, memReqReady, false);
	auto whenReqFireBody = whenReqFireOp.getThenBodyBuilder().getBlock();
	Connect(whenReqFireBody, sentReg, oneBitValue);
	if (store) {
		Connect(whenReqFireBody, oValidRegs[0], oneBitValue);
		Connect(whenReqFireBody, oDataRegs[0], inData2);
	} else {
		Connect(whenReqFireBody, oValidRegs[1], oneBitValue);
		Connect(whenReqFireBody, oDataRegs[1], inData1);
	}

	// mem_res fire
	auto whenResFireOp = AddWhenOp(body, AddAndOp(body, sentReg, memResValid), false);
	auto whenResFireBody = whenResFireOp.getThenBodyBuilder().getBlock();
	Connect(whenResFireBody, sentReg, zeroBitValue);
	if (!store) {
		Connect(whenResFireBody, oValidRegs[0], oneBitValue);
		if(bitWidth!=64){
			auto bitsOp = AddBitsOp(whenResFireBody, memResData, bitWidth-1, 0);
			Connect(whenResFireBody, oDataRegs[0], bitsOp);
		} else{
			Connect(whenResFireBody, oDataRegs[0], memResData);
		}
	}

	// Handshaking
	Connect(body, inReady0, memReqReady);
	Connect(body, inReady1, memReqReady);
	if (store) {
		Connect(body, inReady2, memReqReady);
	}

	Connect(body, outValid0, oValidRegs[0]);
	Connect(body, outData0, oDataRegs[0]);
	auto andOp = AddAndOp(body, outReady0, outValid0);
	Connect(
		// When o0 fires
		AddWhenOp(body, andOp, false).getThenBodyBuilder().getBlock(),
		oValidRegs[0], zeroBitValue
		);
	if (!store) {
        auto outBundle1 = GetOutPort(module, 1);
        auto outReady1 = GetSubfield(body, outBundle1, "ready");
        auto outValid1 = GetSubfield(body, outBundle1, "valid");
        auto outData1  = GetSubfield(body, outBundle1, "data");

		Connect(body, outValid1, oValidRegs[1]);
		Connect(body, outData1, oDataRegs[1]);
		auto andOp = AddAndOp(body, outReady1, outValid1);
		Connect(
			// When o1 fires
			AddWhenOp(body, andOp, false).getThenBodyBuilder().getBlock(),
			oValidRegs[1], zeroBitValue
			);
	}

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenTrigger(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	// Input signals
	auto inBundle0 = GetInPort(module, 0);
	auto inReady0 = GetSubfield(body, inBundle0, "ready");
	auto inValid0 = GetSubfield(body, inBundle0, "valid");
	// auto inData0  = GetSubfield(body, inBundle0, "data");
	auto inBundle1 = GetInPort(module, 1);
	auto inReady1 = GetSubfield(body, inBundle1, "ready");
	auto inValid1 = GetSubfield(body, inBundle1, "valid");
	auto inData1  = GetSubfield(body, inBundle1, "data");
        // Output signals
	auto outBundle = GetOutPort(module, 0);
	auto outReady = GetSubfield(body, outBundle, "ready");
	auto outValid = GetSubfield(body, outBundle, "valid");
	auto outData  = GetSubfield(body, outBundle, "data");

	auto andOp0 = AddAndOp(body, outReady, inValid1);
	auto andOp1 = AddAndOp(body, outReady, inValid0);
	auto andOp2 = AddAndOp(body, inValid0, inValid1);

	Connect(body, inReady0, andOp0);
	Connect(body, inReady1, andOp1);
	Connect(body, outValid, andOp2);
	Connect(body, outData, inData1);

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenPrint(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);

	// Input signals
	auto inBundle = GetInPort(module, 0);
	auto inReady = GetSubfield(body, inBundle, "ready");
	auto inValid = GetSubfield(body, inBundle, "valid");
	auto inData = GetSubfield(body, inBundle, "data");
        // Output signals
	auto outBundle = GetOutPort(module, 0);
	Connect(body, outBundle, inBundle);
	auto trigger = AddAndOp(body,
				AddAndOp(body, inReady, inValid),
				AddNotOp(body, reset));
	auto pn = dynamic_cast<const jlm::hls::print_op *>(&node->operation());
	auto formatString = "print node " + std::to_string(pn->id()) + ": %x\n";
	auto name = "print_node_" + std::to_string(pn->id());
	auto printValue = AddPadOp(body, inData, 64);
	llvm::SmallVector<mlir::Value> operands;
	operands.push_back(printValue);
	body->push_back(builder.create<circt::firrtl::PrintFOp>(location, clock, trigger, formatString, operands, name));
	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenPredicationBuffer(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);
	auto zeroBitValue = GetConstant(body, 1,0);
	auto oneBitValue = GetConstant(body, 1,1);

	std::string validName("buf_valid_reg");
	auto validReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				oneBitValue,
				builder.getStringAttr(validName));
	body->push_back(validReg);

	std::string dataName("buf_data_reg");
	auto dataReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(&node->input(0)->type()),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(dataName));
	body->push_back(dataReg);

	auto inBundle = GetInPort(module, 0);
	auto inReady = GetSubfield(body, inBundle, "ready");
	auto inValid = GetSubfield(body, inBundle, "valid");
	auto inData = GetSubfield(body, inBundle, "data");

	auto outBundle = GetOutPort(module, 0);
	auto outReady = GetSubfield(body, outBundle, "ready");
	auto outValid = GetSubfield(body, outBundle, "valid");
	auto outData = GetSubfield(body, outBundle, "data");

	auto orOp = AddOrOp(body, validReg, inValid);
	Connect(body, outValid, orOp);
	auto muxOp = AddMuxOp(body, validReg, dataReg, inData);
	Connect(body, outData, muxOp);
	auto notOp = AddNotOp(body, validReg);
	Connect(body, inReady, notOp);

	// When
	auto condition = AddAndOp(body, inValid, inReady);
	auto whenOp = AddWhenOp(body, condition, false);
	auto thenBody = whenOp.getThenBodyBuilder().getBlock();
	Connect(thenBody, validReg, oneBitValue);
	Connect(thenBody, dataReg, inData);

	// When
	condition = AddAndOp(body, outValid, outReady);
	whenOp = AddWhenOp(body, condition, false);
	thenBody = whenOp.getThenBodyBuilder().getBlock();
	Connect(thenBody, validReg, zeroBitValue);

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenBuffer(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	auto op = dynamic_cast<const hls::buffer_op *>(&(node->operation()));
	auto capacity = op->capacity;

	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);
	auto zeroBitValue = GetConstant(body, 1, 0);
	auto zeroValue = GetConstant(body, JlmSize(&node->input(0)->type()), 0);
	auto oneBitValue = GetConstant(body, 1, 1);

	// Registers
	llvm::SmallVector<circt::firrtl::RegResetOp> validRegs;
	llvm::SmallVector<circt::firrtl::RegResetOp> dataRegs;
	for (size_t i=0; i<=capacity; i++) {
		std::string validName("buf");
		validName.append(std::to_string(i));
		validName.append("_valid_reg");
		auto validReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(validName));
		body->push_back(validReg);
		validRegs.push_back(validReg);

		std::string dataName("buf");
		dataName.append(std::to_string(i));
		dataName.append("_data_reg");
		auto dataReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(&node->input(0)->type()),
				clock,
				reset,
				zeroValue,
				builder.getStringAttr(dataName));
		body->push_back(dataReg);
		dataRegs.push_back(dataReg);
	}
	// FIXME
	// Resource waste as the registers will constantly be set to zero
	// This simplifies the code below but might waste resources unless
	// the tools are clever anough to replace it with a constant
	Connect(body, validRegs[capacity], zeroBitValue);
	Connect(body, dataRegs[capacity], zeroValue);

	// Add wires
	llvm::SmallVector<circt::firrtl::WireOp> shiftWires;
	llvm::SmallVector<circt::firrtl::WireOp> consumedWires;
	for (size_t i=0; i<=capacity; i++) {
		std::string shiftName("shift_out");
		shiftName.append(std::to_string(i));
		shiftWires.push_back(AddWireOp(body, shiftName, 1));
		std::string consumedName("in_consumed");
		consumedName.append(std::to_string(i));
		consumedWires.push_back(AddWireOp(body, consumedName, 1));
	}

	auto inBundle = GetInPort(module, 0);
	auto inReady = GetSubfield(body, inBundle, "ready");
	auto inValid = GetSubfield(body, inBundle, "valid");
	auto inData = GetSubfield(body, inBundle, "data");

	auto outBundle = GetOutPort(module, 0);
	auto outReady = GetSubfield(body, outBundle, "ready");
	auto outValid = GetSubfield(body, outBundle, "valid");
	auto outData = GetSubfield(body, outBundle, "data");

	// Connect out to buf0
	Connect(body, outValid, validRegs[0]);
	Connect(body, outData, dataRegs[0]);
	auto andOp = AddAndOp(body, outReady, outValid);
	Connect(body, shiftWires[0], andOp);
	if (op->pass_through) {
		auto notOp = AddNotOp(body, validRegs[0]);
		andOp = AddAndOp(body, notOp, outReady);
		Connect(body, consumedWires[0], andOp);
		auto whenOp = AddWhenOp(body, notOp, false);
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, outData, inData);
	} else {
		Connect(body, consumedWires[0], zeroBitValue);
	}

	// The buffer is ready if the last one is empty
	auto notOp = AddNotOp(body, validRegs[capacity-1]);
	Connect(body, inReady, notOp);

	andOp = AddAndOp(body, inReady, inValid);
	for (size_t i = 0; i < capacity; ++i) {
		Connect(body, consumedWires[i+1], consumedWires[i]);
		Connect(body, shiftWires[i+1], zeroBitValue);

		// When valid reg
		auto whenOp = AddWhenOp(body, shiftWires[i], false);
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, validRegs[i], zeroBitValue);

		// When will be empty
		auto notOp = AddNotOp(body, validRegs[i]);
		auto condition = AddOrOp(body, shiftWires[i], notOp);
		whenOp = AddWhenOp(body, condition, false);
		thenBody = whenOp.getThenBodyBuilder().getBlock();
		// Create the condition needed in nested when
		notOp = AddNotOp(thenBody, consumedWires[i]);
		auto elseCondition = AddAndOp(thenBody, andOp, notOp);

		// Nested when valid reg
		whenOp = AddWhenOp(thenBody, validRegs[i+1], true);
		thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, validRegs[i], oneBitValue);
		Connect(thenBody, dataRegs[i], dataRegs[i+1]);
		Connect(thenBody, shiftWires[i+1], oneBitValue);

		// Nested else in available
		auto elseBody = whenOp.getElseBodyBuilder().getBlock();
		auto nestedWhen = AddWhenOp(elseBody, elseCondition, false);
		thenBody = nestedWhen.getThenBodyBuilder().getBlock();
		Connect(thenBody, consumedWires[i+1], oneBitValue);
		Connect(thenBody, validRegs[i], oneBitValue);
		Connect(thenBody, dataRegs[i], inData);
	}

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenDMux(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	auto zeroBitValue = GetConstant(body, 1, 0);
	auto oneBitValue  = GetConstant(body, 1, 1);

	auto inputs = node->ninputs();
	auto outBundle = GetOutPort(module, 0);
	auto outReady = GetSubfield(body, outBundle, "ready");
	// Out valid
	auto outValid = GetSubfield(body, outBundle, "valid");
	Connect(body, outValid, zeroBitValue);
	// Out data
	auto invalid = GetInvalid(body, JlmSize(&node->output(0)->type()));
	auto outData = GetSubfield(body, outBundle, "data");
	Connect(body, outData, invalid);
	// Input ready 0
	auto inBundle0 = GetInPort(module, 0);
	auto inReady0 = GetSubfield(body, inBundle0, "ready");
	auto inValid0 = GetSubfield(body, inBundle0, "valid");
	auto inData0  = GetSubfield(body, inBundle0, "data");
	Connect(body, inReady0, zeroBitValue);

	// Add discard registers
	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);
	llvm::SmallVector<circt::firrtl::RegResetOp> discardRegs;
	llvm::SmallVector<circt::firrtl::WireOp> discardWires;
	mlir::Value anyDiscardReg = GetConstant(body, 1, 0);
	for (size_t i=1; i<inputs; i++) {
		std::string regName("i");
		regName.append(std::to_string(i));
		regName.append("_discard_reg");
		auto reg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(regName));
		body->push_back(reg);
		discardRegs.push_back(reg);
		anyDiscardReg = AddOrOp(body, anyDiscardReg, reg);

		std::string wireName("i");
		wireName.append(std::to_string(i));
		wireName.append("_discard");
		auto wire = AddWireOp(body, wireName, 1);
	        discardWires.push_back(wire);
		Connect(body, wire, reg);
		Connect(body, reg, wire);
	}
	auto notAnyDiscardReg = AddNotOp(body, anyDiscardReg);

	auto processedReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr("processed_reg"));
	body->push_back(processedReg);
	auto notProcessedReg = AddNotOp(body, processedReg);

	for (size_t i=1; i<inputs; i++) {
		auto inBundle = GetInPort(module, i);
		auto inReady = GetSubfield(body, inBundle, "ready");
		auto inValid = GetSubfield(body, inBundle, "valid");
		auto inData  = GetSubfield(body, inBundle, "data");

		Connect(body, inReady, discardWires[i-1]);

		// First when
		auto andOp = AddAndOp(body, inReady, inValid);
		auto whenOp = AddWhenOp(body, andOp, false);
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, discardRegs[i-1], zeroBitValue);

		// Second when
		auto constant = GetConstant(body, 64, i-1);
		auto eqOp = AddEqOp(body, inData0, constant);
		auto andOp0 = AddAndOp(body, inValid0, eqOp);
		auto andOp1 = AddAndOp(body, notAnyDiscardReg, andOp0);
		whenOp = AddWhenOp(body, andOp1, false);
		thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, outValid, inValid);
		Connect(thenBody, outData, inData);
		Connect(thenBody, inReady, outReady);
		auto andOp2 = AddAndOp(thenBody, outReady, inValid);
		Connect(thenBody, inReady0, andOp2);

		// Nested when
		whenOp = AddWhenOp(thenBody, notProcessedReg, false);
		thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, processedReg, oneBitValue);
		for (size_t j = 1; j < inputs; ++j) {
			if (i != j) {
				Connect(thenBody, discardWires[j-1], oneBitValue);
			}
		}
	}

	auto andOp = AddAndOp(body, outValid, outReady);
	auto whenOp = AddWhenOp(body, andOp, false);
	auto thenBody = whenOp.getThenBodyBuilder().getBlock();
	Connect(thenBody, processedReg, zeroBitValue);

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenNDMux(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	auto inputs = node->ninputs();
	auto outBundle = GetOutPort(module, 0);
	auto outReady = GetSubfield(body, outBundle, "ready");
	// Out valid
	auto outValid = GetSubfield(body, outBundle, "valid");
	auto zeroBitValue = GetConstant(body, 1, 0);
	Connect(body, outValid, zeroBitValue);
	// Out data
	auto invalid = GetInvalid(body, JlmSize(&node->output(0)->type()));
	auto outData = GetSubfield(body, outBundle, "data");
	Connect(body, outData, invalid);

	auto inBundle0 = GetInPort(module, 0);
	auto inReady0 = GetSubfield(body, inBundle0, "ready");
	auto inValid0 = GetSubfield(body, inBundle0, "valid");
	Connect(body, inReady0, zeroBitValue);
	auto inData0 = GetSubfield(body, inBundle0, "data");

	// We have already handled the first input (i.e., i == 0)
	for (size_t i = 1; i < inputs; i++) {
		auto inBundle = GetInPort(module, i);
		auto inReady = GetSubfield(body, inBundle, "ready");
		auto inValid = GetSubfield(body, inBundle, "valid");
		auto inData = GetSubfield(body, inBundle, "data");
		Connect(body, inReady, zeroBitValue);
		auto constant = GetConstant(body, JlmSize(&node->input(0)->type()), i-1);
		auto eqOp = AddEqOp(body, inData0, constant);
		auto andOp = AddAndOp(body, inValid0, eqOp);
		auto whenOp = AddWhenOp(body, andOp, false);
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, outValid, inValid);
		Connect(thenBody, outData, inData);
		Connect(thenBody, inReady, outReady);
		auto whenAnd = AddAndOp(thenBody, outReady, inValid);
		Connect(thenBody, inReady0, whenAnd);
	}
	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGenBranch(const jive::simple_node *node) {
	// Create the module and its input/output ports
	auto module = nodeToModule(node);
	auto body = module.getBody();

	auto zeroBitValue = GetConstant(body, 1, 0);

	auto inBundle0 = GetInPort(module, 0);
	auto inReady0 = GetSubfield(body, inBundle0, "ready");
	auto inValid0 = GetSubfield(body, inBundle0, "valid");
	auto inData0 = GetSubfield(body, inBundle0, "data");

	auto inBundle1 = GetInPort(module, 1);
	auto inReady1 = GetSubfield(body, inBundle1, "ready");
	auto inValid1 = GetSubfield(body, inBundle1, "valid");
	auto inData1 = GetSubfield(body, inBundle1, "data");

	Connect(body, inReady0, zeroBitValue);
	Connect(body, inReady1, zeroBitValue);

	auto invalid = GetInvalid(body, 1);
	for (size_t i = 0; i < node->noutputs(); i++) {
		auto outBundle = GetOutPort(module, i);
		auto outReady = GetSubfield(body, outBundle, "ready");
		auto outValid = GetSubfield(body, outBundle, "valid");
		auto outData = GetSubfield(body, outBundle, "data");
		Connect(body, outValid, zeroBitValue);
		Connect(body, outData, invalid);

		auto constant = GetConstant(body, JlmSize(&node->input(0)->type()), i);
		auto eqOp = AddEqOp(body, inData0, constant);
		auto condition = AddAndOp(body, inValid0, eqOp);
		auto whenOp = AddWhenOp(body, condition, false);
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, inReady1, outReady);
		auto andOp = AddAndOp(thenBody, outReady, inValid1);
		Connect(thenBody, inReady0, andOp);
		Connect(thenBody, outValid, inValid1);
		Connect(thenBody, outData, inData1);
	}

	return module;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGen(const jive::simple_node *node) {
	if (dynamic_cast<const hls::sink_op *>(&(node->operation()))) {
		return MlirGenSink(node);
	} else if (dynamic_cast<const hls::fork_op *>(&(node->operation()))) {
		return MlirGenFork(node);
	} else if (dynamic_cast<const jlm::LoadOperation *>(&(node->operation()))) {
		return MlirGenMem(node);
	} else if (dynamic_cast<const jlm::StoreOperation *>(&(node->operation()))) {
		return MlirGenMem(node);
	} else if (dynamic_cast<const hls::predicate_buffer_op *>(&(node->operation()))) {
		return MlirGenPredicationBuffer(node);
	} else if (dynamic_cast<const hls::buffer_op *>(&(node->operation()))) {
		return MlirGenBuffer(node);
	} else if (dynamic_cast<const hls::branch_op *>(&(node->operation()))) {
		return MlirGenBranch(node);
	} else if (dynamic_cast<const hls::trigger_op *>(&(node->operation()))) {
		return MlirGenTrigger(node);
	} else if (dynamic_cast<const hls::sink_op *>(&(node->operation()))) {
		//return sink_to_firrtl(n);
		throw std::logic_error(node->operation().debug_string() + " not implemented!");
	} else if (dynamic_cast<const hls::print_op *>(&(node->operation()))) {
        return MlirGenPrint(node);
	} else if (dynamic_cast<const hls::fork_op *>(&(node->operation()))) {
		//return fork_to_firrtl(n);
		throw std::logic_error(node->operation().debug_string() + " not implemented!");
	} else if (dynamic_cast<const hls::merge_op *>(&(node->operation()))) {
		// return merge_to_firrtl(n);
		throw std::logic_error(node->operation().debug_string() + " not implemented!");
	} else if (auto o = dynamic_cast<const hls::mux_op *>(&(node->operation()))) {
		if (o->discarding) {
			return MlirGenDMux(node);
		} else {
		        return MlirGenNDMux(node);
		}
	}
	return MlirGenSimpleNode(node);
}

std::unordered_map<jive::simple_node *, circt::firrtl::InstanceOp>
jlm::hls::MLIRGenImpl::MlirGen(hls::loop_node *loopNode, mlir::Block *body, mlir::Block *circuitBody) {
	auto subRegion = loopNode->subregion();

	// First we create and instantiate all the modules and keep them in a dictionary
	auto clock = body->getArgument(0);
	auto reset = body->getArgument(1);
	std::unordered_map<jive::simple_node *, circt::firrtl::InstanceOp> instances;
	for (const auto node : jive::topdown_traverser(subRegion)) {
		if (auto sn = dynamic_cast<jive::simple_node *>(node)) {
			instances[sn] = AddInstanceOp(circuitBody, sn);
			body->push_back(instances[sn]);
			Connect(body, instances[sn]->getResult(0), clock);
			Connect(body, instances[sn]->getResult(1), reset);
		} else if (auto oln = dynamic_cast<hls::loop_node *>(node)) {
			auto inst = MlirGen(oln, body, circuitBody);
			instances.merge(inst);
		} else {
			throw jlm::error("Unimplemented op (unexpected structural node) : " +
					 node->operation().debug_string());
		}
	}
	return instances;
}

// Trace the argument back to the "node" generating the value
// Returns the output of a node or the argument of a region that has
// been instantiated as a module
jive::output *
jlm::hls::MLIRGenImpl::TraceArgument(jive::argument *arg) {
	// Check if the argument is part of a hls::loop_node
	auto region = arg->region();
	auto node = region->node();
	if (dynamic_cast<hls::loop_node *>(node)) {
		if (auto ba = dynamic_cast<jlm::hls::backedge_argument*>(arg)) {
            return ba->result()->origin();
        } else {
            // Check if the argument is connected to an input,
            // i.e., if the argument exits the region
            assert(arg->input() != nullptr);
			// Check if we are in a nested region and directly
			// connected to the outer regions argument
			auto origin = arg->input()->origin();
			if (auto o = dynamic_cast<jive::argument *>(origin)) {
				// Need to find the source of the outer regions argument
				return TraceArgument(o);
			} else if (auto o = dynamic_cast<jive::structural_output *>(origin)) {
				// Check if we the input of one loop_node is connected to the output of another structural_node,
				// i.e., if the input is connected to the output of another loop_node
				return TraceStructuralOutput(o);
			}
			// Else we have reached the source
			return origin;
		}
	}
	// Reached the argument of a structural node that is not a hls::loop_node
	return arg;
}

circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::MlirGen(jive::region *subRegion, mlir::Block *circuitBody) {
	// Generate a vector with all inputs and outputs of the module
	llvm::SmallVector<circt::firrtl::PortInfo> ports;

	// Clock and reset ports
	AddClockPort(&ports);
	AddResetPort(&ports);
	// Argument ports
	for (size_t i = 0; i < subRegion->narguments(); ++i) {
		AddBundlePort(
			&ports,
			circt::firrtl::Direction::In,
			get_port_name(subRegion->argument(i)),
			GetIntType(&subRegion->argument(i)->type()));
	}
	// Result ports
	for (size_t i = 0; i < subRegion->nresults(); ++i) {
		AddBundlePort(
			&ports,
			circt::firrtl::Direction::Out,
			get_port_name(subRegion->result(i)),
			GetIntType(&subRegion->result(i)->type()));
	}
	// Memory ports
	AddMemReqPort(&ports);
	AddMemResPort(&ports);

	// Create a name for the module
	auto moduleName = builder.getStringAttr("subregion_mod");
	// Now when we have all the port information we can create the module
	auto module = builder.create<circt::firrtl::FModuleOp>(location, moduleName, ports);
	// Get the body of the module such that we can add contents to the module
	auto body = module.getBody();

	// Initialize the signals of mem_req
	InitializeMemReq(module);

	// Get the clock and reset signal of the module
	auto clock = GetClockSignal(module);
	auto reset = GetResetSignal(module);
	// First we create and instantiate all the modules and keep them in a dictionary
	std::unordered_map<jive::simple_node *, circt::firrtl::InstanceOp> instances;
	for (const auto node : jive::topdown_traverser(subRegion)) {
		if (auto sn = dynamic_cast<jive::simple_node *>(node)) {
			instances[sn] = AddInstanceOp(circuitBody, sn);
			body->push_back(instances[sn]);
			// Connect clock and reset to the instance
			Connect(body, instances[sn]->getResult(0), clock);
			Connect(body, instances[sn]->getResult(1), reset);
		} else if (auto oln = dynamic_cast<hls::loop_node *>(node)) {
			auto inst = MlirGen(oln, body, circuitBody);
			instances.merge(inst);
		} else {
			throw jlm::error("Unimplemented op (unexpected structural node) : " +
					 node->operation().debug_string());
		}
	}

	// Need to keep track of memory operations such that they can be connected
	// to the main memory port.
	//
	// TODO: The use of unorderd_maps for tracking instances maybe can break the
	//       memory order, i.e., not adhear to WAR, RAW, and WAW
	std::unordered_map<jive::simple_node *, circt::firrtl::InstanceOp> memInstances;
	// Wire up the instances
	for (const auto & instance : instances) {
		// RVSDG node
		auto rvsdgNode = instance.first;
		// Corresponding InstanceOp
		auto sinkNode = instance.second;

		// Memory instances will need to be connected to the main memory ports
		// So we keep track of them to handle them later
		if (dynamic_cast<const jlm::LoadOperation *>(&(rvsdgNode->operation()))) {
		  memInstances.insert(instance);
		} else if (dynamic_cast<const jlm::StoreOperation *>(&(rvsdgNode->operation()))) {
		  memInstances.insert(instance);
		}

		// Go through each of the inputs of the RVSDG node and try to connect
		// the corresponding port on the InstanceOp
		for (size_t i = 0; i < rvsdgNode->ninputs(); i++) {
			// The port of the instance is connected to another instance

			// Get the RVSDG node that's the origin of this input
			jive::simple_input *input = rvsdgNode->input(i);
			auto origin = input->origin();

			// If the origin is a jive::simple_node then we connect the source output
			// with the sink input
			if (auto o = dynamic_cast<jive::simple_output *>(origin)) {
				// Get RVSDG node of the source
				auto source = o->node();
				// Calculate the result port of the instance:
				//   2 for clock and reset +
				//   Number of inputs of the node +
				//   The index of the output of the node
				auto sourceIndex = 2 + source->ninputs() + o->index();
				// Get the corresponding InstanceOp
				auto sourceNode = instances[source];

				auto sourcePort = sourceNode->getResult(sourceIndex);
				auto sinkPort = sinkNode->getResult(i+2);

				// If the source is a load operation then its output will be
				// 64 bits wide. The default width of operations is 32 bits so
				// the port widths will differ and cause errors.
				if (dynamic_cast<const jlm::LoadOperation *>(&(source->operation())))
					PartialConnect(body, sinkPort, sourcePort);
				else
					Connect(body, sinkPort, sourcePort);
			} else if (auto o = dynamic_cast<jive::argument *>(origin)) {
				auto origin = TraceArgument(o);
				if (auto o = dynamic_cast<jive::argument *>(origin)) {
					// The port of the instance is connected to an argument
					// of the region
					// Calculate the result port of the instance:
					//   2 for clock and reset +
					//   The index of the input of the region
					auto sourceIndex = 2 + o->index();
					auto sourcePort = body->getArgument(sourceIndex);
					auto sinkPort = sinkNode->getResult(i+2);
					Connect(body, sinkPort, sourcePort);
				} else if (auto o = dynamic_cast<jive::simple_output *>(origin)) {
					// Get RVSDG node of the source
					auto source = o->node();
					// Calculate the result port of the instance:
					//   2 for clock and reset +
					//   Number of inputs of the node +
					//   The index of the output of the node
					auto sourceIndex = 2 + source->ninputs() + o->index();
					// Get the corresponding InstanceOp
					auto sourceNode = instances[source];
					auto sourcePort = sourceNode->getResult(sourceIndex);
					auto sinkPort = sinkNode->getResult(i+2);

					// If the source is a load operation then its output will be
					// 64 bits wide. The default width of operations is 32 bits so
					// the port widths will differ and cause errors.
					if (dynamic_cast<const jlm::LoadOperation *>(&(source->operation())))
						PartialConnect(body, sinkPort, sourcePort);
					else
						Connect(body, sinkPort, sourcePort);
				} else {
					throw std::logic_error("Unsupported output");
				}
			} else if (auto o = dynamic_cast<jive::structural_output *>(origin)) {
				// Need to trace through the region to find the source node
				auto output = TraceStructuralOutput(o);
				// Get the node of the output
				jive::simple_node *source = output->node();
				// Get the corresponding InstanceOp
				auto sourceNode = instances[source];
				// Calculate the result port of the instance:
				//   2 for clock and reset +
				//   Number of inputs of the node +
				//   The index of the output of the node
				auto sourceIndex = 2 + source->ninputs() + output->index();
				auto sourcePort = sourceNode->getResult(sourceIndex);
				auto sinkPort = sinkNode->getResult(i+2);
				Connect(body, sinkPort, sourcePort);
			} else {
				throw std::logic_error("Unsupported output");
			}
		}
	}

	// Connect memory instances to the main memory ports
	mlir::Value previousGranted = GetConstant(body, 1, 0);
	for (const auto & instance : memInstances) {
		// RVSDG node
		auto rvsdgNode = instance.first;
		// Corresponding InstanceOp
		auto node = instance.second;

		// Get the index to the last port of the subregion and the node
		auto mainIndex = body->getArguments().size();
		auto nodeIndex = 2 + rvsdgNode->ninputs() + rvsdgNode->noutputs() - 1;

		// mem_res (last argument of the region and result of the instance)
		auto mainMemRes = body->getArgument(mainIndex-1);
		auto nodeMemRes = node->getResult(nodeIndex+2);
		Connect(body, nodeMemRes, mainMemRes);

		// mem_req (second to last argument of the region and result of the instance)
		// The arbitration is prioritized for now so the first memory operation
		// (as given by memInstances) that makes a request will be granted.
		auto mainMemReq = body->getArgument(mainIndex-2);
		auto nodeMemReq = node->getResult(nodeIndex+1);
		auto memReqReady = GetSubfield(body, nodeMemReq, "ready");
		Connect(body, memReqReady, GetConstant(body, 1, 0));
		auto memReqValid = GetSubfield(body, nodeMemReq, "valid");
		auto notOp = AddNotOp(body, previousGranted);
		auto condition = AddAndOp(body, notOp, memReqValid);
		auto whenOp = AddWhenOp(body, condition, false);
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		// The direction is inverted compared to mem_res
		Connect(thenBody, mainMemReq, nodeMemReq);
		// update for next iteration
		previousGranted = AddOrOp(body, previousGranted, memReqValid);
	}

	// Connect the results of the region
	for (size_t i = 0; i < subRegion->nresults(); i++) {
		auto result = subRegion->result(i);
		auto origin = result->origin();
		jive::simple_output *output;
		if (auto o = dynamic_cast<jive::simple_output *>(origin)) {
			// We have found the source output
			output = o;
		} else if (auto o = dynamic_cast<jive::structural_output *>(origin)) {
			// Need to trace through the region to find the source node
			output = TraceStructuralOutput(o);
		} else {
			throw std::logic_error("Unsupported output");
		}
		// Get the node of the output
		jive::simple_node *source = output->node();
		// Get the corresponding InstanceOp
		auto sourceNode = instances[source];
		// Calculate the result port of the instance:
		//   2 for clock and reset +
		//   Number of inputs of the node +
		//   The index of the output of the node
		auto sourceIndex = 2 + source->ninputs() + output->index();
		auto sourcePort = sourceNode->getResult(sourceIndex);

		// Calculate the result port of the region:
		//   2 for clock and reset +
		//   Number of inputs of the region +
		//   The index of the result of the region (== i)
		auto sinkIndex = 2 + subRegion->narguments() + i;
		auto sinkPort = body->getArgument(sinkIndex);

		// Connect the InstanceOp output to the result of the region
		Connect(body, sinkPort, sourcePort);
	}

        return module;
}


// Trace a structural output back to the "node" generating the value
// Returns the output of the node
jive::simple_output *
jlm::hls::MLIRGenImpl::TraceStructuralOutput(jive::structural_output *output) {
	auto node = output->node();

	// We are only expecting hls::loop_node to have a structural output
	if (!dynamic_cast<hls::loop_node *>(node)) {
		throw std::logic_error("Expected a hls::loop_node but found: " + node->operation().debug_string());
	}
    assert(output->results.size()==1);
    auto origin = output->results.begin().ptr()->origin();
    if (auto o = dynamic_cast<jive::structural_output *>(origin)) {
        // Need to trace the output of the nested structural node
        return TraceStructuralOutput(o);
    } else if (auto o = dynamic_cast<jive::simple_output *>(origin)) {
        // Found the source node
        return o;
    } else {
        throw std::logic_error("Encountered an unexpected output type");
    }
}


// Emit a circuit
circt::firrtl::CircuitOp
jlm::hls::MLIRGenImpl::MlirGen(const jlm::lambda::node *lambdaNode) {

    // Ensure consistent naming across runs
    create_node_names(lambdaNode->subregion());
	// The same name is used for the circuit and main module
	auto moduleName = builder.getStringAttr(lambdaNode->name() + "_lambda_mod");
	// Create the top level FIRRTL circuit
	auto circuit = builder.create<circt::firrtl::CircuitOp>(location, moduleName);
	// The body will be populated with a list of modules
	auto circuitBody = circuit.getBody();

	// Get the region of the function
	auto subRegion = lambdaNode->subregion();

	//
	//   Add ports
	//
	// Generate a vector with all inputs and outputs of the module
	llvm::SmallVector<circt::firrtl::PortInfo> ports;

	// Clock and reset ports
	AddClockPort(&ports);
	AddResetPort(&ports);

	// Input bundle
	using BundleElement = circt::firrtl::BundleType::BundleElement;
	llvm::SmallVector<BundleElement> inputElements;
	inputElements.push_back(GetReadyElement());
	inputElements.push_back(GetValidElement());
	for (size_t i = 0; i < subRegion->narguments(); ++i) {
		std::string portName("data");
		portName.append(std::to_string(i));
		inputElements.push_back(BundleElement(
						builder.getStringAttr(portName),
						false,
						GetIntType(&subRegion->argument(i)->type()))
					 );
	}
	auto inputType = circt::firrtl::BundleType::get(inputElements, builder.getContext());
	struct circt::firrtl::PortInfo iBundle = {
		builder.getStringAttr("i"),
		inputType,
		circt::firrtl::Direction::In,
		{builder.getStringAttr("")},
		location,
	};
	ports.push_back(iBundle);

	// Output bundle
	llvm::SmallVector<BundleElement> outputElements;
	outputElements.push_back(GetReadyElement());
	outputElements.push_back(GetValidElement());
	for (size_t i = 0; i < subRegion->nresults(); ++i) {
		std::string portName("data");
		portName.append(std::to_string(i));
		outputElements.push_back(BundleElement(
						builder.getStringAttr(portName),
						false,
						GetIntType(&subRegion->result(i)->type()))
					 );

	}
	auto outputType = circt::firrtl::BundleType::get(outputElements, builder.getContext());
	struct circt::firrtl::PortInfo oBundle = {
		builder.getStringAttr("o"),
		outputType,
		circt::firrtl::Direction::Out,
		{builder.getStringAttr("")},
		location,
	};
	ports.push_back(oBundle);

	// Memory ports
	AddMemReqPort(&ports);
	AddMemResPort(&ports);

	// Now when we have all the port information we can create the module
	// The same name is used for the circuit and main module
	auto module = builder.create<circt::firrtl::FModuleOp>(location, moduleName, ports);
	// Get the body of the module such that we can add contents to the module
	auto body = module.getBody();

	// Initialize the signals of mem_req
	InitializeMemReq(module);

	// Create a module of the region
	auto srModule = MlirGen(subRegion, circuitBody);
	circuitBody->push_back(srModule);
	// Instantiate the region
	auto instance = builder.create<circt::firrtl::InstanceOp>(location, srModule, "sr");
	body->push_back(instance);
	// Connect the Clock
	auto clock = GetClockSignal(module);
	Connect(body, instance->getResult(0), clock);
	// Connect the Reset
	auto reset = GetResetSignal(module);
	Connect(body, instance->getResult(1), reset);

	//
	// Add registers to the module
	//
	// Reset when low (0 == false) 1-bit
	auto zeroBitValue = GetConstant(body, 1,0);

	// Input registers
	llvm::SmallVector<circt::firrtl::RegResetOp> inputValidRegs;
	llvm::SmallVector<circt::firrtl::RegResetOp> inputDataRegs;
	for (size_t i = 0; i < subRegion->narguments(); ++i) {
		std::string validName("i");
		validName.append(std::to_string(i));
		validName.append("_valid_reg");
		auto validReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(validName));
		body->push_back(validReg);
		inputValidRegs.push_back(validReg);

		std::string dataName("i");
		dataName.append(std::to_string(i));
		dataName.append("_data_reg");
		auto dataReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(&subRegion->argument(i)->type()),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(dataName));
		body->push_back(dataReg);
		inputDataRegs.push_back(dataReg);

		auto port = instance.getResult(i+2);
		auto portValid = GetSubfield(body, port, "valid");
		Connect(body, portValid, validReg);
		auto portData = GetSubfield(body, port, "data");
		Connect(body, portData, dataReg);

		// When statement
		auto portReady = GetSubfield(body, port, "ready");
		auto whenCondition = AddAndOp(body, portReady, portValid);
		auto whenOp = AddWhenOp(body, whenCondition, false);

		// getThenBlock() cause an error during commpilation
		// So we first get the builder and then its associated body
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, validReg, zeroBitValue);
	}

	// Output registers

	// Need to know the number of inputs so we can calculate the
	// correct index for outputs
	auto numInputs = subRegion->narguments();
	llvm::SmallVector<circt::firrtl::RegResetOp> outputValidRegs;
	llvm::SmallVector<circt::firrtl::RegResetOp> outputDataRegs;

	auto oneBitValue = GetConstant(body, 1, 1);
	for (size_t i = 0; i < subRegion->nresults(); ++i) {
		std::string validName("o");
		validName.append(std::to_string(i));
		validName.append("_valid_reg");
		auto validReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(1),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(validName));
		body->push_back(validReg);
		outputValidRegs.push_back(validReg);

		std::string dataName("o");
		dataName.append(std::to_string(i));
		dataName.append("_data_reg");
		auto dataReg = builder.create<circt::firrtl::RegResetOp>(
				location,
				GetIntType(&subRegion->result(i)->type()),
				clock,
				reset,
				zeroBitValue,
				builder.getStringAttr(dataName));
		body->push_back(dataReg);
		outputDataRegs.push_back(dataReg);

		// Get the bundle
		auto port = instance.getResult(2 + numInputs + i);

		auto portReady = GetSubfield(body, port, "ready");
		auto notValidReg = builder.create<circt::firrtl::NotPrimOp>(
				location,
				circt::firrtl::IntType::get(builder.getContext(), false, 1),
				validReg);
		body->push_back(notValidReg);
		Connect(body, portReady, notValidReg);

		// When statement
		auto portValid = GetSubfield(body, port, "valid");
		auto portData = GetSubfield(body, port, "data");
		auto whenCondition = AddAndOp(body, portReady, portValid);
		auto whenOp = AddWhenOp(body, whenCondition, false);

		// getThenBlock() cause an error during commpilation
		// So we first get the builder and then its associated body
		auto thenBody = whenOp.getThenBodyBuilder().getBlock();
		Connect(thenBody, validReg, oneBitValue);
		Connect(thenBody, dataReg, portData);
	}

	// Create the ready signal for the input bundle
	mlir::Value prevAnd = oneBitValue;
	for (size_t i=0; i<inputValidRegs.size(); i++) {
		auto notReg = builder.create<circt::firrtl::NotPrimOp>(
				location,
				circt::firrtl::IntType::get(builder.getContext(), false, 1),
				inputValidRegs[i]);
		body->push_back(notReg);
		auto andOp = AddAndOp(body, notReg, prevAnd);
		prevAnd = andOp;
	}
	auto inBundle = body->getArgument(2);
	auto inReady = GetSubfield(body, inBundle, "ready");
	Connect(body, inReady, prevAnd);

	// Create the valid signal for the output bundle
	prevAnd = oneBitValue;
	for (size_t i=0; i<outputValidRegs.size(); i++) {
		auto andOp = AddAndOp(body, outputValidRegs[i], prevAnd);
		prevAnd = andOp;
	}
	auto outBundle = body->getArgument(3);
	auto outValid = GetSubfield(body, outBundle, "valid");
	Connect(body, outValid, prevAnd);

	// Connect output data signals
	for (size_t i=0; i<outputDataRegs.size(); i++) {
		auto outData = GetSubfield(body, outBundle, 2+i);
		Connect(body, outData, outputDataRegs[i]);
	}

	// Input when statement
	auto inValid = GetSubfield(body, inBundle, "valid");
	auto whenCondition = AddAndOp(body, inReady, inValid);
	auto whenOp = AddWhenOp(body, whenCondition, false);

	// getThenBlock() cause an error during commpilation
	// So we first get the builder and then its associated body
	auto thenBody = whenOp.getThenBodyBuilder().getBlock();
	for (size_t i=0; i<inputValidRegs.size(); i++) {
		Connect(thenBody, inputValidRegs[i], oneBitValue);
		auto inData = GetSubfield(thenBody, inBundle, 2+i);
		Connect(thenBody, inputDataRegs[i], inData);
	}

	// Output when statement
	auto outReady = GetSubfield(body, outBundle, "ready");
	whenCondition = AddAndOp(body, outReady, outValid);
	whenOp = AddWhenOp(body, whenCondition, false);
	// getThenBlock() cause an error during commpilation
	// So we first get the builder and then its associated body
	thenBody = whenOp.getThenBodyBuilder().getBlock();
	for (size_t i=0; i<outputValidRegs.size(); i++) {
		Connect(thenBody, outputValidRegs[i], zeroBitValue);
	}

	// Connect the memory ports
	auto args = body->getArguments().size();
	auto memResBundle = body->getArgument(args-1);
	auto memResValid = GetSubfield(body, memResBundle, "valid");
	auto memResData  = GetSubfield(body, memResBundle, "data");

	auto memReqBundle = body->getArgument(args-2);
	auto memReqValid = GetSubfield(body, memReqBundle, "valid");
	auto memReqAddr  = GetSubfield(body, memReqBundle, "addr");
	auto memReqData  = GetSubfield(body, memReqBundle, "data");
	auto memReqWrite = GetSubfield(body, memReqBundle, "write");
	auto memReqWidth = GetSubfield(body, memReqBundle, "width");

	auto srArgs = instance.getResults().size();
	auto srMemResBundle = instance->getResult(srArgs-1);
	auto srMemResValid = GetSubfield(body, srMemResBundle, "valid");
	auto srMemResData  = GetSubfield(body, srMemResBundle, "data");

	auto srMemReqBundle = instance->getResult(srArgs-2);
	auto srMemReqReady = GetSubfield(body, srMemReqBundle, "ready");
	auto srMemReqValid = GetSubfield(body, srMemReqBundle, "valid");
	auto srMemReqAddr  = GetSubfield(body, srMemReqBundle, "addr");
	auto srMemReqData  = GetSubfield(body, srMemReqBundle, "data");
	auto srMemReqWrite = GetSubfield(body, srMemReqBundle, "write");
	auto srMemReqWidth = GetSubfield(body, srMemReqBundle, "width");

	Connect(body, srMemResValid, memResValid);
	Connect(body, srMemResData,  memResData);
	Connect(body, srMemReqReady, zeroBitValue);

	// When statement
	whenOp = AddWhenOp(body, srMemReqValid, false);
	// getThenBlock() cause an error during commpilation
	// So we first get the builder and then its associated body
	thenBody = whenOp.getThenBodyBuilder().getBlock();
	Connect(thenBody, srMemReqReady, oneBitValue);
	Connect(thenBody, memReqValid, oneBitValue);
	Connect(thenBody, memReqAddr, srMemReqAddr);
	Connect(thenBody, memReqData, srMemReqData);
	Connect(thenBody, memReqWrite, srMemReqWrite);
	Connect(thenBody, memReqWidth, srMemReqWidth);

	// Add the module to the body of the circuit
	circuitBody->push_back(module);

	return circuit;
}

/*
  Helper functions
*/

// Returns a PortInfo of ClockType
void
jlm::hls::MLIRGenImpl::AddClockPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports) {
	struct circt::firrtl::PortInfo port = {
		builder.getStringAttr("clk"),
		circt::firrtl::ClockType::get(builder.getContext()),
		circt::firrtl::Direction::In,
		{builder.getStringAttr("")},
		location,
	};
	ports->push_back(port);
}

// Returns a PortInfo of unsigned IntType with width of 1
void
jlm::hls::MLIRGenImpl::AddResetPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports) {
	struct circt::firrtl::PortInfo port = {
		builder.getStringAttr("reset"),
		circt::firrtl::IntType::get(builder.getContext(), false, 1),
		circt::firrtl::Direction::In,
		{builder.getStringAttr("")},
		location,
	};
	ports->push_back(port);
}

void
jlm::hls::MLIRGenImpl::AddMemReqPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports) {
	using BundleElement = circt::firrtl::BundleType::BundleElement;

	llvm::SmallVector<BundleElement> memReqElements;
	memReqElements.push_back(GetReadyElement());
	memReqElements.push_back(GetValidElement());
	memReqElements.push_back(BundleElement(
					builder.getStringAttr("addr"),
					false,
					circt::firrtl::IntType::get(builder.getContext(),
								    false, 64))
			       );
	memReqElements.push_back(BundleElement(
					builder.getStringAttr("data"),
					false,
					circt::firrtl::IntType::get(builder.getContext(),
								    false, 64))
			       );
	memReqElements.push_back(BundleElement(
					builder.getStringAttr("write"),
					false,
					circt::firrtl::IntType::get(builder.getContext(),
								    false, 1))
			       );
	memReqElements.push_back(BundleElement(
					builder.getStringAttr("width"),
					false,
					circt::firrtl::IntType::get(builder.getContext(),
								    false, 3))
			       );

	auto memType = circt::firrtl::BundleType::get(memReqElements, builder.getContext());
	struct circt::firrtl::PortInfo memBundle = {
		builder.getStringAttr("mem_req"),
		memType,
		circt::firrtl::Direction::Out,
		{builder.getStringAttr("")},
		location,
	};
	ports->push_back(memBundle);
}

void
jlm::hls::MLIRGenImpl::AddMemResPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports) {
	using BundleElement = circt::firrtl::BundleType::BundleElement;

	llvm::SmallVector<BundleElement> memResElements;
	memResElements.push_back(GetValidElement());
	memResElements.push_back(BundleElement(
					builder.getStringAttr("data"),
					false,
					circt::firrtl::IntType::get(builder.getContext(),
								    false, 64))
			       );

	auto memResType = circt::firrtl::BundleType::get(memResElements, builder.getContext());
	struct circt::firrtl::PortInfo memResBundle = {
		builder.getStringAttr("mem_res"),
		memResType,
		circt::firrtl::Direction::In,
		{builder.getStringAttr("")},
		location,
	};
	ports->push_back(memResBundle);
}

void
jlm::hls::MLIRGenImpl::AddBundlePort(
			llvm::SmallVector<circt::firrtl::PortInfo> *ports,
			circt::firrtl::Direction direction,
			std::string name,
			circt::firrtl::FIRRTLType type) {
	using BundleElement = circt::firrtl::BundleType::BundleElement;

	llvm::SmallVector<BundleElement> elements;
	elements.push_back(GetReadyElement());
	elements.push_back(GetValidElement());
	elements.push_back(BundleElement(
					builder.getStringAttr("data"),
					false,
					type));

	auto bundleType = circt::firrtl::BundleType::get(elements, builder.getContext());
	struct circt::firrtl::PortInfo bundle = {
		builder.getStringAttr(name),
		bundleType,
		direction,
		{builder.getStringAttr("")},
		location,
	};
	ports->push_back(bundle);
}

circt::firrtl::SubfieldOp
jlm::hls::MLIRGenImpl::GetSubfield(mlir::Block *body, mlir::Value value, int index) {
	auto subfield = builder.create<circt::firrtl::SubfieldOp>(location, value, index);
	body->push_back(subfield);
	return subfield;
}

circt::firrtl::SubfieldOp
jlm::hls::MLIRGenImpl::GetSubfield(mlir::Block *body, mlir::Value value, llvm::StringRef fieldName) {
	auto subfield = builder.create<circt::firrtl::SubfieldOp>(location, value, fieldName);
	body->push_back(subfield);
	return subfield;
}

mlir::BlockArgument
jlm::hls::MLIRGenImpl::GetPort(circt::firrtl::FModuleOp& module, std::string portName) {
    for (size_t i = 0; i < module.getNumPorts(); ++i) {
        if(module.getPortName(i) == portName){
            return module.getArgument(i);
        }
    }
    llvm_unreachable("port not found");
}

mlir::BlockArgument
jlm::hls::MLIRGenImpl::GetInPort(circt::firrtl::FModuleOp& module, size_t portNr) {
    return GetPort(module, "i"+std::to_string(portNr));
}

mlir::BlockArgument
jlm::hls::MLIRGenImpl::GetOutPort(circt::firrtl::FModuleOp& module, size_t portNr) {
    return GetPort(module, "o"+std::to_string(portNr));
}

void
jlm::hls::MLIRGenImpl::Connect(mlir::Block *body, mlir::Value sink, mlir::Value source) {
	body->push_back(builder.create<circt::firrtl::ConnectOp>(
				location,
				sink,
				source));
}

void
jlm::hls::MLIRGenImpl::PartialConnect(mlir::Block *body, mlir::Value sink, mlir::Value source) {
	body->push_back(builder.create<circt::firrtl::PartialConnectOp>(
				location,
				sink,
				source));
}

circt::firrtl::BitsPrimOp
jlm::hls::MLIRGenImpl::AddBitsOp(mlir::Block *body, mlir::Value value, int high, int low) {
	auto intType = builder.getIntegerType(32);
	auto op = builder.create<circt::firrtl::BitsPrimOp>(
				location,
				value,
				builder.getIntegerAttr(intType, high),
				builder.getIntegerAttr(intType, low));
	body->push_back(op);
	return op;
}

circt::firrtl::AndPrimOp
jlm::hls::MLIRGenImpl::AddAndOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::AndPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::XorPrimOp
jlm::hls::MLIRGenImpl::AddXorOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::XorPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::OrPrimOp
jlm::hls::MLIRGenImpl::AddOrOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::OrPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::NotPrimOp
jlm::hls::MLIRGenImpl::AddNotOp(mlir::Block *body, mlir::Value first) {
	auto op = builder.create<circt::firrtl::NotPrimOp>(
				location,
				first);
	body->push_back(op);
	return op;
}

circt::firrtl::AddPrimOp
jlm::hls::MLIRGenImpl::AddAddOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::AddPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::SubPrimOp
jlm::hls::MLIRGenImpl::AddSubOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::SubPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::MulPrimOp
jlm::hls::MLIRGenImpl::AddMulOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::MulPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::DivPrimOp
jlm::hls::MLIRGenImpl::AddDivOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::DivPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::DShrPrimOp
jlm::hls::MLIRGenImpl::AddDShrOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::DShrPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::DShlPrimOp
jlm::hls::MLIRGenImpl::AddDShlOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::DShlPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::RemPrimOp
jlm::hls::MLIRGenImpl::AddRemOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::RemPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::EQPrimOp
jlm::hls::MLIRGenImpl::AddEqOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::EQPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::NEQPrimOp
jlm::hls::MLIRGenImpl::AddNeqOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::NEQPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::GTPrimOp
jlm::hls::MLIRGenImpl::AddGtOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::GTPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::GEQPrimOp
jlm::hls::MLIRGenImpl::AddGeqOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::GEQPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::LTPrimOp
jlm::hls::MLIRGenImpl::AddLtOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::LTPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::LEQPrimOp
jlm::hls::MLIRGenImpl::AddLeqOp(mlir::Block *body, mlir::Value first, mlir::Value second) {
	auto op = builder.create<circt::firrtl::LEQPrimOp>(
				location,
				first,
				second);
	body->push_back(op);
	return op;
}

circt::firrtl::MuxPrimOp
jlm::hls::MLIRGenImpl::AddMuxOp(mlir::Block *body, mlir::Value select, mlir::Value high, mlir::Value low) {
	auto op = builder.create<circt::firrtl::MuxPrimOp>(
				location,
				select,
				high,
				low);
	body->push_back(op);
	return op;
}

circt::firrtl::AsSIntPrimOp
jlm::hls::MLIRGenImpl::AddAsSIntOp(mlir::Block *body, mlir::Value value) {
	auto op = builder.create<circt::firrtl::AsSIntPrimOp>(
				location,
				value);
	body->push_back(op);
	return op;
}

circt::firrtl::AsUIntPrimOp
jlm::hls::MLIRGenImpl::AddAsUIntOp(mlir::Block *body, mlir::Value value) {
	auto op = builder.create<circt::firrtl::AsUIntPrimOp>(
				location,
				value);
	body->push_back(op);
	return op;
}

circt::firrtl::PadPrimOp
jlm::hls::MLIRGenImpl::AddPadOp(mlir::Block *body, mlir::Value value, int amount) {
	auto op = builder.create<circt::firrtl::PadPrimOp>(
				location,
				value,
				amount);
	body->push_back(op);
	return op;
}

circt::firrtl::CvtPrimOp
jlm::hls::MLIRGenImpl::AddCvtOp(mlir::Block *body, mlir::Value value) {
	auto op = builder.create<circt::firrtl::CvtPrimOp>(
				location,
				value);
	body->push_back(op);
	return op;
}

circt::firrtl::WireOp
jlm::hls::MLIRGenImpl::AddWireOp(mlir::Block *body, std::string name, int size) {
	auto op = builder.create<circt::firrtl::WireOp>(
				location,
				GetIntType(size),
				name);
	body->push_back(op);
	return op;
}

circt::firrtl::WhenOp
jlm::hls::MLIRGenImpl::AddWhenOp(mlir::Block *body, mlir::Value condition, bool elseStatement) {
	auto op = builder.create<circt::firrtl::WhenOp>(
				location,
				condition,
				elseStatement);
	body->push_back(op);
	return op;
}

circt::firrtl::InstanceOp
jlm::hls::MLIRGenImpl::AddInstanceOp(mlir::Block *body, jive::simple_node *node) {
	auto name = GetModuleName(node);
	// Check if the module has already been instantiated else we need to generate it
	if (!modules[name]) {
		auto module = MlirGen(node);
		modules[name] = module;
		body->push_back(module);
	}
	// We increment a counter for each node that is instantiated
	// to assure the name is unique while still being relatively
	// easy to ready (which helps when debugging).
    auto node_name = get_node_name(node);
	return builder.create<circt::firrtl::InstanceOp>(location, modules[name], node_name);
}

circt::firrtl::ConstantOp
jlm::hls::MLIRGenImpl::GetConstant(mlir::Block *body, int size, int value) {
	auto intType= GetIntType(size);
	auto constant = builder.create<circt::firrtl::ConstantOp>(
			location,
			intType,
			llvm::APInt(size, value));
	body->push_back(constant);
	return constant;
}

circt::firrtl::InvalidValueOp
jlm::hls::MLIRGenImpl::GetInvalid(mlir::Block *body, int size) {

	auto invalid = builder.create<circt::firrtl::InvalidValueOp>(
				location,
				GetIntType(size));
	body->push_back(invalid);
	return invalid;
}

// Get the clock signal in the module
mlir::BlockArgument
jlm::hls::MLIRGenImpl::GetClockSignal(circt::firrtl::FModuleOp module) {
	auto clock = module.getArgument(0);
	auto ctype = clock.getType().cast<circt::firrtl::FIRRTLType>();
	if (!ctype.isa<circt::firrtl::ClockType>()) {
		assert("Not a ClockType");
	}
	return clock;
}


// Get the reset signal in the module
mlir::BlockArgument
jlm::hls::MLIRGenImpl::GetResetSignal(circt::firrtl::FModuleOp module) {
	auto reset = module.getArgument(1);
	auto rtype = reset.getType().cast<circt::firrtl::FIRRTLType>();
	if (!rtype.isa<circt::firrtl::ResetType>()) {
		assert("Not a ResetType");
	}
	return reset;
}

circt::firrtl::BundleType::BundleElement
jlm::hls::MLIRGenImpl::GetReadyElement() {
	using BundleElement = circt::firrtl::BundleType::BundleElement;

	return BundleElement(
		builder.getStringAttr("ready"),
		true,
		circt::firrtl::IntType::get(builder.getContext(), false, 1));
}

circt::firrtl::BundleType::BundleElement
jlm::hls::MLIRGenImpl::GetValidElement() {
	using BundleElement = circt::firrtl::BundleType::BundleElement;

	return BundleElement(
		builder.getStringAttr("valid"),
		false,
		circt::firrtl::IntType::get(builder.getContext(), false, 1));
}

void
jlm::hls::MLIRGenImpl::InitializeMemReq(circt::firrtl::FModuleOp module) {
	mlir::BlockArgument mem = GetPort(module, "mem_req");
	mlir::Block *body = module.getBody();

	auto zeroBitValue = GetConstant(body, 1, 0);
	auto invalid1 = GetInvalid(body, 1);
	auto invalid3 = GetInvalid(body, 3);
	auto invalid64 = GetInvalid(body, 64);

	auto memValid = GetSubfield(body, mem, "valid");
	auto memAddr  = GetSubfield(body, mem, "addr");
	auto memData  = GetSubfield(body, mem, "data");
	auto memWrite = GetSubfield(body, mem, "write");
	auto memWidth = GetSubfield(body, mem, "width");

	Connect(body, memValid, zeroBitValue);
	Connect(body, memAddr,  invalid64);
	Connect(body, memData,  invalid64);
	Connect(body, memWrite, invalid1);
	Connect(body, memWidth, invalid3);
}

// Takes a jive::simple_node and creates a firrtl module with an input
// bundle for each node input and output bundle for each node output
// Returns a circt::firrtl::FModuleOp with an empty body
circt::firrtl::FModuleOp
jlm::hls::MLIRGenImpl::nodeToModule(const jive::simple_node *node, bool mem) {
	// Generate a vector with all inputs and outputs of the module
	llvm::SmallVector<circt::firrtl::PortInfo> ports;

	// Clock and reset ports
	AddClockPort(&ports);
	AddResetPort(&ports);
	// Input bundle port
	for (size_t i = 0; i < node->ninputs(); ++i) {
		std::string name("i");
		name.append(std::to_string(i));
		AddBundlePort(
			&ports,
			circt::firrtl::Direction::In,
			name,
			GetIntType(&node->input(i)->type()));
	}
	for (size_t i = 0; i < node->noutputs(); ++i) {
		std::string name("o");
		name.append(std::to_string(i));
		AddBundlePort(
			&ports,
			circt::firrtl::Direction::Out,
			name,
			GetIntType(&node->output(i)->type()));
	}

    if(mem){
        AddMemReqPort(&ports);
        AddMemResPort(&ports);
    }

	// Creat a name for the module
	auto nodeName = GetModuleName(node);
	mlir::StringAttr name = builder.getStringAttr(nodeName);
	// Create the module
	return builder.create<circt::firrtl::FModuleOp>(location, name, ports);
}

//
// HLS only works with wires so all types are represented as unsigned integers
//

// Returns IntType of the specified width
circt::firrtl::IntType
jlm::hls::MLIRGenImpl::GetIntType(int size) {
	return circt::firrtl::IntType::get(builder.getContext(), false, size);
}

// Return unsigned IntType with the bit width specified by the
// jive::type. The extend argument extends the width of the IntType,
// which is usefull for, e.g., additions where the result has to be 1
// larger than the operands to accomodate for the carry.
circt::firrtl::IntType
jlm::hls::MLIRGenImpl::GetIntType(const jive::type *type, int extend) {
	return circt::firrtl::IntType::get(builder.getContext(), false, JlmSize(type)+extend);
}

std::string
jlm::hls::MLIRGenImpl::GetModuleName(const jive::node *node) {

	std::string append = "";
	for (size_t i = 0; i < node->ninputs(); ++i) {
		append.append("_I");
		append.append(std::to_string(JlmSize(&node->input(i)->type())));
		append.append("W");
	}
	for (size_t i = 0; i < node->noutputs(); ++i) {
		append.append("_O");
		append.append(std::to_string(JlmSize(&node->output(i)->type())));
		append.append("W");
	}
    if(auto op = dynamic_cast<const jlm::GetElementPtrOperation *>(&node->operation())){
        const jive::type *pointeeType = &op->GetPointeeType();
        for (size_t i = 1; i < node->ninputs(); i++) {
            int bits = JlmSize(pointeeType);
            if (dynamic_cast<const jive::bittype *>(pointeeType)) {
                ;
            } else if (auto arrayType = dynamic_cast<const jlm::arraytype *>(pointeeType)) {
                pointeeType = &arrayType->element_type();
            } else {
                throw std::logic_error(pointeeType->debug_string() + " pointer not implemented!");
            }
            int bytes = bits / 8;
            append.append("_");
            append.append(std::to_string(bytes));
        }
    }
	auto name = jive::detail::strfmt("op_", node->operation().debug_string() + append);
	// Remove characters that are not valid in firrtl module names
	std::replace_if(name.begin(), name.end(), isForbiddenChar, '_');
	return name;
}

bool
jlm::hls::MLIRGenImpl::IsIdentityMapping(const jive::match_op &op) {
	for (const auto &pair : op) {
		if (pair.first != pair.second)
			return false;
	}

	return true;
}

// Used for debugging a module by wrapping it in a circuit and writing it to a file
// Node is simply a convenience for generating the circuit name
void
jlm::hls::MLIRGenImpl::WriteModuleToFile(const circt::firrtl::FModuleOp fModuleOp, const jive::node *node) {
	if (!fModuleOp)
		return;

	auto name = GetModuleName(node);
	auto moduleName = builder.getStringAttr(name);

	// Adde the fModuleOp to a circuit
	auto circuit = builder.create<circt::firrtl::CircuitOp>(location, moduleName);
	auto body = circuit.getBody();
	body->push_back(fModuleOp);

	WriteCircuitToFile(circuit, name);
}

// Verifies the circuit and writes the FIRRTL to a file
void
jlm::hls::MLIRGenImpl::WriteCircuitToFile(const circt::firrtl::CircuitOp circuit, std::string name) {
	// Add the circuit to a top module
	auto module = mlir::ModuleOp::create(location);
	module.push_back(circuit);

	// Verify the module
	if (failed(mlir::verify(module))) {
		module.emitError("module verification error");
        throw std::logic_error("Verification of firrtl failed");
	}

	// Print the FIRRTL IR
	module.print(llvm::outs());

	// Write the module to file
	std::string fileName = name + extension();
	std::error_code EC;
	llvm::raw_fd_ostream output(fileName, EC);
	auto status = circt::firrtl::exportFIRFile(module, output);
	if (status.failed())
		throw jlm::error("Exporting of FIRRTL failed");
	output.close();
	std::cout << "\nWritten firrtl to " << fileName << "\n";
}

std::string
jlm::hls::MLIRGenImpl::toString(const circt::firrtl::CircuitOp circuit) {
	// Add the circuit to a top module
	auto module = mlir::ModuleOp::create(location);
	module.push_back(circuit);

	// Verify the module
	if (failed(mlir::verify(module))) {
		module.emitError("module verification error");
        throw std::logic_error("Verification of firrtl failed");
	}

	// Export FIRRTL to string
	std::string outputString;
	llvm::raw_string_ostream output(outputString);
	auto status = circt::firrtl::exportFIRFile(module, output);
	if (status.failed())
		throw std::logic_error("Exporting of firrtl failed");

	return outputString;
}

#endif //CIRCT
