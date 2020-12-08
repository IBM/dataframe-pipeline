##############################################################################
# Copyright 2020 IBM Corp. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##############################################################################

import types
import sys
import opcode
import code
import pandas
import inspect
import os
import numpy as np
from . import DFPBase
from . import StringSplitter
from . import compilation
import networkx as nx

from bytecode import Bytecode, dump_bytecode, Label, Instr, Compare

from onnx import helper

map_onnx_operator = {
    'BINARY_ADD' : 'Add',
    'BINARY_SUBTRACT' : 'Sub',
    'BINARY_TRUE_DIVIDE' : 'Div',
    'BINARY_MULTIPLY' : 'Mul',
    'LOAD_CONST' : 'Constant',

    # 0 means dummy operator. We won't create an onnx operator, but we have to handle it inside.
    'LOAD_FAST': 0,
    'LOAD_GLOBAL': 0,
    'LOAD_ATTR': 0,
    'LOAD_DEREF': 0,
    'CALL_FUNCTION' : 0,
    'BUILD_TUPLE' : 0,
    'COMPARE_OP' : 0,
    'POP_JUMP_IF_FALSE' : 0,
    'POP_JUMP_IF_TRUE' : 0,
    'RETURN_VALUE' : 0,
    'BINARY_SUBSCR' : 0,
}

map_function_onnx = {
    'sqrt': 'Sqrt',
    'abs': 'Abs',
    'absolute': 'Abs',
    'ceil': 'Ceil',
    'exp': 'Exp',
    'floor': 'Floor',
    'log': 'Log',
    'min': 'Min',
    'max': 'Max',
    'mean': 'Mean',
    'power': 'Pow',
    'str': 'Cast',
    'timedelta': 'TimeDelta',
    'isnan': 'isNaN',
    'lower': 'StrLower',
}

num_edges = {       # (inputs, outputs)
    'BINARY_ADD' : (2, 1),
    'BINARY_SUBTRACT' : (2, 1),
    'BINARY_SUBSCR' : (2, 1),
    'BINARY_TRUE_DIVIDE' : (2, 1),
    'BINARY_MULTIPLY' : (2, 1),
    'COMPARE_OP' : (2, 1),
    'LOAD_ATTR' : (1, 1),
    'LOAD_FAST' : (0, 1),
    'LOAD_GLOBAL' : (0, 1),
    'LOAD_DEREF' : (0, 1),
    'LOAD_CONST' : (0, 1),
    'POP_JUMP_IF_FALSE' : (1, 0),
    'POP_JUMP_IF_TRUE' : (1, 0),
    'RETURN_VALUE' : (1, 0),

    'JUMP_FORWARD' : (0, 0),
}

jmp_opcode = {
    'POP_JUMP_IF_FALSE',
    'POP_JUMP_IF_TRUE',
    'RETURN_VALUE',
    'JUMP_FORWARD',
}

jmp_patch = {
    'POP_JUMP_IF_FALSE',
    'POP_JUMP_IF_TRUE',
    'JUMP_FORWARD',
}

conditional_branches = {
    'POP_JUMP_IF_FALSE',
    'POP_JUMP_IF_TRUE',
}

output_TimeDelta = [
    'MY',
    'WY',
    'DY',
    'DM',
    'DW',
    'HD',
]

compOpToOnnx = {    # (onnxOp, isNot)
    Compare.EQ : ('Equal', False),
    Compare.IS : ('Equal', False),
    Compare.NE : ('Equal', True),
    Compare.IS_NOT : ('Equal', True),
    Compare.LT : ('Less', False),
    Compare.LE : ('LessOrEqual', False),
    Compare.GT : ('Greater', False),
    Compare.GE : ('GreaterOrEqual', False),
    Compare.IN : ('In', False),
    Compare.NOT_IN : ('In', True),
}

class InfoInstr:
    def __init__(self, instr: Instr, pos: int):
        self._instr = instr
        self._pos = pos

    def __repr__(self):
        return "<InfoInstr: %s, %d>" % (self._instr, self._pos)

    def checkName(self, check: str):
        return self.name() == check

    def name(self):
        return self._instr.name

    def arg(self):
        return self._instr.arg

    def instr(self):
        return self._instr

    def pos(self):
        return self._pos

class CallFunction:

    def __init__(self, callInstr: InfoInstr):
        self.__callInstr = callInstr
        self.__attrInstr = None
        self.__type = None
        self.__args = []
        self.__kwarg = None

    def getCallInstr(self):
        return self.__callInstr

    def getAttrInstr(self):
        return self.__attrInstr

    def getName(self):
        return self.__attrInstr.instr().arg

    def getType(self):
        return self.__type

    def numArgs(self):
        return len(self.__args)

    def getArgs(self):
        return self.__args

    def addArg(self, arg: list):
        self.__args.append(arg)

    def getKwarg(self):
        return self.__kwarg

    def setKwarg(self, kwarg):
        self.__kwarg = kwarg

    def setType(self, type: str):
        self.__type = type

    def setAttrInstr(self, attrInstr: Instr):
        self.__attrInstr = attrInstr

    def __repr__(self):
        return "<CallFunction: %s, %s, %s %s>" % (self.__callInstr, self.__attrInstr, self.__args, self.__kwarg)

delimiter='#'
name_originaldf='original_df'
class WalkBytecode:
    def __init__(self, func, input_columns, output_columns, onnx_inputs, onnx_outputs, inc_node_count, pipeline):
        self.__input_columns = input_columns
        self.__output_columns = output_columns
        self.__inc_node_count = inc_node_count
        self.__onnx_inputs = onnx_inputs
        self.__onnx_outputs = onnx_outputs
        self.__pipeline = pipeline
        self.__arg_name_dict = {}
        self.__onnxedge_dict = {} # k; v = position: ([children], nameOfOnnxEdge)
        self.__onnxnode_dict = {} # k: v = position: onnxNode
        self.__onnxnode_results = []
        self.__onnxnode_splitdf = None
        self.__originaldf_splitdf = None
        self.__onnxnode_mergedf = None
        self.__alreadyAnalyzed = []
        if hasattr(func, '__code__'):
            self.__sig = inspect.signature(func)
            self.__originalBytecodeList = Bytecode.from_code(func.__code__)
            self.__makeArgMap()
            #self.__createSplitDF()
        else:
            self.__sig = None
            if func is np.sqrt:
                self.__originalBytecodeList = "sqrt"

    def __makeArgMap(self):
        arg_name_dict = {}
        parm = self.__sig.parameters
        flat_inputs = []
        #for input in self.__input_columns:
        for input in self.__onnx_inputs:
            if type(input) is tuple:
                for inin in input:
                    flat_inputs.append(inin)
            else:
                flat_inputs.append(input)
        split_outputs = []
        for i, k in enumerate(parm.keys()):
            nameOfOnnxEdge = flat_inputs[i]
            #nameOfOnnxEdge = flat_inputs[i] + delimiter + split_id + delimiter + str(i)
            #split_outputs.append(nameOfOnnxEdge)
            arg_name_dict[k] = nameOfOnnxEdge
        self.__arg_name_dict = arg_name_dict

    def __createSplitDF(self):
        split_id = str(self.__inc_node_count())
        self.__makeArgMap(split_id)
        self.__originaldf_splitdf = name_originaldf + delimiter + str(self.__inc_node_count())
        split_outputs.append(self.__originaldf_splitdf)
        splitdf = helper.make_node('SplitDF', self.__onnx_inputs, split_outputs, 'SplitDF' + delimiter + split_id)
        self.__onnxnode_splitdf = splitdf
        self.__onnxnode_results.append(splitdf)


    def __handleStack(self, instr: InfoInstr, stack: list) -> CallFunction:
        if isinstance(instr.instr(), Instr) == False:
            pass
        elif instr.name() == 'JUMP_FORWARD':
            pass
        elif instr.name() == 'LOAD_GLOBAL':
            stack.append(([instr], None))
        elif instr.name() == 'LOAD_FAST':
            stack.append(([instr], None))
        elif instr.name() == 'LOAD_ATTR':
            newChain = [instr]
            chain, t = stack.pop()
            newChain.extend(chain)
            stack.append((newChain, t))
        elif instr.name() == 'LOAD_CONST':
            stack.append(([instr], None))
        elif instr.name() == 'STORE_DEREF':
            stack.pop()
        elif instr.name() == 'LOAD_CLOSURE':
            stack.append(([instr], None))
        elif instr.name() == 'LOAD_DEREF':
            stack.append(([instr], None))
        elif instr.name() == 'BUILD_TUPLE':
            newChain = [instr]
            for i in range(0, instr.arg()):
                chain, t = stack.pop()
                newChain.extend([chain])
            stack.append((newChain, None))
        elif instr.name() == 'MAKE_FUNCTION':
            newChain = [instr]
            lambdaChain, lambdaType = stack.pop()
            codeChain, codeType = stack.pop()
            newChain.extend(lambdaChain)
            newChain.extend(codeChain)
            if (instr.arg() & 8) != 0:
               tupleChain, tupleType = stack.pop()
               newChain.extend(tupleChain)
            stack.append((newChain, None))
        elif instr.name() == 'CALL_FUNCTION' or instr.name() == 'CALL_FUNCTION_KW':
            cf = CallFunction(instr)
            newChain = [instr]

            if instr.name() == 'CALL_FUNCTION_KW':
                chain, t = stack.pop()
                newChain.extend([chain])
                cf.setKwarg(chain)
                
            for i in range(0, instr.arg()):
                chain, t = stack.pop()
                newChain.extend([chain])
                cf.addArg(chain)

            chain, t = stack.pop()
            newChain.extend(chain)
            cf.setAttrInstr(chain[0])
            cf.setType(t)

            if chain[0].arg() == 'astype' or chain[0].arg() == 'map':
                stack.append((newChain, t))
            else:
                stack.append((newChain, None))

            return cf
        elif instr.name() == 'ROT_TWO':
            tos0, tos0Type = stack.pop()
            tos1, tos1Type = stack.pop()
            stack.append((tos0, tos0Type))
            stack.append((tos1, tos1Type))
        elif instr.name() == 'COMPARE_OP':
            newChain = [instr]
            tos0, tos0Type = stack.pop()
            tos1, tos1Type = stack.pop()
            newChain.extend(tos0)
            newChain.extend(tos1)
            stack.append((newChain, None))
        elif (instr.name() == 'BINARY_SUBSCR' or
              instr.name() == 'BINARY_SUBTRACT' or
              instr.name() == 'BINARY_TRUE_DIVIDE'):
            newChain = [instr]
            tos0, tos0Type = stack.pop()
            tos1, tos1Type = stack.pop()
            newChain.extend(tos0)
            newChain.extend(tos1)
            stack.append((newChain, None))
        elif (instr.name() == 'BINARY_ADD' or
              instr.name() == 'BINARY_MULTIPLY'):
            newChain = [instr]
            tos0, tos0Type = stack.pop()
            tos1, tos1Type = stack.pop()
            newChain.extend(tos0)
            newChain.extend(tos1)
            stack.append((newChain, None))
        elif instr.name() == 'BUILD_LIST':
            newChain = [instr]
            tos0, tos0Type = stack.pop()
            newChain.extend(tos0)
            stack.append((newChain, None))
        elif (instr.name() == 'POP_JUMP_IF_FALSE' or
                instr.name() == 'POP_JUMP_IF_TRUE'):
            tos0, tos0Type = stack.pop()
        elif instr.name() == 'STORE_FAST':
            tos0, tos0Type = stack.pop()
        elif instr.name() == 'RETURN_VALUE':
            tos0, tos0Type = stack.pop()
        else:
            print("!!!!!!! ", instr.name(), " !!!!!!")
            assert False

        return None

    def getEdgeNameOfLoadGlobal(self, attrInfo):
        posAttr = attrInfo.pos()
        #print("posAttr=", posAttr)
        #print("self.__onnxedge_dict=", self.__onnxedge_dict[posAttr])
        if len(self.__onnxedge_dict[posAttr][0]) > 0:
            childPos = self.__onnxedge_dict[posAttr][0][0]
            edgeName = self.__onnxedge_dict[childPos][1]
        else:
            edgeName = self.__onnxedge_dict[posAttr][1]
        return edgeName
        
    def checkNumPy(self, attrInfo):
        return self.getEdgeNameOfLoadGlobal(attrInfo) == 'numpy'

    def createOnnxNodeFromCF(self, cf, onnx_op, pythonName = None):
        children = []
        input_onnxedges = []
        kwarg = {}
        lenKwarg = 0
        kwargTuple = None
        if cf.getKwarg() != None:
            kwargTuple = cf.getKwarg()[0].arg()
            lenKwarg = len(kwargTuple)
        for i, j in enumerate(cf.getArgs()):
            pos = j[0].pos()
            children.append(pos)
            chinst = self.__originalBytecodeList[pos]
            _, edgeName = self.__onnxedge_dict[pos]
            input_onnxedges.append(edgeName)
            if chinst.name == 'LOAD_FAST':
                columnName = edgeName
            if kwargTuple != None and i < lenKwarg:
                kwarg[kwargTuple[lenKwarg - (i + 1)]] = edgeName
        nodeName = onnx_op + delimiter + str(self.__inc_node_count())
        if onnx_op == 'TimeDelta':
            assert False, "Not implemented yet"
            output = [columnName + delimiter + x for x in output_TimeDelta]
            for x in output:
                self.__pipeline.update_column_info(columnName, x)
        else:
            edgeName = columnName + delimiter + nodeName
            self.__pipeline.update_column_info(columnName, edgeName)
            output = [edgeName]
        if pythonName == 'str':
            kwarg['to'] = onnx.TensorProto.STRING
        #print("kwarg=", kwarg)
        #onnxNode = helper.make_node(onnx_op, input_onnxedges, output, nodeName, domain='ai.onnx.ml', **kwarg)
        onnxNode = helper.make_node(onnx_op, input_onnxedges, output, nodeName, **kwarg)
        return (children, output, onnxNode)

    def makeOnnxCompareOp(self, instr, children):
        op = instr.arg
        onnx_op, isNot = compOpToOnnx[op]
        input_onnxedges = []
        columnName = None
        for pos in children:
            chinst = self.__originalBytecodeList[pos]
            _, edgeName = self.__onnxedge_dict[pos]
            input_onnxedges.append(edgeName)
            if chinst.name == 'LOAD_FAST':
                columnName = edgeName
            else:
                cnames = edgeName.split(delimiter)
                columnName = cnames[0]
        nodeName = onnx_op + delimiter + str(self.__inc_node_count())
        edgeName = columnName + delimiter + nodeName if columnName != None else nodeName
        if isNot:
            edgeName += delimiter + 'Not'
        if columnName != None:
            self.__pipeline.update_column_info(columnName, edgeName)
        output = [edgeName]
        onnxNode = helper.make_node(onnx_op, input_onnxedges, output, nodeName)
        return onnxNode

    def makeArgsConstant(self, constValue):
        kwarg = {}
        typeStr = DFPBase.convert_type_string_For_Const(constValue)
        kwarg["value_" + typeStr] = constValue
        return kwarg

    def makeEdgeName(self, opName):
        return opName + delimiter + str(self.__inc_node_count())

    def createONNXConst(self, value, edgeName = None):
        if edgeName is None:
            edgeName = self.makeEdgeName('Constant')
        constNode = helper.make_node('Constant', [], [edgeName], edgeName, **self.makeArgsConstant(value))
        return (constNode, edgeName)

    def searchLabel(self, tgtLabel):
        for i, instr in enumerate(self.__originalBytecodeList):
            if instr == tgtLabel:
                #print(type(instr))
                return i 
        return None

    def analyzeNodeUntilJumpOrLabel(self, stack, i, output_onnx_list):
        first_i = i
        maxlen = len(self.__originalBytecodeList)
        while i < maxlen:
            instr = self.__originalBytecodeList[i]
            if isinstance(instr, Label):
                if i != first_i:
                    return i-1
            elif instr.name in jmp_opcode:
                return i-1
            self.handleOneInstr(stack, i, instr, output_onnx_list)
            i = i + 1
        return None

    def analyzeIF(self, stack, i, instr, child, org_output_onnx_list):
        output_onnx_list_jmp = []
        output_onnx_list_fallthrough = []
        #print('analyzeIF:', i, instr)
        #print(child)
        cmpOpEdgeName = self.__onnxedge_dict[child.pos()][1]
        '''
        cmpOp = child.instr().arg
        ch0Pos = cmpOpEdge[0]
        ch1Pos = cmpOpEdge[1]
        ch0EdgeName = self.__onnxedge_dict[ch0Pos][1]
        ch1EdgeName = self.__onnxedge_dict[ch1Pos][1]
        print(ch0EdgeName, cmpOp, ch1EdgeName)
        '''
        fallthroughStart = i + 1
        fallthroughEnd = self.analyzeNodeUntilJumpOrLabel(stack, fallthroughStart, output_onnx_list_fallthrough)
        fallthroughEndEdgeName = self.__onnxedge_dict[fallthroughEnd][1]
        fallthorughEndNext = self.__originalBytecodeList[fallthroughEnd + 1]
        if fallthorughEndNext.name.startswith('POP_JUMP_IF_'):
            #print("Found nested-if")
            fallthroughEnd += 1
            self.analyzeIF(stack, fallthroughEnd, fallthorughEndNext, stack[-1][0][0], output_onnx_list)
            self.__alreadyAnalyzed.append(fallthroughEnd)
            fallthroughEndEdgeName = self.__onnxedge_dict[fallthroughEnd][1]    # Use If for the output

            fallthroughEnd = self.analyzeNodeUntilJumpOrLabel(stack, fallthroughEnd + 1)    # "+ 1" means skipping POP_JUMP_IF_*
            fallthorughEndNext = self.__originalBytecodeList[fallthroughEnd + 1]
            #print("After analyzing nested if:", fallthroughEnd, fallthroughEndEdgeName, fallthorughEndNext)
        jmpStart = self.searchLabel(instr.arg) + 1
        jmpEnd = self.analyzeNodeUntilJumpOrLabel(stack, jmpStart, output_onnx_list_jmp)
        jmpEndEdgeName = self.__onnxedge_dict[jmpEnd][1]
        #print("fallthroughStart, fallthroughEnd, jmpStart, jmpEnd: ", fallthroughStart, fallthroughEnd, jmpStart, jmpEnd)
        jmpEndNext = self.__originalBytecodeList[jmpEnd + 1]
        goodDiamondShape = False
        # Analyze good diamond shape or not
        if fallthorughEndNext.name == 'JUMP_FORWARD':
            if fallthorughEndNext.arg == jmpEndNext:
                jmpEnd += 1
                fallthroughEnd += 1
                self.__alreadyAnalyzed.append(jmpStart-1)
                self.__alreadyAnalyzed.append(fallthroughEnd)
                self.__alreadyAnalyzed.append(jmpEnd)
                goodDiamondShape = True
        elif fallthorughEndNext.name == 'RETURN_VALUE':
            if jmpEndNext.name == 'RETURN_VALUE':
                fallthroughEnd += 1
                self.__alreadyAnalyzed.append(fallthroughEnd)
                goodDiamondShape = True
        assert goodDiamondShape
        #print("2 fallthroughStart, fallthroughEnd, jmpStart, jmpEnd:", fallthroughStart, fallthroughEnd, jmpStart, jmpEnd)
        # create the node "If"
        kwarg = {}
        inputs = [cmpOpEdgeName]
        reverse = instr.name == 'POP_JUMP_IF_FALSE'
        if cmpOpEdgeName.split(delimiter)[-1] == 'Not':
            reverse = not reverse
        if reverse:
            kwarg['then_branch'] = fallthroughEndEdgeName
            kwarg['else_branch'] = jmpEndEdgeName
            inputs.append(fallthroughEndEdgeName)
            inputs.append(jmpEndEdgeName)
        else:
            kwarg['then_branch'] = jmpEndEdgeName
            kwarg['else_branch'] = fallthroughEndEdgeName
            inputs.append(jmpEndEdgeName)
            inputs.append(fallthroughEndEdgeName)
        #print("inputs=", inputs)
        newChain = [InfoInstr(instr, i)]
        tos0, tos0Type = stack.pop()
        tos1, tos1Type = stack.pop()
        newChain.extend(tos0)
        newChain.extend(tos1)
        stack.append((newChain, None))
        #kwarg['cmpop'] = cmpOp
        onnx_op = 'If'
        edgeName = self.makeEdgeName(onnx_op)
        onnxNode = helper.make_node(onnx_op, DFPBase.makeInputList(inputs, self.__onnx_inputs, self.__pipeline), [edgeName], edgeName, **kwarg)
        self.__onnxnode_dict[i] = onnxNode
        org_output_onnx_list.append(onnxNode)
        self.__onnxedge_dict[i] = ([child.pos()], edgeName)
        #print("analyzeIF create edge: ", i, self.__onnxedge_dict[i])
        #print("exit analyzeIF")


    def handleOneInstr(self, stack, i, instr, output_onnx_list):
        if i in self.__alreadyAnalyzed:
            return

        #print("Analyzing... ", i)
        ch0 = None
        ch1 = None
        if len(stack) >= 2:
            ch0 = stack[-1]
            ch1 = stack[-2]
        elif len(stack) >= 1:
            ch0 = stack[-1]

        cf = self.__handleStack(InfoInstr(instr, i), stack)
        onnx_op = None
        onnxNode = None
        edgeName = None
        children = []
        self.__alreadyAnalyzed.append(i)
        if cf != None:  # Call Function
            #print(cf)
            attrInstr = cf.getAttrInstr()
            fname = attrInstr.arg()
            nameLoadGlobal = self.getEdgeNameOfLoadGlobal(attrInstr)
            #print("nameLoadGlobal=", nameLoadGlobal, " fname=", fname)
            if attrInstr.name() == "LOAD_GLOBAL" or nameLoadGlobal == 'numpy' or nameLoadGlobal == 'datetime' or nameLoadGlobal == 'str':
                onnx_op = map_function_onnx.get(fname)
                if onnx_op != None:
                    children, edgeName, onnxNode = self.createOnnxNodeFromCF(cf, onnx_op, pythonName=fname)
                    edgeName = edgeName[0]
                elif fname == 'log1p':  # log(1 + x)
                    constNode, edgeNameConst = self.createONNXConst(instr.arg)
                    children, edgeNameAdd, onnxNode = self.createOnnxNodeFromCF(cf, 'Add')
                    onnxNode.input.append(edgeNameConst)
                    output_onnx_list.append(constNode)
                    output_onnx_list.append(onnxNode)
                    onnx_op = 'Log'
                    edgeName = onnx_op + delimiter + str(self.__inc_node_count())
                    onnxNode = helper.make_node(onnx_op, edgeNameAdd, [edgeName], edgeName)
                elif fname == 'expm1':  # exp(x) - 1
                    constNode, edgeNameConst = self.createONNXConst(-instr.arg)
                    children, edgeNameExp, onnxNode = self.createOnnxNodeFromCF(cf, 'Exp')
                    output_onnx_list.append(constNode)
                    output_onnx_list.append(onnxNode)
                    onnx_op = 'Add'
                    edgeName = onnx_op + delimiter + str(self.__inc_node_count())
                    onnxNode = helper.make_node(onnx_op, [edgeNameExp[0], edgeNameConst], [edgeName], edgeName)
            elif fname == 'split':
                children = []
                _, ch0EdgeName = self.__onnxedge_dict[attrInstr.pos()]
                splitName = ch0EdgeName.split(delimiter)
                edgeName = splitName[0] + delimiter + 'func'
                for j in cf.getArgs():
                    pos = j[0].pos()
                    children.append(pos)
                    edgeName = edgeName + delimiter + "_arg=" + j[0].arg()
                for j in range(1, len(splitName)):
                    edgeName = edgeName + delimiter + splitName[j]

            if onnxNode != None:
                self.__onnxnode_dict[i] = onnxNode
                output_onnx_list.append(onnxNode)
            if edgeName != None:
                self.__onnxedge_dict[i] = (children, edgeName)
                #print("create edge: ", i, self.__onnxedge_dict[i])
        elif isinstance(instr, Instr):
            onnx_op = map_onnx_operator.get(instr.name)
            if onnx_op != None:
                if instr.name == 'BUILD_TUPLE':
                    assert False, "BUILD_TUPLE: Not implemented yet"
                ne = num_edges.get(instr.name)
                assert ne != None
                inputs, outputs = ne
                if inputs == 2:
                    children = [ch1[0][0].pos(), ch0[0][0].pos()]
                    if instr.name == 'COMPARE_OP':
                        onnxNode = self.makeOnnxCompareOp(instr, children)
                        edgeName = onnxNode.output[0]
                        self.__onnxnode_dict[i] = onnxNode
                        output_onnx_list.append(onnxNode)
                    elif instr.name == 'BINARY_SUBSCR':
                        index = ch0[0][0].arg()
                        _, ch0EdgeName = self.__onnxedge_dict[children[0]]
                        splitList = ch0EdgeName.split(delimiter)
                        lenList = len(splitList)
                        colPos = 2
                        splitDelimiter=None
                        if lenList >= 3:
                            if splitList[0] == 'split' and splitList[1] == 'func':
                                if splitList[2].startswith("_arg="):
                                    splitDelimiter = splitList[2].split('=')[-1]
                                    colPos += 1
                                if splitDelimiter is None:
                                    splitDelimiter = 'None'
                                #print("ch0EdgeName=", splitList, index)
                                #print("splitDelimiter=", splitDelimiter, " colPos=", colPos)
                                column = splitList[colPos]
                                for j in range(colPos+1, lenList):
                                    column = column + delimiter + splitList[j]
                                resultColumn = splitList[colPos] + "_result"
                                onnxNode = StringSplitter.make_onnx_SplitStr(self.__inc_node_count(), [column], [resultColumn], splitDelimiter, index, self.__onnx_inputs, self.__onnx_outputs, self.__pipeline)
                                edgeName = onnxNode.output[0]
                                self.__onnxnode_dict[i] = onnxNode
                                output_onnx_list.append(onnxNode)
                elif inputs == 1:
                    children = [ch0[0][0].pos()]
                    _, ch0EdgeName = self.__onnxedge_dict[children[0]]
                    if instr.name == 'LOAD_ATTR':
                        edgeName = instr.arg + delimiter + ch0EdgeName
                    elif instr.name.startswith('POP_JUMP_IF_'):
                        self.analyzeIF(stack, i, instr, ch0[0][0], output_onnx_list)
                        return
                elif inputs == 0:
                    if instr.name == 'LOAD_FAST':
                        edgeName = self.__arg_name_dict[instr.arg]
                    elif instr.name == 'LOAD_GLOBAL':
                        if instr.arg == 'np' or instr.arg == 'numpy':
                            edgeName = 'numpy'
                        else:
                            edgeName = DFPBase.makeInputList([instr.arg], self.__onnx_inputs, self.__pipeline, add_new_column=True)[0]
                    elif instr.name == 'LOAD_DEREF':
                        edgeName = DFPBase.makeInputList([instr.arg.name], self.__onnx_inputs, self.__pipeline, add_new_column=True)[0]

                if onnx_op != 0:   # Non dummy operator
                    input_onnxedges = []
                    is_return = instr.name == 'RETURN_VALUE'
                    #print("refer children: ", children)
                    for j in children:
                        #print("refer edge", j, self.__onnxedge_dict)
                        _, edgeName = self.__onnxedge_dict[j]
                        if is_return:
                            assert len(self.__output_columns) == 1, "Not implemented yet"
                            newEdgeName = self.__output_columns[0] + delimiter + str(self.__inc_node_count())
                            childONNX = self.__onnxnode_dict[j]
                            DFPBase.replaceValuesList(childONNX.output, edgeName, newEdgeName)
                            edgeName = newEdgeName
                        input_onnxedges.append(edgeName)

                    edgeName = onnx_op + delimiter + str(self.__inc_node_count())
                    if instr.name == 'LOAD_CONST':
                        onnxNode, _ = self.createONNXConst(instr.arg, edgeName=edgeName)
                    elif is_return:
                        #input_onnxedges.append(self.__originaldf_splitdf)
                        # print("is_return: output is ", self.__onnx_outputs)
                        onnxNode = helper.make_node(onnx_op, input_onnxedges, self.__onnx_outputs, edgeName)
                        self.__onnxnode_mergedf = onnxNode
                    else:
                        onnxNode = helper.make_node(onnx_op, input_onnxedges, [edgeName], edgeName)
                    self.__onnxnode_dict[i] = onnxNode
                    output_onnx_list.append(onnxNode)
                else:
                    if instr.name == 'RETURN_VALUE':
                        #print("refer children: ", children)
                        for j in children:
                            #print("refer edge", j, self.__onnxedge_dict)
                            _, edgeName = self.__onnxedge_dict[j]
                            assert len(self.__output_columns) == 1, "Not implemented yet"
                            #newEdgeName = self.__output_columns[0] + delimiter + str(self.__inc_node_count())
                            newEdgeName = self.__onnx_outputs[0]
                            childONNX = self.__onnxnode_dict[j]
                            DFPBase.replaceValuesList(childONNX.output, edgeName, newEdgeName)
                            edgeName = newEdgeName

                #print("edgeName=", edgeName)
                self.__onnxedge_dict[i] = (children, edgeName)
                #print("create edge: ", i, self.__onnxedge_dict[i])


    def createONNX(self):
        if type(self.__originalBytecodeList) is str:
            return self.__originalBytecodeList

        self.__alreadyAnalyzed = []
        #self.dumpInstr(self.__originalBytecodeList)
        stack = list()
        self.__onnxedge_dict = {}
        for i, instr in enumerate(self.__originalBytecodeList):
            self.handleOneInstr(stack, i, instr, self.__onnxnode_results)
        #print(self.__onnxnode_results)
        return self.__onnxnode_results

    @staticmethod
    def dumpInstr(inlist):
        for k, instr in enumerate(inlist):
            print(k, instr)


class GraphFactory:
    def __init__(self, func):
        if hasattr(func, '__code__'):
            self.__originalBytecodeList = Bytecode.from_code(func.__code__)
        else:
            assert False, "Not implemented yet"

    def buildCFG(self, comp):
        start = 0
        for i, instr in enumerate(self.__originalBytecodeList):
            if isinstance(instr, Instr):
                comp.nodes().add_node(i, name=instr.name, arg=instr.arg, instr=instr)
                if instr.name in jmp_opcode:
                    self.__branches.append(i)
                    comp.cfg().addBB(start, i)
                    start = i+1
            else:
                comp.nodes().add_node(i, name='Label', arg=instr, instr=instr)
                self.__labels.append(i)
                if start <= i-1:
                    comp.cfg().addBB(start, i-1)
                    start = i

        for l in self.__labels:
            label = self.__originalBytecodeList[l]
            labelm1 = self.__originalBytecodeList[l-1]
            lbb = comp.cfg().findBBNum(l)
            lbbm1 = comp.cfg().findBBNum(l-1)
            if labelm1.name not in jmp_opcode:
                comp.cfg().add_edge(lbbm1, lbb)
            for j in self.__branches:
                jinst = self.__originalBytecodeList[j]
                jbb = comp.cfg().findBBNum(j)
                if jinst.name in conditional_branches:
                    comp.cfg().add_edge(jbb, jbb+1)
                if jinst.arg == label:
                    comp.cfg().add_edge(jbb, lbb)
                    break
        comp.cfg().makeExitBlock()

    def run(self):
        if type(self.__originalBytecodeList) is str:
            return self.__originalBytecodeList

        self.compilation = compilation(self.__originalBytecodeList)
        self.__labels = []
        self.__branches = []
        # WalkBytecode.dumpInstr(self.__originalBytecodeList)
        self.buildCFG(self.compilation)
        factory = BlockInfoFactory(self.compilation)
        factory.run()
        self.compilation.dumpCFGandInst()
        '''
        for i in self.compilation.nodes().nodes:
            self.compilation.nodes().showNode(i)
        '''
        return self.compilation


class BlockInfoFactory:
    def __init__(self, compilation):
        self.compilation = compilation
        self.bytecode = compilation.bytecode()
        # { block offset -> BlockInfo }
        self.infos = {}
        self.edge_process = {}

    def run(self):
        for blk in self.compilation.cfg().nodes:
            bb = self.compilation.getBlock(blk)
            self.infos[blk] = self.run_on_block(bb)

        cfg = self.compilation.cfg()
        nodes = self.compilation.nodes()
        for newinst in reversed(range(len(self.bytecode), nodes.number_of_nodes())):
            if len(nodes.getChildIndices(newinst)) > 0:
                tgtinst = nodes.getChildIndices(newinst)[0]
                block = cfg.findBlock(tgtinst)
                block.insertAfter(tgtinst, newinst)
            elif len(nodes.getParentIndices(newinst)) > 0:
                tgtinst = nodes.getParentIndices(newinst)[0]
                block = cfg.findBlock(tgtinst)
                block.insertBefore(tgtinst, newinst)
        #self.dump()

    def run_on_block(self, blk):
        incoming_blocks = []
        info = BlockInfo(blk, blk.getBBNum(), incoming_blocks, self.compilation.nodes())
        edge_callbacks = []

        for ibbnum in blk.pred():
            ib = self.infos[ibbnum]
            incoming_blocks.append(ib)
            if (ib.bbnum, blk.getBBNum()) in self.edge_process:
                edge_callbacks.append(self.edge_process[(ib.bbnum, blk.getBBNum())])

            # Compute stack offset at block entry
            # The stack effect of our predecessors should be known
            assert ib.stack_offset is not None, ib
            new_offset = ib.stack_offset + ib.stack_effect
            if new_offset < 0:
                raise RuntimeError("computed negative stack offset for %s"
                                   % blk)
            if info.stack_offset is None:
                info.stack_offset = new_offset
            elif info.stack_offset != new_offset:
                warnings.warn("inconsistent stack offset for %s" % blk,
                              RuntimeWarning)

            # Compute syntax blocks at block entry
            assert ib.syntax_blocks is not None, ib
            if info.syntax_blocks is None:
                info.syntax_blocks = ib.syntax_blocks[:]
            elif info.syntax_blocks != ib.syntax_blocks:
                warnings.warn("inconsistent entry syntax blocks for %s" % blk,
                              RuntimeWarning)

        if info.stack_offset is None:
            # No incoming blocks => assume it's the entry block
            info.stack_offset = 0
            info.syntax_blocks = []
        info.stack_effect = 0

        for callback in edge_callbacks:
            callback(info)

        for offset in blk.getInsts():
            inst = self.bytecode[offset]
            info.bcindex = offset
            if isinstance(inst, Instr):
                self.dispatch(info, inst)

        return info

    def dump(self):
        for blk in self.infos.values():
            blk.dump()

    def dispatch(self, info, inst):
        fname = "op_%s" % inst.name
        fn = getattr(self, fname, self.handle_unknown_opcode)
        fn(info, inst)

    def handle_unknown_opcode(self, info, inst):
        raise UnsupportedError(
            "Use of unknown opcode '{}'".format(inst.opname),
            loc=Loc(filename=self.bytecode.func_id.filename,
                    line=inst.lineno)
        )

    def dup_topx(self, info, inst, count):
        orig = [info.pop() for _ in range(count)]
        orig.reverse()
        # We need to actually create new temporaries if we want the
        # IR optimization pass to work correctly (see issue #580)
        duped = [info.make_temp() for _ in range(count)]
        info.append(inst, orig=orig, duped=duped)
        for val in orig:
            info.push(val)
        for val in duped:
            info.push(val)

    def add_syntax_block(self, info, block):
        """
        Add an inner syntax block.
        """
        block.stack_offset = info.stack_offset
        info.syntax_blocks.append(block)

    def pop_syntax_block(self, info):
        """
        Pop the innermost syntax block and revert its stack effect.
        """
        block = info.syntax_blocks.pop()
        assert info.stack_offset >= block.stack_offset
        while info.stack_offset + info.stack_effect > block.stack_offset:
            info.pop(discard=True)
        return block

    def op_NOP(self, info, inst):
        pass

    def op_DUP_TOPX(self, info, inst):
        count = inst.arg
        assert 1 <= count <= 5, "Invalid DUP_TOPX count"
        self.dup_topx(info, inst, count)

    def op_DUP_TOP(self, info, inst):
        self.dup_topx(info, inst, count=1)

    def op_DUP_TOP_TWO(self, info, inst):
        self.dup_topx(info, inst, count=2)

    def op_ROT_TWO(self, info, inst):
        first = info.pop()
        second = info.pop()
        info.push(first)
        info.push(second)

    def op_ROT_THREE(self, info, inst):
        first = info.pop()
        second = info.pop()
        third = info.pop()
        info.push(first)
        info.push(third)
        info.push(second)

    def op_ROT_FOUR(self, info, inst):
        first = info.pop()
        second = info.pop()
        third = info.pop()
        forth = info.pop()
        info.push(first)
        info.push(forth)
        info.push(third)
        info.push(second)

    def op_UNPACK_SEQUENCE(self, info, inst):
        count = inst.arg
        iterable = info.pop()
        stores = [info.make_temp() for _ in range(count)]
        tupleobj = info.make_temp()
        info.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
        for st in reversed(stores):
            info.push(st)

    def op_BUILD_TUPLE(self, info, inst):
        count = inst.arg
        items = list(reversed([info.pop() for _ in range(count)]))
        tup = info.make_temp()
        info.append(inst, items=items, res=tup)
        info.push(tup)

    def op_BUILD_LIST(self, info, inst):
        count = inst.arg
        items = list(reversed([info.pop() for _ in range(count)]))
        lst = info.make_temp()
        info.append(inst, items=items, res=lst)
        info.push(lst)

    def op_LIST_APPEND(self, info, inst):
        value = info.pop()
        index = inst.arg
        target = info.peek(index)
        appendvar = info.make_temp()
        res = info.make_temp()
        info.append(inst, target=target, value=value, appendvar=appendvar, res=res)

    def op_BUILD_MAP(self, info, inst):
        dct = info.make_temp()
        count = inst.arg
        items = []
        # BUILD_MAP takes <count> pairs from the stack
        for i in range(count):
            v, k = info.pop(), info.pop()
            items.append((k, v))
        info.append(inst, items=items[::-1], size=count, res=dct)
        info.push(dct)

    def op_BUILD_SET(self, info, inst):
        count = inst.arg
        # Note: related python bug http://bugs.python.org/issue26020
        items = list(reversed([info.pop() for _ in range(count)]))
        res = info.make_temp()
        info.append(inst, items=items, res=res)
        info.push(res)

    def op_POP_TOP(self, info, inst):
        info.pop(discard=True)

    def op_STORE_ATTR(self, info, inst):
        target = info.pop()
        value = info.pop()
        info.append(inst, target=target, value=value)

    def op_DELETE_ATTR(self, info, inst):
        target = info.pop()
        info.append(inst, target=target)

    def op_STORE_FAST(self, info, inst):
        value = info.pop()
        info.append(inst, value=value)

    def op_STORE_MAP(self, info, inst):
        key = info.pop()
        value = info.pop()
        dct = info.tos
        info.append(inst, dct=dct, key=key, value=value)

    def op_STORE_DEREF(self, info, inst):
        value = info.pop()
        info.append(inst, value=value)

    def op_LOAD_FAST(self, info, inst):
        name = inst.arg
        res = info.make_temp(name)
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_CONST(self, info, inst):
        res = info.make_temp('const')
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_GLOBAL(self, info, inst):
        res = info.make_temp()
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_DEREF(self, info, inst):
        res = info.make_temp()
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_ATTR(self, info, inst):
        item = info.pop()
        res = info.make_temp()
        info.append(inst, item=item, res=res)
        info.push(res)

    def op_BINARY_SUBSCR(self, info, inst):
        index = info.pop()
        target = info.pop()
        res = info.make_temp()
        info.append(inst, index=index, target=target, res=res)
        info.push(res)

    def op_STORE_SUBSCR(self, info, inst):
        index = info.pop()
        target = info.pop()
        value = info.pop()
        info.append(inst, target=target, index=index, value=value)

    def op_DELETE_SUBSCR(self, info, inst):
        index = info.pop()
        target = info.pop()
        info.append(inst, target=target, index=index)

    def op_GET_ITER(self, info, inst):
        value = info.pop()
        res = info.make_temp()
        info.append(inst, value=value, res=res)
        info.push(res)

    def op_FOR_ITER(self, info, inst):
        iterator = info.tos
        pair = info.make_temp()
        indval = info.make_temp()
        pred = info.make_temp()
        info.append(inst, iterator=iterator, pair=pair, indval=indval, pred=pred)
        info.push(indval)
        # Setup for stack POP (twice) at loop exit (before processing instruction at jump target)
        def pop_info(info):
            info.pop()
            info.pop()
        self.edge_process[(info.block.offset, inst.get_jump_target())] = pop_info

    def op_CALL_FUNCTION(self, info, inst):
        narg = inst.arg
        args = list(reversed([info.pop() for _ in range(narg)]))
        func = info.pop()

        res = info.make_temp()
        info.append(inst, func=func, args=args, res=res)
        info.push(res)

    def op_CALL_FUNCTION_KW(self, info, inst):
        narg = inst.arg
        names = info.pop()  # tuple of names
        args = list(reversed([info.pop() for _ in range(narg)]))
        func = info.pop()

        res = info.make_temp()
        info.append(inst, func=func, args=args, names=names, res=res)
        info.push(res)

    def op_CALL_FUNCTION_EX(self, info, inst):
        if inst.arg & 1:
            errmsg = 'CALL_FUNCTION_EX with **kwargs not supported'
            raise NotImplementedError(errmsg)
        vararg = info.pop()
        func = info.pop()
        res = info.make_temp()
        info.append(inst, func=func, vararg=vararg, res=res)
        info.push(res)

    def _build_tuple_unpack(self, info, inst):
        # Builds tuple from other tuples on the stack
        tuples = list(reversed([info.pop() for _ in range(inst.arg)]))
        temps = [info.make_temp() for _ in range(len(tuples) - 1)]
        info.append(inst, tuples=tuples, temps=temps)
        # The result is in the last temp var
        info.push(temps[-1])

    def op_BUILD_TUPLE_UNPACK_WITH_CALL(self, info, inst):
        # just unpack the input tuple, call inst will be handled afterwards
        self._build_tuple_unpack(info, inst)

    def op_BUILD_TUPLE_UNPACK(self, info, inst):
        self._build_tuple_unpack(info, inst)

    def op_BUILD_CONST_KEY_MAP(self, info, inst):
        keys = info.pop()
        vals = list(reversed([info.pop() for _ in range(inst.arg)]))
        keytmps = [info.make_temp() for _ in range(inst.arg)]
        res = info.make_temp()
        info.append(inst, keys=keys, keytmps=keytmps, values=vals, res=res)
        info.push(res)

    def op_PRINT_ITEM(self, info, inst):
        warnings.warn("Python2 style print partially supported.  Please use "
                      "Python3 style print.", RuntimeWarning)
        item = info.pop()
        printvar = info.make_temp()
        res = info.make_temp()
        info.append(inst, item=item, printvar=printvar, res=res)

    def op_PRINT_NEWLINE(self, info, inst):
        printvar = info.make_temp()
        res = info.make_temp()
        info.append(inst, printvar=printvar, res=res)

    def _unaryop(self, info, inst):
        val = info.pop()
        res = info.make_temp()
        info.append(inst, value=val, res=res)
        info.push(res)

    op_UNARY_NEGATIVE = _unaryop
    op_UNARY_POSITIVE = _unaryop
    op_UNARY_NOT = _unaryop
    op_UNARY_INVERT = _unaryop

    def _binaryop(self, info, inst):
        rhs = info.pop()
        lhs = info.pop()
        res = info.make_temp()
        info.append(inst, lhs=lhs, rhs=rhs, res=res)
        info.push(res)

    op_COMPARE_OP = _binaryop

    op_INPLACE_ADD = _binaryop
    op_INPLACE_SUBTRACT = _binaryop
    op_INPLACE_MULTIPLY = _binaryop
    op_INPLACE_DIVIDE = _binaryop
    op_INPLACE_TRUE_DIVIDE = _binaryop
    op_INPLACE_FLOOR_DIVIDE = _binaryop
    op_INPLACE_MODULO = _binaryop
    op_INPLACE_POWER = _binaryop
    op_INPLACE_MATRIX_MULTIPLY = _binaryop

    op_INPLACE_LSHIFT = _binaryop
    op_INPLACE_RSHIFT = _binaryop
    op_INPLACE_AND = _binaryop
    op_INPLACE_OR = _binaryop
    op_INPLACE_XOR = _binaryop

    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop
    op_BINARY_MATRIX_MULTIPLY = _binaryop

    op_BINARY_LSHIFT = _binaryop
    op_BINARY_RSHIFT = _binaryop
    op_BINARY_AND = _binaryop
    op_BINARY_OR = _binaryop
    op_BINARY_XOR = _binaryop

    def op_SLICE_0(self, info, inst):
        """
        TOS = TOS[:]
        """
        tos = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        info.push(res)

    def op_SLICE_1(self, info, inst):
        """
        TOS = TOS1[TOS:]
        """
        tos = info.pop()
        tos1 = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, start=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        info.push(res)

    def op_SLICE_2(self, info, inst):
        """
        TOS = TOS1[:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, stop=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        info.push(res)

    def op_SLICE_3(self, info, inst):
        """
        TOS = TOS2[TOS1:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos, res=res,
                    slicevar=slicevar, indexvar=indexvar)
        info.push(res)

    def op_STORE_SLICE_0(self, info, inst):
        """
        TOS[:] = TOS1
        """
        tos = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos, value=value, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_1(self, info, inst):
        """
        TOS1[TOS:] = TOS2
        """
        tos = info.pop()
        tos1 = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, start=tos, slicevar=slicevar,
                    value=value, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_2(self, info, inst):
        """
        TOS1[:TOS] = TOS2
        """
        tos = info.pop()
        tos1 = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, stop=tos, value=value, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_3(self, info, inst):
        """
        TOS2[TOS1:TOS] = TOS3
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos, value=value,
                    slicevar=slicevar, indexvar=indexvar)

    def op_DELETE_SLICE_0(self, info, inst):
        """
        del TOS[:]
        """
        tos = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_1(self, info, inst):
        """
        del TOS1[TOS:]
        """
        tos = info.pop()
        tos1 = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, start=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_2(self, info, inst):
        """
        del TOS1[:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, stop=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_3(self, info, inst):
        """
        del TOS2[TOS1:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos,
                    slicevar=slicevar, indexvar=indexvar)

    def op_BUILD_SLICE(self, info, inst):
        """
        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)
        """
        argc = inst.arg
        if argc == 2:
            tos = info.pop()
            tos1 = info.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = info.pop()
            tos1 = info.pop()
            tos2 = info.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception("unreachable")
        slicevar = info.make_temp()
        res = info.make_temp()
        info.append(inst, start=start, stop=stop, step=step, res=res,
                    slicevar=slicevar)
        info.push(res)

    def op_POP_JUMP_IF_TRUE(self, info, inst):
        pred = info.pop()
        info.append(inst, pred=pred)
        info.terminator = inst

    def op_POP_JUMP_IF_FALSE(self, info, inst):
        pred = info.pop()
        info.append(inst, pred=pred)
        info.terminator = inst

    def op_JUMP_IF_TRUE(self, info, inst):
        pred = info.tos
        info.append(inst, pred=pred)
        info.terminator = inst

    def op_JUMP_IF_FALSE(self, info, inst):
        pred = info.tos
        info.append(inst, pred=pred)
        info.terminator = inst

    op_JUMP_IF_FALSE_OR_POP = op_JUMP_IF_FALSE
    op_JUMP_IF_TRUE_OR_POP = op_JUMP_IF_TRUE

    def op_JUMP_ABSOLUTE(self, info, inst):
        info.append(inst)
        info.terminator = inst

    def op_JUMP_FORWARD(self, info, inst):
        info.append(inst)
        info.terminator = inst

    def op_BREAK_LOOP(self, info, inst):
        self.pop_syntax_block(info)
        info.append(inst)
        info.terminator = inst

    def op_RETURN_VALUE(self, info, inst):
        info.append(inst, retval=info.pop(), castval=info.make_temp())
        info.terminator = inst

    def op_YIELD_VALUE(self, info, inst):
        val = info.pop()
        res = info.make_temp()
        info.append(inst, value=val, res=res)
        info.push(res)

    def op_SETUP_LOOP(self, info, inst):
        self.add_syntax_block(info, LoopBlock())
        info.append(inst)

    def op_SETUP_WITH(self, info, inst):
        cm = info.pop()    # the context-manager
        self.add_syntax_block(info, WithBlock())
        yielded = info.make_temp()
        info.push(yielded)
        info.append(inst, contextmanager=cm)

    def op_WITH_CLEANUP(self, info, inst):
        """
        Note: py2 only opcode
        """
        # TOS is the return value of __exit__()
        info.pop()
        info.append(inst)

    def op_WITH_CLEANUP_START(self, info, inst):
        # TOS is the return value of __exit__()
        info.pop()
        info.append(inst)

    def op_WITH_CLEANUP_FINISH(self, info, inst):
        info.append(inst)

    def op_END_FINALLY(self, info, inst):
        info.append(inst)

    def op_POP_BLOCK(self, info, inst):
        block = self.pop_syntax_block(info)
        info.append(inst)

    def op_RAISE_VARARGS(self, info, inst):
        if inst.arg == 0:
            exc = None
        elif inst.arg == 1:
            exc = info.pop()
        else:
            raise ValueError("Multiple argument raise is not supported.")
        info.append(inst, exc=exc)

    def op_MAKE_FUNCTION(self, info, inst, MAKE_CLOSURE=False):
        name = info.pop()
        code = info.pop()
        closure = annotations = kwdefaults = defaults = None
        if inst.arg & 0x8:
            closure = info.pop()
        if inst.arg & 0x4:
            annotations = info.pop()
        if inst.arg & 0x2:
            kwdefaults = info.pop()
        if inst.arg & 0x1:
            defaults = info.pop()
        res = info.make_temp()
        info.append(inst, name=name, code=code, closure=closure, annotations=annotations,
                    kwdefaults=kwdefaults, defaults=defaults, res=res)
        info.push(res)

    def op_MAKE_CLOSURE(self, info, inst):
        self.op_MAKE_FUNCTION(info, inst, MAKE_CLOSURE=True)

    def op_LOAD_CLOSURE(self, info, inst):
        res = info.make_temp()
        info.append(inst, res=res)
        info.push(res)

    #NOTE: Please see notes in `interpreter.py` surrounding the implementation
    # of LOAD_METHOD and CALL_METHOD.

    def op_LOAD_METHOD(self, *args, **kws):
        self.op_LOAD_ATTR(*args, **kws)

    def op_CALL_METHOD(self, *args, **kws):
        self.op_CALL_FUNCTION(*args, **kws)

    def _ignored(self, info, inst):
        pass

