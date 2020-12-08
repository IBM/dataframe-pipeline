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

import pandas as pd
import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx import update_registered_converter
from lightgbm import LGBMClassifier
import sys

import dfpipeline as dfp

class OnnxTensor:
    """
    Edge in an ONNX graph.

    Parameters
    ----------
    column : str
        Corresponding column name
    name : str
        Tensor name
    type : type
        Data type of this tensor
    """
    def __init__(self, column, name, type):
        self.column = column
        self.name = name
        self.type = type
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def search_end_op(self, end_op, nodes):
        found = False
        for n in self.nodes:
            f = n.search_end_op(end_op, nodes)
            found = found or f
        return found
    
class OnnxNode:
    """
    A node or a set of nodes in an ONNX graph.

    Parameters
    ----------
    ops : List
        An ONNX node or a set of ONNX nodes
    in_tensors : List
        A set of tensors inputted into this node
    out_tensors : List
        A set of tensors outputted from this node
    """
    def __init__(self, ops, in_tensors, out_tensors):
        self.ops = ops
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors

    def search_end_op(self, end_op, nodes):
        found = False
        for tensor in self.out_tensors:
            f = tensor.search_end_op(end_op, nodes)
            found = found or f

        # Collecting all of the nodes that must be included in an ONNX model
        if found or end_op in self.ops:
            for op in self.ops:
                if op not in nodes:
                    nodes.append(op)
            return True
        else:
            return False
        
class OnnxGraph:
    def __init__(self, ml_model_input_name, ml_models, path, with_pre_process, name, transformers, input_columns, input_dtypes):
        self.ml_model_input_name = ml_model_input_name
        self.ml_models = ml_models
        self.path = path
        self.with_pre_process = with_pre_process
        self.name = name
        self.transformers = transformers
        self.input_columns = input_columns
        self.input_dtypes = input_dtypes
        self.tensors = {}
        self.node_count = 0
        self.tensor_id = 0
        self.root_tensors = []
        self.drop_columns = []

    def initialize(self):
        self.tensors.clear()
        self.node_count = 0
        self.tensor_id = 0
        self.root_tensors.clear()
        self.drop_columns.clear()
        for c in self.input_columns:
            self.get_current_tensor(c)

    def __alias_ml_models(self, nodes):
        outputs = []
        initializers = []
        for i, ml_model in enumerate(self.ml_models):
            for node in ml_model.graph.node:
                node.name = node.name + '_' + str(i)
                for j in range(len(node.input)):
                    if node.input[j] != self.ml_model_input_name:
                        node.input[j] = node.input[j] + '_' + str(i)
                for j in range(len(node.output)):
                    node.output[j] = node.output[j] + '_' + str(i)
                nodes.append(node)
            for output in ml_model.graph.output:
                output.name = output.name + '_' + str(i)
                outputs.append(output)
            initializers.extend(ml_model.graph.initializer)

        return outputs, initializers

    def create(self, pipeline):
        self.initialize()
        for tr in self.transformers:
            if type(tr) is dfp.FunctionTransformer:
                tr.to_onnx_operator(graph=self, pipeline=pipeline)
            else:
                tr.to_onnx_operator(graph=self)

        concat_tensors = []
        concat_tensor_names = []
        for column, tensor in self.tensors.items():
            if column not in self.drop_columns:
                if tensor.type != TensorProto.FLOAT:
                    output_tensor = self.get_next_tensor(column, TensorProto.FLOAT)
                    cast_kwargs = {'to': TensorProto.FLOAT}
                    self.add([tensor], [output_tensor], [helper.make_node('Cast', [tensor.name], [output_tensor.name], self.get_node_name('Cast'), **cast_kwargs)])
                    tensor = output_tensor
                concat_tensors.append(tensor)
                concat_tensor_names.append(tensor.name)
        kwargs = {}
        kwargs['axis'] = 1
        end_op = helper.make_node('Concat', concat_tensor_names, [self.ml_model_input_name], self.get_node_name('Concat'), **kwargs)
        self.add(concat_tensors, [], [end_op])

        if self.with_pre_process:
            inputs = []
            input_columns_to_onnx = []
            nodes = []
            for tensor in self.root_tensors:
                if tensor.search_end_op(end_op, nodes):
                    inputs.append(helper.make_tensor_value_info(tensor.column, self.__get_tensor_type(self.input_dtypes[tensor.column]), [None, 1]))
                    input_columns_to_onnx.append(tensor.column)

            if len(self.ml_models) > 0:
                outputs, initializers = self.__alias_ml_models(nodes)
                graph = helper.make_graph(nodes, self.name, inputs, outputs, initializer=initializers)
            else:
                output = helper.make_tensor_value_info(self.ml_model_input_name, TensorProto.FLOAT, [None, len(concat_tensors)])
                graph = helper.make_graph(nodes, self.name, inputs, [output])
        else:
            assert len(self.ml_models) > 0
            nodes = []
            outputs, initializers = self.__alias_ml_models(nodes)
            graph = helper.make_graph(nodes, self.name, self.ml_models[0].graph.input, outputs, initializer=initializers)
            input_columns_to_onnx = [self.ml_model_input_name]

        model = helper.make_model(graph)
        # print(model)
        onnx.save(model, self.path)

        return input_columns_to_onnx

    def get_current_tensor(self, column):
        if column not in self.tensors:
            if column not in self.input_columns:
                assert False, column + ' column does not exist'
            self.tensors[column] = OnnxTensor(column, column, self.__get_tensor_type(self.input_dtypes[column]))
            self.root_tensors.append(self.tensors[column])

        return self.tensors[column]

    def get_next_tensor(self, column, tensor_type):
        if column not in self.tensors:
            self.tensors[column] = OnnxTensor(column, column, tensor_type)
        elif self.tensors[column].name == column:
            self.tensors[column] = OnnxTensor(column, column + '_0', tensor_type)
        else:
            id = int(self.tensors[column].name.split('_')[-1])
            self.tensors[column] = OnnxTensor(column, column + '_' + str(id + 1), tensor_type)
        return self.tensors[column]

    def add(self, in_tensors, out_tensors, ops):
        node = OnnxNode(ops, in_tensors, out_tensors)
        for tensor in in_tensors:
            tensor.add_node(node)

    def drop(self, drop_columns):
        self.drop_columns.extend(drop_columns)
        
    def __get_tensor_type(self, dtype):
        if dtype == np.object or dtype == str or str(dtype) == 'category':
            return TensorProto.STRING

        elif dtype == np.int8:
            return TensorProto.INT8
        elif dtype == np.uint8:
            return TensorProto.UINT8
        elif dtype == np.int16:
            return TensorProto.INT16
        elif dtype == np.uint16:
            return TensorProto.UINT16
        elif dtype == np.int32:
            return TensorProto.INT32
        elif dtype == np.uint32:
            return TensorProto.UINT32
        elif dtype == np.int64 or dtype == int:
            return TensorProto.INT64
        elif dtype == np.uint64:
            return TensorProto.UINT64

        elif dtype == np.float16:
            return TensorProto.FLOAT16
        elif dtype == np.float32:
            return TensorProto.FLOAT
        elif dtype == np.float64 or dtype == float:
            return TensorProto.DOUBLE

        else:
            assert False, 'Unknown type ' + str(dtype)
    
    def is_int_tensor(self, tensor_type):
        return (tensor_type == TensorProto.INT8 or tensor_type == TensorProto.UINT8 or tensor_type == TensorProto.INT16 or tensor_type == TensorProto.UINT16 or
                tensor_type == TensorProto.INT32 or tensor_type == TensorProto.UINT32 or tensor_type == TensorProto.INT64 or tensor_type == TensorProto.UINT64)

    def is_float_tensor(self, tensor_type):
        return (tensor_type == TensorProto.FLOAT16 or tensor_type == TensorProto.FLOAT or tensor_type == TensorProto.DOUBLE)

    def is_string_tensor(self, tensor_type):
        return tensor_type == TensorProto.STRING

    def is_int(self, dtype):
        return isIntTensor(self.__get_tensor_type(dtype))

    def is_float(self, dtype):
        return isFloatTensor(self.__get_tensor_type(dtype))

    def is_string(self, dtype):
        return isStringTensor(self.__get_tensor_type(dtype))

    def inc_node_count(self):
        self.node_count+=1
        return self.node_count

    def get_node_name(self, base):
        return base + '_' + str(self.inc_node_count())

    def get_tmp_tensor(self):
        tensor = 'tensor_' + str(self.tensor_id)
        self.tensor_id+=1
        return tensor

