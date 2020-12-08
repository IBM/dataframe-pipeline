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

import numpy as np
import onnx
from onnx import helper

import math

class DFPBase:
    node_count = 0
    _PARM_ALL = '__ALL'
    def __init__(self):
        self.nodes = []

    @classmethod
    def replace_PARM_ALL(cls, df, tgt):
        if tgt is cls._PARM_ALL:
            tgt = df.columns
        return tgt

    def fit_transform(self, X, **params):
        obj = self.fit(X, **params)
        return obj.transform(X)

    def fit(self, X, **params):
        return self # Do nothing for default

    def inc_node_count(self):
        ret = DFPBase.node_count
        DFPBase.node_count+=1
        return ret

    def to_onnx_operator(self, inputs, outputs, pipeline=None):
        """
        Each transformer must override this function to create its own operator.
        """
        op = helper.make_node('None', inputs, outputs, 'None_' + str(DFPBase.node_count))
        DFPBase.node_count+=1
        return op

    @classmethod
    def function_transformer_by_element(cls, df, inputs, outputs, _func):
        for input, output in zip(inputs, outputs):
            if type(input) is tuple:
                '''
                isInt = True
                for i in range(len(input)):
                    if (df[input[i]].dtype != 'int64'):
                        isInt = False
                        break
                '''

                if len(input) == 2:
                    s1 = df[input[0]]
                    s2 = df[input[1]]
                    out = []
                    for a1, a2 in zip(s1, s2):
                        out.append(_func(a1, a2))
                    df[output] = out
                else:
                    for index, _ in df.iterrows():
                        args = []
                        for i in range(len(input)):
                            args.append(df.loc[index, input[i]])
                        df.loc[index, output] = _func(*args)
                        '''
                        if isInt == True:
                            f, _ = math.modf(df.loc[index, output])
                            if f != 0.0:
                                isInt = False
                        '''

                '''
                if isInt == True:
                    df[output] = df[output].astype('int64')
                '''
            else:
                df.loc[:,output] = df[input].apply(func=_func)
        return df

    @classmethod
    def function_transformer_by_column(cls, df, inputs, outputs, _func):
        for input, output in zip(inputs, outputs):
            args = []
            if type(input) is tuple:
                for i in range(len(input)):
                    args.append(df[input[i]])
            else:
                args.append(df[input])
            #df[output] = _func(*args)
            df.loc[:,output] = _func(*args)
        return df

    @classmethod
    def convert_type_string(cls, t):    # string for key or value, string for default
        if t is str:
            return 'strings', 'string'
        elif t is float:
            return 'floats', 'float'
        elif t is int:
            return 'int64s', 'int64'
        else:
            return None

    @classmethod
    def convert_type_string_For_Const(cls, v):
        t = type(v)
        if t is tuple:
            t = type(v[0])

        if t is str:
            return 'string'
        elif t is float:
            return 'float'
        elif t is int:
            return 'int'
        else:
            return None

    @classmethod
    def conv_primitive_types_string(cls, types):
        ret = []
        for n in types:
            if n is np.int8:
                ret.append('int8')
            elif n is np.int16:
                ret.append('int16')
            elif n is np.int32:
                ret.append('int32')
            elif n is np.int64:
                ret.append('int64')
            elif n is np.uint8:
                ret.append('uint8')
            elif n is np.uint16:
                ret.append('uint16')
            elif n is np.uint32:
                ret.append('uint32')
            elif n is np.uint64:
                ret.append('uint64')
            elif n is np.float16:
                ret.append('float16')
            elif n is np.float32:
                ret.append('float32')
            elif n is np.float64:
                ret.append('float64')
            elif n is np.float128:
                ret.append('float128')
            elif n is np.bool:
                ret.append('bool')
            elif n is int:
                ret.append('int')
            elif n is float:
                ret.append('float')
            elif n is str:
                ret.append('string')
            else:
                ret.append(str(n))
        return ret

    @classmethod
    def conv_primitive_types_TensorProto(cls, types):
        ret = []
        for n in types:
            if n is np.int8:
                ret.append(onnx.TensorProto.INT8)
            elif n is np.int16:
                ret.append(onnx.TensorProto.INT16)
            elif n is np.int32:
                ret.append(onnx.TensorProto.INT32)
            elif n is np.int64:
                ret.append(onnx.TensorProto.INT64)
            elif n is np.uint8:
                ret.append(onnx.TensorProto.UINT8)
            elif n is np.uint16:
                ret.append(onnx.TensorProto.UINT16)
            elif n is np.uint32:
                ret.append(onnx.TensorProto.UINT32)
            elif n is np.uint64:
                ret.append(onnx.TensorProto.UINT64)
            elif n is np.float16:
                ret.append(onnx.TensorProto.FLOAT)
            elif n is np.float32:
                ret.append(onnx.TensorProto.FLOAT)
            elif n is np.float64:
                ret.append(onnx.TensorProto.FLOAT)
            elif n is np.float128:
                ret.append(onnx.TensorProto.FLOAT16)
            elif n is np.bool:
                ret.append(onnx.TensorProto.BOOL)
            elif n is int:
                ret.append(onnx.TensorProto.INT32)
            elif n is float:
                ret.append(onnx.TensorProto.FLOAT)
            elif n is str:
                ret.append(onnx.TensorProto.STRING)
            else:
                ret.append(str(n))
        return ret

    @classmethod
    def get_typed_key_value_pair(cls, dic, ignore_typecheck = False) -> (type, type, list, list):
        key_type = None
        value_type = None
        keys = []
        values = []
        for key, value in dic.items():
            if key_type is None:
                key_type = type(key)
            elif ignore_typecheck is False and type(key) is not key_type:
                print('keys have different types')
                raise Exception

            if value_type is None:
                value_type = type(value)
            elif ignore_typecheck is False and type(value) is not value_type:
                print('values have different types')
                raise Exception

            keys.append(key)
            values.append(value)

        return (key_type, value_type, keys, values)

    @classmethod
    def replaceValuesDict(cls, dic, oldValue, newValue):
        for k, v in dic.items():
            if v == oldValue:
                dic[k] = newValue   # dic may have multiple oldValue, so not break
        return

    @classmethod
    def replaceValuesList(cls, lst, oldValue, newValue):
        for i, v in enumerate(lst):
            if v == oldValue:
                lst[i] = newValue   # lst may have multiple oldValue, so not break
        return

    @classmethod
    def updateColumn(cls, targetList, column, newColumn):
        cdelimiter = column + '#'
        for i, c in enumerate(targetList):
            if c == column or c.startswith(cdelimiter):
                targetList[i] = newColumn
                break

    @classmethod
    def makeInputList(cls, targetColumns, referenceList, pipeline, add_new_column = False):
        ret = []
        for c in targetColumns:
            cdelimiter = c + '#'
            found = False
            for rc in referenceList:
                if rc.startswith(cdelimiter) or rc == c:
                    ret.append(rc)
                    found = True
                    break
            if not found:
                ret.append(pipeline.get_last_column_info(c, add_new_column))
        return ret

