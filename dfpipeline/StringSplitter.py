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
from . import DFPBase

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

class StringSplitter(DFPBase):
    """
    Split strings in a colum.
    
    Parameters
    ----------
    inputs : List of strings
        Column labels.

    outputs: List of strings
        Column labels.

    separator: String
        A string to separate a string.

    index: Int
        An index to split a string.

    keep: Int (0 or -1), default is 0
        When this value is 0, the first string is stored to the output column. When this value is -1, the last string is stored to the output column.

    Examples:
    ----------
    >>> df = pd.DataFrame({'Email': ['taro.jp.com', 'alice.us.com', 'bob.us']})
    
    Keep the first string
    >>> tf1 = StringSplitter(inputs=['Email'],
                             outputs=['Email_prefix'],
                             separator='.',
                             keep=0)
    >>> tf1.fit_transform(df)
            Email           Email_prefix
    0       taro.jp.com     taro
    1       alice.us.com    alice
    2       bob.us          bob

    Keep the last string
    >>> tf2 = StringSplitter(inputs=['Email'],
                             outputs=['Email_suffix'],
                             separator='.',
                             keep=-1)
    >>> tf2.fit_transform(df)
            Email           Email_suffix
    0       taro.jp.com     com
    1       alice.us.com    com
    2       bob.us          us
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        separator=None,
        index=None,
        keep=0
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.separator = separator
        self.index = index
        self.keep = keep

    def transform(self, df):
        if self.separator is not None:
            for input, output in zip(self.inputs, self.outputs):
                df[output] = df[input].map(lambda x: str(x).split(self.separator)[self.keep])
        elif self.index is not None:
            if self.keep == 0:
                for input, output in zip(self.inputs, self.outputs):
                    df[output] = df[input].map(lambda x: str(x)[:self.index])
            elif self.keep == -1:
                for input, output in zip(self.inputs, self.outputs):
                    df[output] = df[input].map(lambda x: str(x)[self.index:])
            else:
                assert False, 'keep can be set only to 0 or -1'
        else:
            assert False, 'Specify separator or index'
        return df

    def to_onnx_operator(self, graph):
        for input_column, output_column in zip(self.inputs, self.outputs):

            ops = []

            input_tensor = graph.get_current_tensor(input_column)
            input_tensor_name = input_tensor.name
            if input_tensor.type != TensorProto.STRING:
                cast_kwargs = {'to': TensorProto.STRING}
                cast_tensor = graph.get_tmp_tensor()
                ops.append(helper.make_node('Cast', [input_tensor.name], [cast_tensor], graph.get_node_name('Cast'), **cast_kwargs))
                input_tensor_name = cast_tensor.name

            output_tensor = graph.get_next_tensor(output_column, TensorProto.STRING)

            kwargs = {}
            if self.separator is not None:
                kwargs['separator'] = self.separator
            elif self.index is not None:
                kwargs['index'] = self.index
            else:
                assert False, 'Seprator or index needs to be specified'
            kwargs['keep'] = self.keep

            ops.append(helper.make_node('StringSplit', [input_tensor_name], [output_tensor.name], graph.get_node_name('StringSplit'), domain='ai.onnx.ml', **kwargs))
            graph.add([input_tensor], [output_tensor], ops)
