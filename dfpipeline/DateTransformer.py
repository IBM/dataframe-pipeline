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

import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

class DateTransformer(DFPBase):
    """
    Create time features.

    Parameters
    ----------
    column : string
        Column name holding the time data. Each element of the column must be a string representing a date such as '2018-02-02 18:31' or an int value representing the time in seconds. When the time is represented as seconds, the origin argument needs to be specified to calculate the date from the time data. From this column, the following six features (columns) are created. The names of the created columns have this column name as a prefix.
          - MY (months in a year)
          - WY (weeks in a year)
          - DY (days in a year)
          - DM (days in a month)
          - DW (days in a week)
          - HD (hours in a day)

    origin: string (default is 1970-01-01)
        An origin of the time to calculate dates. This is needed when a columm has the time values in seconds. This is not needed when a column has the string values representing dates.

    Examples:
    ----------
    >>> df = pd.DataFrame({'DT': ['2018-02-02 18:31', '2018-02-03 11:15', '2018-02-03 13:11']})
    >>> tf1 = TimeTransformer(datetime='DT')
    """
    def __init__(
        self,
        column=None,
        origin=None
    ):
        super().__init__()
        self.column = column
        self.origin = origin
        self.date_fields = ['MY', 'WY', 'DY', 'DM', 'DW', 'HD']

    def transform(self, df):
        if self.origin is not None:
            df[self.column] = pd.to_datetime(df[self.column], origin=self.origin, unit='s')
        else:
            df[self.column] = pd.to_datetime(df[self.column])

        for f in self.date_fields:
            output_column = self.column + '_' + f
            if f == 'MY':
                df[output_column] = df[self.column].dt.month
            elif f == 'WY':
                df[output_column] = df[self.column].dt.isocalendar().week.astype(np.int64)
            elif f == 'DY':
                df[output_column] = df[self.column].dt.dayofyear
            elif f == 'DM':
                df[output_column] = df[self.column].dt.day
            elif f == 'DW':
                df[output_column] = df[self.column].dt.dayofweek
            elif f == 'HD':
                df[output_column] = df[self.column].dt.hour
            else:
                assert False, 'Uknown date field ' + f

        return df

    def to_onnx_operator(self, graph):
        input_tensor = graph.get_current_tensor(self.column)

        output_tensors = []
        output_tensor_names = []
        for f in self.date_fields:
            output_column = self.column + '_' + f
            output_tensor = graph.get_next_tensor(output_column, TensorProto.INT32)
            output_tensors.append(output_tensor)
            output_tensor_names.append(output_tensor.name)

        kwargs = {}
        kwargs['format'] = '%Y-%m-%d'

        op = helper.make_node('Date', [input_tensor.name], output_tensor_names, graph.get_node_name('Date'), domain='ai.onnx.ml', **kwargs)
        graph.add([input_tensor], output_tensors, [op])
