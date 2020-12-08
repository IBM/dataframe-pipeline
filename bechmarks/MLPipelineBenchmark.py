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

import sys
import numpy as np
import dfpipeline as dfp

class MLPipelineBenchmark:

    def __init__(self, path):
        self.path = path

    def load_data(self):
        assert False, 'Not implemented'

    def define_pipeline(self, colums, dtypes):
        assert False, 'Not implemented'

    def do_training(self, X, Y):
        assert False, 'Not implemented'

    def transform(self, X, do_fit=False):
        if do_fit == True:
            return self.pipeline.fit_transform(X)
        else:
            return self.pipeline.transform(X)

    def __load_onnx(self, file_name):
        import onnxruntime as rt

        sess = rt.InferenceSession(file_name)
        print('Inputs ' + str(len(sess.get_inputs())))
        print('  ', sess.get_inputs()[0])
        print('Outputs ' + str(len(sess.get_outputs())))
        print('  ', sess.get_outputs()[0])
        print()

        return sess
        
    def create_onnx(self, model_name, onnx_ml_models):
        print('Creating onnx-all ...')
        file_name = self.path + '/' + model_name + '_all.onnx'
        self.input_columns_to_onnx_all = self.pipeline.export('dense_input', file_name, onnx_ml_models, with_pre_process=True)
        self.onnx_all = self.__load_onnx(file_name)

        print('Creating onnx-preprocess ...')
        file_name = self.path + '/' + model_name + '_onnx-preprocess.onnx'
        self.input_columns_to_onnx_preprocess = self.pipeline.export('dense_input', file_name, with_pre_process=True)
        self.onnx_preprocess = self.__load_onnx(file_name)

        print('Creating onnx-model ...')
        file_name = self.path + '/' + model_name + '_onnx-model.onnx'
        self.input_columns_to_onnx_model = self.pipeline.export('dense_input', file_name, onnx_ml_models, with_pre_process=False)
        self.onnx_model = self.__load_onnx(file_name)

    def do_prediction(self, mode, X):
        preds = np.zeros(len(X))
        if mode == 'onnx-all':
            for i, x in enumerate(X):
                tensors = dfp.DataFramePipeline.convert_to_tensors(x, self.input_columns_to_onnx_all)
                outputs = self.onnx_all.run(None, tensors)
                for j in range(1, len(outputs), 2):
                    preds[i] += outputs[j][:,1]
        elif mode == 'onnx-preprocess':
            for i, x in enumerate(X):
                tensors = dfp.DataFramePipeline.convert_to_tensors(x, self.input_columns_to_onnx_preprocess)
                x = self.onnx_preprocess.run(None, tensors)[0]
                p = 0.0
                for clf in self.clfs:
                    preds[i] += clf.predict_proba(x)[:,1][0]
        elif mode == 'onnx-model':
            for i, x in enumerate(X):
                x = self.transform(x)
                if len(x.index) == 0: continue  # A transformer may remove this row.
                tensors = {'dense_input': x.to_numpy().astype(np.float32)}
                outputs = self.onnx_model.run(None, tensors)
                for j in range(1, len(outputs), 2):
                    preds[i] += outputs[j][:,1]
        else:
            for i, x in enumerate(X):
                x = self.transform(x)
                if len(x.index) == 0: continue  # A transformer may remove this row.
                for clf in self.clfs:
                    preds[i] += clf.predict_proba(x)[:,1][0]
        preds = preds / len(self.clfs)
        return preds

    def do_batch_prediction(self, mode, X):
        if mode == 'onnx-all':
            preds = np.zeros(len(X))
            tensors = dfp.DataFramePipeline.convert_to_tensors(X, self.input_columns_to_onnx_all)
            outputs = self.onnx_all.run(None, tensors)
            for i in range(1, len(outputs), 2):
                preds += outputs[i][:,1]
        elif mode == 'onnx-preprocess':
            preds = np.zeros(len(X))
            tensors = dfp.DataFramePipeline.convert_to_tensors(X, self.input_columns_to_onnx_preprocess)
            x = self.onnx_preprocess.run(None, tensors)[0]
            for clf in self.clfs:
                preds += clf.predict_proba(x)[:,1]
        elif mode == 'onnx-model':
            x = self.transform(X)
            preds = np.zeros(len(x))  # A transformer may remove a row.
            tensors = {'dense_input': x.to_numpy().astype(np.float32)}
            outputs = self.onnx_model.run(None, tensors)
            for i in range(1, len(outputs), 2):
                preds += outputs[i][:,1]
        else:
            x = self.transform(X)
            preds = np.zeros(len(x))  # A transformer may remove a row.
            for clf in self.clfs:
                preds += clf.predict_proba(x)[:,1]
        preds = preds / len(self.clfs)
        return preds
