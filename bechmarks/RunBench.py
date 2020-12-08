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

#!/usr/bin/env python3

import argparse
import time
import cProfile, pstats, io
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import copy

from fraud_detection_1.FraudDetection1 import FraudDetection1
from insurance_1.Insurance1 import Insurance1
from mental_health_1.MentalHealth1 import MentalHealth1
from categorical_encoding_1.CategoricalEncoding1 import CategoricalEncoding1
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('benchmark_location')
parser.add_argument('--num_tests', type=int, default=100)
parser.add_argument('--frac', type=float, default=1.0)
parser.add_argument('--profile', type=bool, default=False)
parser.add_argument('--pypy', type=bool, default=False)
parser.add_argument('--it', type=int, default=5)
parser.add_argument('--mode', type=str, default='onnx-all,onnx-preprocess,onnx-model,python')
parser.add_argument('--deploy', type=str, default='online,batch')

args = parser.parse_args()
args.mode = args.mode.split(',')
args.deploy = args.deploy.split(',')

benchmark = None
if args.benchmark_location == 'fraud_detection_1':
    benchmark = FraudDetection1(args.benchmark_location)
elif args.benchmark_location == 'insurance_1':
    benchmark = Insurance1(args.benchmark_location)
elif args.benchmark_location == 'categorical_encoding_1':
    benchmark = CategoricalEncoding1(args.benchmark_location)
elif args.benchmark_location == 'mental_health_1':
    benchmark = MentalHealth1(args.benchmark_location)
else:
    assert False, 'Unknown benchmark location ' + args.benchmark_location

start = time.time()

print('Loading data ...')
X_train, Y_train, X_test, Y_test = benchmark.load_data(args.frac, args.num_tests)
print()

Y_test = Y_test.to_numpy()
batch_tests = X_test
batch_targets = Y_test

online_tests = []
online_targets = np.zeros(args.num_tests)
count = 0
for i, r in X_test.iterrows():
    if count == args.num_tests:
        break
    online_tests.append(pd.DataFrame(r.to_dict(), index=[i]))
    online_targets[count] = Y_test[count]
    count+=1

print('Benchmark:  ' + args.benchmark_location)
print('Num_tests:  ' + str(args.num_tests))
print('Frac:       ' + str(args.frac))
print('Profile:    ' + str(args.profile))
print('PyPy:       ' + str(args.pypy))
print('Iterations: ' + str(args.it))
print('Mode:       ' + str(args.mode))
print('Deploy:     ' + str(args.deploy))
print()

print('Defining pipeline ...')
benchmark.define_pipeline(X_train.columns, X_train.dtypes)
print()

print('Training ...')
create_onnx = not (len(args.mode) == 1 and args.mode[0] == 'python')
print('create_onnx=' + str(create_onnx), flush=True)
benchmark.do_training(X_train, Y_train, create_onnx=create_onnx)

end = time.time()
print('Data loading and training took ' + str(end - start) + ' sec')
print(flush=True)

def enable_profile(i):
    if args.profile == True and i > 0:
        pr = cProfile.Profile()
        pr.enable()
        return pr
    return None

def disable_profile(pr, i):
    if args.profile == True and i > 0:
        pr.disable()

def print_profile(pr, i):
    if args.profile == True and i > 0:
        s = io.StringIO()
        # sortby = 'tottime'
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        ps.print_callees()
        print(s.getvalue())
    
def do_prediction(mode):
    if args.pypy:
        # Warming up
        for i in range(5):
            print('Warming-up iteration ' + str(i))
            benchmark.do_batch_prediction(mode, copy.deepcopy(batch_tests))
            benchmark.do_prediction(mode, copy.deepcopy(online_tests))

    for deploy in args.deploy:

        do_prediction = benchmark.do_prediction
        targets = online_targets
        if deploy == 'batch':
            do_prediction = benchmark.do_batch_prediction
            targets = batch_targets

        print('Copying started ...', flush=True)
        start = time.time()
        x = []
        for i in range(args.it):
            if deploy == 'batch':
                x.append(copy.deepcopy(batch_tests))
            else:
                x.append(copy.deepcopy(online_tests))
        end = time.time()
        print('Copying took ' + str(end - start) + ' sec', flush=True)

        print('Predicting with ' + mode + ' and ' + deploy + ' ' + str(args.it) + ' times', flush=True)
        for i in range(args.it):
            print('Iteration ' + str(i), flush=True)

            pr = enable_profile(i)
            start = time.time()

            preds = do_prediction(mode, x[i])

            end = time.time()
            disable_profile(pr, i)

            total_time = end - start
            time_one_row = total_time / len(x[i])

            predictions = [round(value) for value in preds]
            accuracy = accuracy_score(targets, predictions)
            if deploy == 'batch':
                score = roc_auc_score(targets, preds)
            else:
                score = 0.0

            print('Elapsed time (sec): ' + str(total_time))
            print('Elapsed time for one row (sec): ' + str(time_one_row))
            print("Accuracy: %.4f" % accuracy)
            print("RocAuc: %.4f" % score)
            print(flush=True)

            print_profile(pr, i)

for mode in args.mode:
    if args.pypy:
        if mode == 'python':
            do_prediction(mode)
        else:
            pass # Do not run
    else:
        do_prediction(mode)
