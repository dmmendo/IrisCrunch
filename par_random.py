import numpy as np
import time
import random
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from multiprocessing import Process,Manager
import tensorflow_datasets as tfds
import time

from itertools import permutations
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_train_partition(data,split):
  return data[0:int(split*len(data))]
def get_test_partition(data,split):
  return data[int(split*len(data)):len(data)]

ds_train, ds_info = tfds.load(
    'iris',
    split=['train'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_numpy = tfds.as_numpy(ds_train)
profile_features = []
labels = []
for ex in ds_numpy:
  features.append(ex[0])
  labels.append(ex[1])

def Error(labels,preds):
  return 1 - accuracy_score(labels,preds)

"""## Limited Data Experiments"""
print("begin experiment")
num_trials = 32
sub_proc_trials = 10000
this_train_sizes = np.linspace(0.01,1,100)
results = Manager().list([[0 for j in range(sub_proc_trials*num_trials)] for i in range(len(this_train_sizes))])

def run_trial(profile_features,labels,this_train_sizes,results,num_trials,n):
  print("trial",n)
  random.seed(n)
  np.random.seed(n)
  for i in range(0,len(this_train_sizes)):
    for j in range(num_trials):
      profile_features,labels = shuffle(profile_features,labels)
      if 1-this_train_sizes[i] > 0:
        cur_X_train, cur_X_test, cur_y_train, cur_y_test = train_test_split(profile_features,labels,test_size=1-this_train_sizes[i],random_state = n)
      else:
        cur_X_train, cur_y_train = profile_features,labels
      reg = RandomForestClassifier().fit(cur_X_train,cur_y_train)
      results[i][n*num_trials + j] = Error(labels,reg.predict(profile_features))

procs = []
for n in range(num_trials):
  p = Process(target=run_trial, args=(profile_features,labels,this_train_sizes,results,sub_proc_trials,n))
  p.start()
  procs.append(p)
for n in range(num_trials):
  procs[n].join()

results = np.array(results)
min_results = np.min(results,axis=0)
avg_results = np.mean(results,axis=0)
max_results = np.max(results,axis=0)

json.dump(avg_results.tolist(),open("avg_random_"+str(num_trials*sub_proc_trials)+"sim.json","w"))
json.dump(min_results.tolist(),open("min_random_"+str(num_trials*sub_proc_trials)+"sim.json","w"))
json.dump(max_results.tolist(),open("max_random_"+str(num_trials*sub_proc_trials)+"sim.json","w"))
json.dump(this_train_sizes.tolist(),open("trainsize_random_"+str(num_trials*sub_proc_trials)+"sim.json","w"))
