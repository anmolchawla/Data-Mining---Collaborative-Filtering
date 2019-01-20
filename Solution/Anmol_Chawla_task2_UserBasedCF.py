# Task 2

def dif(a,b,user,movie):
  d = a-b
  c = d**2
  return c,d,((user,movie),b)
  

    

def pearsons(data):
  global pairs
  global user_dict
  user_dict = defaultdict(dict)
  table = defaultdict(dict)
  all_users = []
  for i in data:
    all_users.append(i[0])
    user_dict[i[0]] = i[1]
                     
      
  pairs = list(itertools.combinations(sorted(all_users),2))
  
  for j in pairs:
    add = 0
    add1 = 0
    hold = 0
    one = defaultdict(dict)
    two = defaultdict(dict)
    lt1 = user_dict[j[0]]
    lt2 = user_dict[j[1]]
     
    for i in lt1:
      one[i[0]] = i[1]
    
    for i in lt2:
      two[i[0]] = i[1]
      
    common = [i for i in one if i in two]
    
    
    if common!= []:
      len_common = len(common)
      
      for i in common:
        add  = add  + one[i]
        add1 = add1 + two[i]

      avg_one = add/len_common
      avg_two = add1/len_common

      num = sum( [(one[i]-avg_one)*(two[i]-avg_two) for i in common]  )
      den = math.sqrt(sum([(one[i]-avg_one)**2 for i in common])) * math.sqrt(sum([(two[i]-avg_two)**2 for i in common]))
      
      if den!= 0:
        relation = float(num)/den
        hold = (relation,avg_one,avg_two)
      else:
        hold = (0,avg_one,avg_two)
      
    else:
   	  hold = (0,0,0)
  

    table[j] = [hold]


  return table
  

    
    
def memory_user_cf(data):
  global pearson_table
  global user_train
  global items_list
  global avg_table
  den = []
  num = []
  item = data[1]
  user = data[0]
  all_users =[]
  pairs = []
  
 
  other_users = [i[0] for i in items_list[item]]
  
 
  
  for i in sorted(other_users):
    hold = sorted((user,i))
    pairs.append(tuple(hold))


  for i in pairs:
    val = pearson_table[i]
    if i[0] == user:
      a = val[0][2]
      for tup in items_list[item]:
        if tup[0] == i[1]:
          b = tup[1]  

    else:
      a = val[0][1]
      for tup in items_list[item]:
        if tup[0] == i[0]:
          b = tup[1]
    
    c = (b-a)*val[0][0]
    
    
    den.append(abs(val[0][0]))
    
    num.append(c)
    

  deno = sum(den)
  avg = avg_table[user]
  numo = sum(num)
  prediction = 0
  if deno != 0:
    prediction = (numo/deno) + avg
  else:
    prediction = avg
    
  return ((user,item), prediction) 

# imports and dependencies
from collections import defaultdict
import math
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
import sys
import os
from itertools import chain, combinations
from operator import add
from itertools import izip
from pyspark.sql import SparkSession
import itertools
import time
import timeit
import math
from collections import OrderedDict
from collections import Counter


# Setup
sc = SparkContext('local[*]','cf')


# Input arguments
alldata = sys.argv[1]
testingdata = sys.argv[2]
filename = "Anmol_Chawla_UserBasedCF.txt"



start = timeit.default_timer()

#Load data
entiredata = sc.textFile(alldata)
testdata = sc.textFile(testingdata)

# MODEL Based CF
#Pre-process data starts here ----------------------------------------------------------------------------------
# test data without lable
top_test = testdata.first()
testing_data_nolabel = testdata.filter(lambda x : x != top_test).map(lambda line:line.split(",")).map(lambda x: ( (int(x[0]),int(x[1])), "0" ) )
testing = testing_data_nolabel.map(lambda x: (int(x[0][0]),int(x[0][1]))) ###############################################################################################################################
#print(testing_data_nolabel.take(2))
#print("testing len",len(testing_data_nolabel.collect()) )


# Entire data set
top_all = entiredata.first()
full_data = entiredata.filter(lambda x : x != top_all).map(lambda line:line.split(",")).map(lambda x: (  ( int(x[0]),int(x[1])) , float(x[2]))  )  
#print(full_data.take(2))
#print("full len", len(full_data.collect()) )

# TRAINING data with labels
training_data = full_data.subtractByKey(testing_data_nolabel)
train = training_data.map(lambda x: (x[0][0], x[0][1], float(x[1]) ) )
#print(train.take(2))
#print("train len", len(training_data.collect()))

# Testing data with labels
#testing_data_label = full_data.subtractByKey(training_data)
#testing_labeled = testing_data_label.map(lambda x: (  (int(x[0][0]), int(x[0][1])), float(x[1])  )   )
#print(testing_data_label.take(2))
#print("test lable len", len(testing_data_label.collect()))


# Pre-process data stops here --------------------------------------------------------------------------------------------





# Task 2 part 2 starts -------------------------------------------------------------------------------------------------------- 

user_train = train.map(lambda x: (x[0],(x[1],x[2]))).groupByKey().map(lambda x: (x[0],list(x[1]))).collect()
print("user_train",user_train[:10])

# Average of all users
avg_table = defaultdict(dict)
for i in user_train:
  add = []
  for k in i[1]:
    add.append(k[1])
  
  aver = sum(add) / len(add)  
  
  avg_table[i[0]] = aver

#print("avg table",avg_table)

# Items with all the users having rated them
items = train.map(lambda x: (x[1], (x[0],float(x[2])) ) ).groupByKey().map(lambda x: (x[0],list(x[1]))).collect()

items_list = defaultdict(dict)
for i in items:
  items_list[i[0]] = i[1]
    

#print("items with users",items[:10])

# Pre-calculate pearson table
pearson_table = pearsons(user_train)
#print("pearson table", pearson_table[(4,512)])


#Same as the other code
preds = testing.map(lambda x: memory_user_cf(x))#.flatMap(lambda x:x)
#print("predicted",preds.take(10))
#print("to be joined", full_data.take(10))
stop = timeit.default_timer()

ratesAndPreds = full_data.join(preds)
#print("joined true, pred", ratesAndPreds.take(10))

#Computing absolute differnece
RMSE = ratesAndPreds.map(lambda r: dif(r[1][0],r[1][1],r[0][0],r[0][1]) ).collect()
#print("squared and diff",RMSE[:10])

add = 0
one = 0
two = 0
three = 0
four = 0
five = 0
pred = []
for i in RMSE:
  pred.append(i[2])
  add = add + float(i[0])
  d = abs(i[1])
  if (d>= 0 and d<1):
    one = one + 1
  elif (d>=1 and d<2):
    two = two +1
  elif (d>=2 and d<3):
    three = three +1
  elif (d>=3 and d<4):
    four = four +1
  else:
    five = five +1
  
  
rmse = math.sqrt(add/len(RMSE))



print(">=0 and < 1: {}".format(one))
print(">=1 and < 2: {}".format(two))
print(">=2 and < 3: {}".format(three))
print(">=3 and < 4: {}".format(four))
print(">=4: {}".format(five))
print("RMSE: {}".format(rmse))
print("Time: {}".format(stop - start))



pred = sorted(pred, key = lambda x:x[0])

with open(filename, 'w') as caseop:
  for p in pred:
    user = p[0][0]
    movie = p[0][1]
    pred  = p[1]
    caseop.write("{}, {}, {}".format(user,movie,pred))
    caseop.write("\n")



# # Task 2 part 2 ends -------------------------------------------------------------------------------------------------------- 




# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx - Code Ends - xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx