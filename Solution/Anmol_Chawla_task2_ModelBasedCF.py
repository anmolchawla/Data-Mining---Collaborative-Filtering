# Task 2

def dif(a,b,user,movie):
  d = a-b
  c = d**2
  return c,d,((user,movie),b)
  

    
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
# Setup
sc = SparkContext('local[*]','cf')




# Input arguments
alldata = sys.argv[1]
testingdata = sys.argv[2]
filename = "Anmol_Chawla_ModelBasedCF.txt"





#Load data
entiredata = sc.textFile(alldata)
testdata = sc.textFile(testingdata)


# MODEL Based CF
#Pre-process data starts here ----------------------------------------------------------------------------------
# test data without lable
top_test = testdata.first()
testing_data_nolabel = testdata.filter(lambda x : x != top_test).map(lambda line:line.split(",")).map(lambda x: ( (int(str(x[0])),int(str(x[1]))), 6 ) )
testing = testing_data_nolabel.map(lambda x: (  x[0][0],x[0][1]   )  )
#print(testing_data_nolabel.take(2))
#print("testing len",len(testing_data_nolabel.collect()) )


# Entire data set
top_all = entiredata.first()
full_data = entiredata.filter(lambda x : x != top_all).map(lambda line:line.split(",")).map(lambda x: (  ( int(str(x[0])),int(str(x[1]))) , float(str(x[2])))  )  
#print(full_data.take(2))
#print("full len", len(full_data.collect()) )

# TRAINING data with labels
training_data = full_data.subtractByKey(testing_data_nolabel)
train = training_data.map(lambda x: ( x[0][0], x[0][1], x[1] ) )
#print(training_data.take(2))
#print("train len", len(training_data.collect()))

# Testing data with labels
#testing_data_label = full_data.subtractByKey(training_data)
#testing_labeled = testing_data_label.map(lambda x: (  (int(x[0][0]), int(x[0][1])), float(x[1])  )   )
#print(testing_data_label.take(2))
#print("test lable len", len(testing_data_label.collect()))


# Pre-process data stops here --------------------------------------------------------------------------------------------



# Task 2 part 1 ----------------------------------------------------------------------------------------------------------
co = testing.count()
# Build the recommendation model using Alternating Least Squares 20500


if co <= 20500:
  rank = 5
  numIterations = 12

else:
  print("this is change")
  rank = 5
  numIterations = 20

#   rank = 10
#   numIterations = 10f
# else:
#   rank = 5
#   numIterations = 20




model = ALS.train(train, rank, numIterations)


# # Predicting
predictions = model.predictAll(testing).map(lambda r: ((r[0], r[1]), r[2]))




# #print("predictions",predictions.take(5))
# #print("to be joined",testing_labeled.take(5))

ratesAndPreds = full_data.join(predictions)


# # #print("joined true, pred", ratesAndPreds.take(10))
# # #print("length of predicted", predictions.count())

# # #Computing absolute differnece
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
  #print("rmse one",i)
  pred.append(i[2])
  add = add + float(i[0])
  d = abs(float(i[1]))
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
  
  
rmse = math.sqrt((add/len(RMSE)))


print(">=0 and < 1: {}".format(one))
print(">=1 and < 2: {}".format(two))
print(">=2 and < 3: {}".format(three))
print(">=3 and < 4: {}".format(four))
print(">=4: {}".format(five))
print("RMSE: {}".format(rmse))


pred = sorted(pred, key = lambda x:x[0])

print("Success LOL")

with open(filename, 'w') as caseop:
  for p in pred:
    user = p[0][0]
    movie = p[0][1]
    pred  = p[1]
    caseop.write("{}, {}, {}".format(user,movie,pred))
    caseop.write("\n")



# Task 2 part 1 ends -------------------------------------------------------------------------------------------------------- 



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx - Code Ends - xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx