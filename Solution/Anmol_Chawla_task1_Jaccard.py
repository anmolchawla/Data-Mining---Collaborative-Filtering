def charmatrix(data):
    global movies
    op = []
    data = list(data)
    user         = data[0]
    usermovies   = list(data[1])
  
    for movie in movies:
        if movie in usermovies:
            op.append(1)
        else:
            op.append(0)
   
    return (user,op)
  
def hashing(data):
    data = list(data)
    user = data[0]
    user_movie_list = list(data[1])
    for i in range(10):
        a = randint(1,9066)
        b = randint(1,9066)
        hold = ((a*user) + b)  % 671
        for j in range(len(user_movie_list)):
            if user_movie_list[j] == 1:
                yield ((i,j),hold)


def getHash(iterator):
    # rearrange the (movie, minHash) tuple to uniformly sorted minHashes
    iterator = list(iterator)
    hashindex = iterator[0]
    list_of_tuples = list(iterator[1])
    return_list = [0 for i in range(len(list_of_tuples))]
    for tup in list_of_tuples:
        return_list[tup[0]] = tup[1]
    
    return (hashindex, return_list)


def canidate(data):
    global movies
    columns = []
    data = list(data)
    row1 = data[0]
    row2 = data[1]
    for i in range(len(movies)):
        hold = (movies[i],row1[1][i],row2[1][i])
        columns.append(hold)
  
    for i in range(0,len(columns)-1):
        for j in range(i+1,len(columns)):
            if (  (columns[i][1] == columns[j][1]) and (columns[i][2] == columns[j][2])  ):
                temp = (columns[i][0],columns[j][0])
                yield temp

def jaccard(index, data):
    data = list(data)
    global canidates
    global movies_dict
    user = []
    #print("index {} length {}".format(index, len(data)))
    #print("index {} can length {}".format(index, len(canidates)))
        
    count = 0
    for j in canidates:
        count+=1
        intersection = 0
        union = 0
        for i in data:
            a = movies_dict[j[0]]
            b = movies_dict[j[1]]
            one = i[1][a]
            two = i[1][b]
            c = one + two
            if c == 2:
                union += 1
                intersection += 1
                #yield (j,(1,1))
      
            elif c == 1:
                union += 1
                
        if union != 0:
          yield (j,(intersection, union))
        
        if index == 1:
            if count % 20000 == 0:
                print(count)
                #yield (j,(0,1))
                
            
            
from pyspark import SparkContext
from collections import defaultdict
from collections import OrderedDict
import sys
import os
from itertools import chain, combinations
from operator import add
from itertools import izip
from pyspark.sql import SparkSession
import itertools
import time
import math
from collections import OrderedDict
from collections import Counter
from random import randint




# Setup
sc = SparkContext('local[*]','cf')
# Input arguments
ipfile = sys.argv[1]
filename = "Anmol_Chawla_SimilarMovies_Jaccard.txt"




start = time.time()
data = sc.textFile(ipfile)
top  = data.first()
data = data.filter(lambda x : x != top).map(lambda line:line.split(",")).map(lambda x:(int(str(x[0])),int(str(x[1]))))
#print("ip data",data.take(5))
#print("ip data len",data.count())


# Unique Movies # 9066 movies
movies = sorted(data.map(lambda x: x[1]).distinct().collect())
#print("unique movies",movies[:32])
#print("lenght movies",len(movies))


movies_dict = dict([(i, index) for index, i in enumerate(movies)])



# User with movies 671 unique users
user= data.groupByKey().sortByKey(ascending=True).map(lambda x: (x[0], sorted(list(x[1])) ) )
#print("user set",user.take(5))
#print("user set len",user.count())



# Char Matrix
usermovie = user.map(charmatrix)
#print("char matrix",usermovie.take(5))
#print("char matrix columns",len((usermovie.take(1))[0][1]))
#print("char matrix rows",usermovie.count())


hashvals = usermovie.map(lambda x: hashing(x)).flatMap(lambda x: x).reduceByKey(lambda x,y: x if x<y else y).map(lambda x:(x[0][0], (x[0][1], x[1]))).groupByKey().map(getHash).collect()#.flatMap(lambda x:x).map(lambda x:x[1]).collect()
# print("hashed values",hashvals[:5])
# print("hashed rows",(len(hashvals)))
# print("len of columns", len(hashvals[0]) )



#Band and row divsion
bands = sc.parallelize(hashvals,5)
canidates = bands.mapPartitions(canidate).collect()
#print("can",canidates[:10])
#print("cani",len(canidates))


freq = usermovie.mapPartitionsWithIndex(jaccard).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda (x,y): float(x)/y).filter(lambda x: x[1]>=0.5)
pred = freq.collect()

stop = time.time()
print("Time: {}".format(stop-start))

pred = sorted(pred, key = lambda x:x[0])

with open(filename, 'w') as caseop:
  for p in pred:
    movie1 = p[0][0]
    movie2 = p[0][1]
    jack  = p[1]
    caseop.write("{}, {}, {}".format(movie1,movie2,jack))
    caseop.write("\n")
