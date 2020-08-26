import numpy as np
from numpy.random import randint
from time import time
from sorting import insertion_sort, merge_sort, quick_sort,selection_sort,radix_sort,radix_Sort
from numpy import mean
import pandas as pd
import matplotlib.pyplot as plt

size = [100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000]
#size = [10, 15, 20, 25]
df = pd.DataFrame(columns = size, index = ["selection_sort","insertion_sort","merge_sort","quick_sort","radix_sort"])
#size =1
# return a random array
def random_array(ArraySize):
    #np.random.seed(0)
    testArray = randint(1,ArraySize*2,ArraySize)
    return testArray
	
def benchmark(func):
    results=[]
    rowdata=[]
    #times = {f.__name__:[] for f in func}
    #
    array_size=[100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000]
    #array_size=[10,15,20 ,25]
        #if not results:
     #   print("%s\t" %(func.__name__), end=''),

    for i in array_size:
        res=0
        unsorted = list(random_array(i))
        #print("Unsorted",unsorted)
        #unsorted.sort()
        #unsorted.reverse()
        for k in range(10):
           # print("Unsorted",unsorted)
            start =time()
            sorted1=func(unsorted)
            end=time()
            res=end-start
            results.append(res)
           # print("Sorted:",sorted1)
            np.random.shuffle(unsorted)
            
        rowdata.append((round(mean(results)* 1000,3)))
        print(rowdata)
        results=[]
    df.loc[f.__name__]= rowdata
   
	
functions =[selection_sort,insertion_sort,merge_sort,quick_sort,radix_sort]

for f in functions:
    benchmark(f)
pd.set_option('display.max_columns',None)
with pd.option_context('display.float_format', '{:0.3f}'.format):
    print(df)
df2=df.T
df2.plot(kind='line')
plt.title("Time taken by sorting algorithms to sort list")
plt.xlabel("n-Input size")
plt.ylabel("T(n) in milliseconds")
plt.show()
#benchmark(selection_sort,insertion_sort)
#benchmark(insertion_sort)
#benchmark(merge_sort)
#benchmark(quick_sort)
#benchmark(radix_sort)
#benchmark(mergeSort)
