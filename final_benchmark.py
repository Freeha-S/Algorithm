import numpy as np
from numpy.random import randint
from time import time
from sorting import insertion_sort, merge_sort, quick_sort,selection_sort,radix_sort,radix_Sort
from numpy import mean
import pandas as pd
import matplotlib.pyplot as plt
import math


"""
def random_array(ArraySize):
    numpy.random.seed(0)
	testArray = randint(1,ArraySize*2,ArraySize)
	return testArray
"""	
def benchmark(func,array):
    results=[]
    unsorted=list(array)
    #print("unsorted",unsorted)# to check results
    for j in range(0,10):
    #unsorted = random_array(i)
        start =time() #start time
        sorted1= list(func(unsorted))
        end=time()#end time
        res=end-start #time elapsed
        np.random.shuffle(unsorted)#shuffle array for next run
        #print("sorted",sorted1)# to cofirm if it is working fine
        results.append(res)
    print(results)
    return round(mean(results)*1000,3)# return the average time in milliseconds and rounded to 3 didgits

if __name__ == "__main__":
    
    # create a dataframe
    size = [100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000]
    df = pd.DataFrame(columns = size, index = ["selection_sort","insertion_sort","merge_sort","quick_sort","radix_sort","radix_Sort"])
    
    for ArraySize in (100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000):
        results=[]
  # create a randowm array of values to use in the tests below
    
        array1 = [randint(1,ArraySize*2)for i in range(ArraySize)]
        #array = [randint(0, 1000) for i in range(ArraySize)]
        results.append(ArraySize)
        #print (ArraySize)
    #call the benchmark function and put the returned value in appropriate row in dataframe
        df.loc["selection_sort",ArraySize] = benchmark(selection_sort,array1)
        df.loc["insertion_sort",ArraySize] = benchmark(insertion_sort,array1)
        df.loc["merge_sort",ArraySize]= benchmark(merge_sort,array1)
        df.loc["quick_sort",ArraySize]= benchmark(quick_sort,array1)
        df.loc["radix_sort",ArraySize]=benchmark(radix_sort,array1)
        df.loc["radix_Sort",ArraySize]=benchmark(radix_Sort,array1)
        
     #df.loc[[0,3],'Z'] =used this as reference how to fill dataframe
       
    print()
    print("Average time taken for 10 runs for each array size") 
    pd.set_option('display.max_columns',None)
    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print(df)
        df.to_csv("resultsort.csv",header=True)
    df2=df.T
    df2.plot(kind='line')
    plt.title("Average Time taken by sorting algorithms to sort list")
    plt.xlabel("n-Input size")
    plt.ylabel("T(n) in milliseconds")
    plt.show()
