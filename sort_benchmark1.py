import numpy as np
from numpy.random import randint
from time import time

from sorting import insertion_sort, merge_sort, quick_sort,selection_sort,radix_sort
from numpy import mean

size =1
def random_array(ArraySize):
	
	testArray = randint(1,ArraySize*2,ArraySize)

	return testArray
	
def benchmark(func):

	results=[]

	n = [100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000]
	global size
	while size==1:
		print("Size         ", end=' ')
		size=2
		for j in n:
			print("%7d"%(j), end=' '),
		print()
	if not results:
		print("%s\t" %(func.__name__), end=''),
	for i in n:
		unsorted = random_array(i)
		start =time()
		func(unsorted)
		end=time()
		res=end-start
		results.append(res)
		#print("%s(%d)   \t %.4gs" % (func.__name__, i, toc()))
	for r in results:
		print("%6.3f " %(float(r)),end=' ')
	print()


benchmark(selection_sort)
benchmark(insertion_sort)
benchmark(merge_sort)
benchmark(quick_sort)
benchmark(radix_sort)
#benchmark(mergeSort)
