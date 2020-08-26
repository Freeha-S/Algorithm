from numpy.random import randint
from timeit import repeat
import math
#from numpy.random import randint
import numpy as np
import pandas as pd
from time import time
from numpy import mean


def run_sorting_algorithm(algorithm, array):
    # Set up the context and prepare the call to the specified
    # algorithm using the supplied array. Only import the
    # algorithm function if it's not the built-in `sorted()`.
    setup_code = f"from __main__ import {algorithm}" \
    #setup_code = f"from __main__import {algorithm}" 
    #\
     #   if algorithm != "sorted" else ""

    stmt = f"{algorithm}({array})"
    
    # Execute the code ten different times and return the time
    # in seconds that each execution took
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)

    # Finally, display the name of the algorithm and the
    # minimum time it took to run
    print(f"Algorithm: {algorithm}. average execution time: {round(min(times),3)}")
    return round(min(times),3)

         


#source:https://stackabuse.com/sorting-algorithms-in-python/#selectionsort
def selection_sort(alist):
    """
    Sorts a list using the selection sort algorithm.
    alist - The unsorted list.
    Examples
        selection_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    a selection sort looks for the smallest value as it makes a pass and, 
    after completing the pass, places it in the proper location
    Worst Case: O(n^2)
    Returns the sorted list.
    """
	
    # Traverse through all array elements
    for i in range(len(alist)):
        # assume that the first item of the unsorted segment is the smallest
        lowest_value_index = i
        # This loop iterates over the unsorted items
        for j in range(i + 1, len(alist)):
            if alist[j] < alist[lowest_value_index]:
                lowest_value_index = j
        # Swap values of the lowest unsorted element with the first unsorted
        # element
        alist[i], alist[lowest_value_index] = alist[lowest_value_index], alist[i]

    return alist


  
# source: https://realpython.com/sorting-algorithms-python/
def insertion_sort(alist):
	"""
    Public: Sorts a list using the insertion sort algorithm.
    alist - The unsorted list.
    Examples
        insertion_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    Worst Case: O(n^2)
    Returns the sorted list.
    """
    # Loop from the second element of the array until
    # the last element
	for index in range(1,len(alist)):
       # This is the element we want to position in its
        # correct place
		currentvalue = alist[index]
		# Initialize the variable that will be used to
        # find the correct position of the element referenced
        # by `currentvalue`
		position = index
    # Run through the list of items (the left
        # portion of the array) and find the correct position
        # of the element referenced by `currentvalue`. Do this only
        # if `currentvalue` is smaller than its adjacent values.
		while position>0 and alist[position-1]>currentvalue:
         # Shift the value one position to the left
			alist[position]=alist[position-1]
            # and reposition position to point to the next element
            # (from right to left)
			position = position-1
     # When you finish shifting the elements, you can position
        # `currentvalue` in its correct location
		alist[position]=currentvalue
    
	return alist
# source reference:https://realpython.com/sorting-algorithms-python/#the-merge-sort-algorithm-in-python
#source reference: https://stackabuse.com/sorting-algorithms-in-python/#mergesort
def merge(left, right):
    
	 # If the first array is empty, then nothing needs
    # to be merged, and you can return the second array as the result
    if len(left) == 0:
        return right

    # If the second array is empty, then nothing needs
    # to be merged, and you can return the first array as the result
    if len(right) == 0:
        return left

    result = []
    index_left = index_right = 0

    # Now go through both arrays until all the elements
    # make it into the resultant array
    while len(result) < len(left) + len(right):
        # The elements need to be sorted to add them to the
        # resultant array, so you need to decide whether to get
        # the next element from the first or the second array
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1

        # If you reach the end of either array, then you can
        # add the remaining elements from the other array to
        # the result and break the loop
        if index_right == len(right):
            result += left[index_left:]
            break

        if index_left == len(left):
            result += right[index_right:]
            break

    return result

def merge_sort(alist):

    """
    Sorts a list using the merge sort algorithm.
    alist - The unsorted list.
    Examples
        merge_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    Worst Case: O(n*Log(n))
    Returns the sorted list.
    """
    # If the list is a single element, return it
    if len(alist) <= 1:
        return alist

    # Use floor division to get midpoint, indices must be integers
    mid = len(alist) // 2

    # Sort and merge each half
    left_list = merge_sort(alist[:mid])
    right_list = merge_sort(alist[mid:])

    # Merge the sorted lists into a new one
    return merge(left_list, right_list)


# source:https://realpython.com/sorting-algorithms-python/#the-quicksort-algorithm-in-python
def quick_sort(array):
    # If the input array contains fewer than two elements,
    # then return it as the result of the function
    if len(array) < 2:
        return array

    low, same, high = [], [], []

    # Select a `pivot` element randomly
    pivot = array[randint(0, len(array) - 1)]

    for item in array:
        # Elements that are smaller than the `pivot` go to
        # the `low` list. Elements that are larger than
        # `pivot` go to the `high` list. Elements that are
        # equal to `pivot` go to the `same` list.
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)

    # The final result combines the sorted `low` list
    # with the `same` list and the sorted `high` list
    return quick_sort(low) + same + quick_sort(high)
#source reference:https://www.rosettacode.org/wiki/Sorting_algorithms/Radix_sort
def flatten(l):
	return [y for x in l for y in x]
 
def radix(l, p=None, s=None):
	if s == None:
		s = len(str(max(l)))
	if p == None:
		p = s
	
	i = s - p
	
	if i >= s:
		return l
		
	bins = [[] for _ in range(10)]
	
	for e in l:
		bins[int(str(e).zfill(s)[i])] += [e]
	
	return flatten([radix(b, p-1, s) for b in bins])
def counting_sort(arr, digit, radix):
    #"output" is a list to be sorted, radix is the base of the number system, digit is the digit
    #we want to sort by
    n=len(arr)
    #create a list output which will be the sorted list
    output = [0]*n
    count = [0]*int(radix)
    #counts the number of occurences of each digit in arr 
    for i in range(0, n):
        digit_of_Arri = int(arr[i]/radix**digit)%radix
        count[digit_of_Arri] = count[digit_of_Arri] +1 
        #now count[i] is the value of the number of elements in arr equal to i

    #this FOR loop changes cont to show the cumulative # of digits up to that index of count
    for j in range(1,radix):
        count[j] = count[j] + count[j-1]
        #here C is modifed to have the number of elements <= i
    for m in range(len(arr)-1, -1, -1): #to count down (go through A backwards)
        digit_of_Arri = int(arr[m]/radix**digit)%radix
        count[digit_of_Arri] = count[digit_of_Arri] -1
        output[count[digit_of_Arri]] = arr[m]

    return output

#alist = [9,3,1,4,5,7,7,2,2]
#print countingSort(alist,0,10)

def radix_sort(A):
    #radix is the base of the number system
    radix = 10
    #k is the largest number in the list
    k = max(A)
    #output is the result list we will build
    output = A
    #compute the number of digits needed to represent k
    digits = int(math.floor(math.log(k, radix)+1))
    #print(digits)
    for i in range(digits):
        output = counting_sort(output,i,radix)

    return output
def radix_Sort(nums):
    base=10
    result_list = []
    power = 0
    while nums:
        bins = [[] for _ in range(base)]# create 10 bins
        for x in nums:
            bins[x // base**power % base].append(x)#add the numbers to the bins
        nums = []
        for bin in bins:
            for x in bin:
                if x < base**(power+1):
                    result_list.append(x)#append numbers to the sorted list
                else:
                    nums.append(x)#append the remaing number in the unsorted list
        power += 1
   ## print(result_list)
    return result_list

if __name__ == "__main__":
    # Generate an array of `ARRAY_LENGTH` items consisting
    # of random integer values between 0 and 999
   # sort_functions_list = [selection_sort,insertion_sort,merge_sort,quick_sort,radix_sort]
    #sort_functions_list=[radix_sort]
    i=0
    size = [100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000]
    df = pd.DataFrame(columns = size, index = ["selection_sort","insertion_sort","merge_sort","quick_sort","radix_sort","radix_Sort"])
    for ArraySize in (100, 250, 500,750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000):
        results=[]
  # create a randowm array of values to use in the tests below
    
        #array = [randint(1,ArraySize*2)for i in range(ArraySize)]
        array = [randint(0, 1000) for i in range(ArraySize)]
        results.append(ArraySize)
        print (ArraySize)
    #array = [randint(0, 1000) for i in range(ARRAY_LENGTH)]
        df.loc["selection_sort",ArraySize] =run_sorting_algorithm(algorithm="selection_sort", array=array)
        df.loc["insertion_sort",ArraySize]=run_sorting_algorithm(algorithm="insertion_sort", array=array)
        df.loc["merge_sort",ArraySize]=run_sorting_algorithm(algorithm="merge_sort", array=array)
        df.loc["quick_sort",ArraySize]=run_sorting_algorithm(algorithm="quick_sort", array=array)
        df.loc["radix_sort",ArraySize]=run_sorting_algorithm(algorithm="radix_sort", array=array)
        df.loc["radix_Sort",ArraySize]=run_sorting_algorithm(algorithm="radix_Sort", array=array)
     #df.loc[[0,3],'Z'] =
        i=1+1
    print()
    print("Average time taken to for each array size") 
    pd.set_option('display.max_columns',None)
    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print(df)
        df.to_csv(r"D:\cta2020\Sorting Project\resultsort.csv",header=True)