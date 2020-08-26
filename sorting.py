import math
from random import randint
import math
from numpy.random import randint
import numpy as np
import pandas as pd
from time import time
from boto.sdb.db import key



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
    #print(alist)
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

"""
    Public: Sorts a list using the insertion sort algorithm.
    alist - The unsorted list.
    Examples
        insertion_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    Worst Case: O(n^2)
    Returns the sorted list.
    """
def insertion_sort(arr):
    for i in range(1, len(arr)): # Traverse through 1 to len(arr) 
  
        key = arr[i] 
  
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j] #shifting the element
                j -= 1
        arr[j+1] = key #When you finish shifting the elements, you can put key  in its correct location
    return arr
        

"""
    Sorts a list using the merge sort algorithm.Merge sort is a recursive algorithm that continually splits a list in half
    If the list is empty or has one item, it is sorted by definition (the base case)
    If the list has more than one item, we split the list and recursively invoke a merge sort on both halves.
    Once the two halves are sorted, the fundamental operation, called a merge, is performed. 
    alist - The unsorted list.
    Examples
        merge_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    
    Worst Case: O(n*Log(n))
    Returns the sorted list.
    """
# source reference:https://realpython.com/sorting-algorithms-python/#the-merge-sort-algorithm-in-python
#source reference: https://stackabuse.com/sorting-algorithms-in-python/#mergesort
 
def merge_sort(alist):
   
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




# source:https://realpython.com/sorting-algorithms-python/#the-quicksort-algorithm-in-python
"""
    Sorts a list using the quick sort algorithm.a quick sort choose a pivot point, and partitioning the collection around the pivot,
    so that elements smaller than the pivot are before it,
    and elements larger than the pivot are after it.
    It continues to choose a pivot point and break down the collection into single-element lists,
    before combing them back together to form one sorted list.
    alist - The unsorted list.
    Examples
        quick_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    
    Worst Case: O(n^2)
    
    Returns the sorted list.
    """
def quick_sort(alist):
   
    # If the input array contains fewer than two elements,
    # then return it as the result of the function
    if len(alist) < 2:
        return alist

    low, same, high = [], [], []

    # Select a `pivot` element randomly
    pivot = alist[randint(0, len(alist) - 1)]

    for item in alist:
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
"""
    Sorts a list using the radix sort algorithm.Radix sort is non comparative sorting method .it is an integer sorting algorithm,
    that sorts by grouping numbers by their individual digits (or by their radix).
    It uses each radix/digit as a key, and implements counting sort under the hood in order to do the work of sorting.
    A - The unsorted list.
    Examples
        radix_sort([4,7,8,3,2,9,1])
        # => [1,2,3,4,7,8,9]
    Worst Case: O(kn))
    n is the number of elements and 
    k is the number of digits required to represent largest element in the array
    Returns the sorted list.
    """
def radix_sort(A):
    #radix is the base of the number system
    radix = 10
    #k is the largest number in the list
    k = max(A)
    #output is the result list we will build
    output = A
    #compute the number of digits needed to represent k
    digits = int(math.floor(math.log(k, radix)+1))
    #print(digits)# to check 
    for i in range(digits):
        output = counting_sort(output,i,radix)
        #print("Pass",i)
        #print(output) #to look at how the array is sorting in every pass

    return output
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
        #print(digit_of_Arri)
        count[digit_of_Arri] = count[digit_of_Arri] +1 
        #now count[i] is the value of the number of elements in arr equal to i
    #print(count) to look at the count list
    #this FOR loop changes cont to show the cumulative # of digits up to that index of count
    for j in range(1,radix):
        count[j] = count[j] + count[j-1]
        #here count is modifed to have the number of elements <= i
    #print(count) to check working
    for m in range(len(arr)-1, -1, -1): #to count down (go through arr backwards)
        digit_of_Arri = int(arr[m]/radix**digit)%radix
        
        count[digit_of_Arri] = count[digit_of_Arri] -1
        output[count[digit_of_Arri]] = arr[m]
        #print(output) to check working

    return output
#https://stackoverflow.com/a/35421603
#radix sort algorithm using buckets
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


    

#source reference:https://www.rosettacode.org/wiki/Sorting_algorithms/Radix_sort
# i am  not using this  radix algoritm
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
#print([10,52,5,209,19,44,6,78,98,12])
#print("Sorted:",insertion_sort([195, 20, 115, 199, 3, 95, 83, 116, 132, 17, 158, 133, 147, 105, 150, 46, 49, 86, 148, 164, 77, 105, 142, 20, 105, 56, 37, 174, 67, 12, 75, 39, 176, 185, 30, 142, 75, 62, 174, 57, 169, 15, 133, 57, 107, 52, 77, 37, 9, 73, 137, 33, 22, 171, 148, 113, 148, 23, 43, 79, 41, 154, 22, 84, 109, 99, 40, 5, 80, 169, 188, 199, 128, 38, 81, 162, 196, 191, 26, 134, 69, 37, 2, 62, 186, 52, 138, 178, 88, 90, 180, 14, 2, 52, 99, 168, 169, 161, 22, 58]))