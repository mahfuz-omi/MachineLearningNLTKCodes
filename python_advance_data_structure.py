#Python has two main built-in numeric classes that implement
# the integer and floating point data types.
# These Python classes are called int and float.
# The standard arithmetic operations, +, -, *, /, and ** (exponentiation),
#  can be used with parentheses forcing the order of operations away from normal operator precedence.
# Other very useful operations are the remainder (modulo) operator, %, and integer division, //.
#  Note that when two integers are divided, the result is a floating point.
# The integer division operator returns the integer portion of the quotient by truncating any fractional part.
import math

print(2+3*4)
print((2+3)*4)
print(2**10)
print(6/3)
print(7/3)
print(7//3)
print(7%3)
print(3/6)

# integer division- cast to int after division
print(3//6)
print(3%6)
print(2**100)

# List operations:
# Operation Name	Operator	Explanation
# indexing	[ ]	Access an element of a sequence
# concatenation	+	Combine sequences together
# repetition	*	Concatenate a repeated number of times
# membership	in	Ask whether an item is in a sequence
# length	len	Ask the number of items in the sequence
# slicing	[ : ]	Extract a part of a sequence

list1 = ['omi',5,True,30.0]
print(list1)
print(type(list1))
print(type('omi'))
print(type(5))
print(type(5.0))

# <class 'list'>
# <class 'str'>
# <class 'int'>
# <class 'float'>

myList = [0] * 6
print(myList)

#[0, 0, 0, 0, 0, 0]

myList = [1,2,3,4]
A = myList*3
print(A)

# [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

myList = [1,2,3,4]
A = [myList]*3
print(A)

# [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

# Method Name	Use	Explanation
# append	alist.append(item)	Adds a new item to the end of a list
# insert	alist.insert(i,item)	Inserts an item at the ith position in a list
# pop	alist.pop()	Removes and returns the last item in a list
# pop	alist.pop(i)	Removes and returns the ith item in a list
# sort	alist.sort()	Modifies a list to be sorted
# reverse	alist.reverse()	Modifies a list to be in reverse order
# del	del alist[i]	Deletes the item in the ith position
# index	alist.index(item)	Returns the index of the first occurrence of item
# count	alist.count(item)	Returns the number of occurrences of item
# remove	alist.remove(item)	Removes the first occurrence of item

# for i in range(10):
#     print(i)
#     print('\n')

print(list(range(1,10,2)))
print(list(range(10,1,-1)))

#A major difference between lists and strings is that lists can be modified while strings cannot.
# This is referred to as mutability.
# Lists are mutable; strings are immutable.
#  For example, you can change an item in a list by using indexing and assignment.
#  With a string that change is not allowed.

name = 'omi'
country = 'bangladesh'

# splitted is a list of string
splitted = country.split('g')
print(splitted)

#Tuples are very similar to lists in that they are heterogeneous sequences of data.
# The difference is that a tuple is immutable, like a string.
# A tuple cannot be changed. Tuples are written as comma-delimited values enclosed in parentheses.
#  As sequences, they can use any operation described above.

myTupple = ('omi',1,3.0)
print(myTupple)

print(type(myTupple[0]))

#<class 'str'>
#A set is an unordered collection of zero or more immutable Python data objects.
# Sets do not allow duplicates and are written as comma-delimited values enclosed in curly braces.
# The empty set is represented by set(). Sets are heterogeneous.

set = set()
set.add('3')
set.add('omi')
print(set)

print(type(set))

# {'omi', '3'}
# <class 'set'>

# Operation Name	Operator	Explanation
# membership	in	Set membership
# length	len	Returns the cardinality of the set
# |	aset | otherset	Returns a new set with all elements from both sets
# &	aset & otherset	Returns a new set with only those elements common to both sets
# -	aset - otherset	Returns a new set with all items from the first set not in second
# <=	aset <= otherset	Asks whether all elements of the first set are in the second

# Method Name	Use	Explanation
# union	aset.union(otherset)	Returns a new set with all elements from both sets
# intersection	aset.intersection(otherset)	Returns a new set with only those elements common to both sets
# difference	aset.difference(otherset)	Returns a new set with all items from first set not in second
# issubset	aset.issubset(otherset)	Asks whether all elements of one set are in the other
# add	aset.add(item)	Adds item to the set
# remove	aset.remove(item)	Removes item from the set
# pop	aset.pop()	Removes an arbitrary element from the set
# clear	aset.clear()	Removes all elements from the set

capitals = {'Iowa':'DesMoines','Wisconsin':'Madison'}

# set
keys = capitals.keys()
for key in keys:
    print(capitals[key])

# Operator	Use	Explanation
# []	myDict[k]	Returns the value associated with k, otherwise its an error
# in	key in adict	Returns True if key is in the dictionary, False otherwise
# del	del adict[key]	Removes the entry from the dictionary

print(capitals.keys())
print(capitals.values())
print(capitals.items())
print(capitals.get('Iowa2','default'))

# user input from console
# read until enter pressed
# name = input('enter your name\n')
# print(name)

print('omi','popy','sumi',sep='--')

print('kdkxkkl',end='[]omi\n')

name = 'omi'
age = 28

print("%s is %d years old." % (name, age))

sqlist=[]
for x in range(1,11):
    sqlist.append(x*x)

print(sqlist)

# list comprehension
sqlist_new = [x*x for x in range(1,11)]
print(sqlist_new)

sqlist_new = [x*x for x in range(1,11) if x%2 != 0]
print(sqlist_new)

char = [ch.upper() for ch in 'comprehension' if ch not in 'aeiou']
print(char)

char = []
for ch in 'comprehension':
    if(ch not in 'aeiou'):
        char.append(ch.upper())

# exceptions
anumber = int(input("Please enter an integer"))

try:
    print(math.sqrt(anumber))
except:
    print("Bad Value for square root")
    print("Using absolute value instead")
    print(math.sqrt(abs(anumber)))

#25
# raise RuntimeError('hello... this is omi error :p')

from fraction import Fraction
fraction = Fraction(12,20)
print(fraction)
print(fraction.printFraction())