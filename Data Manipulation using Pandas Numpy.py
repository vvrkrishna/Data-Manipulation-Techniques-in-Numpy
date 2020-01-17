#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[3]:


arr=np.array([[1,-8,9],[2,-8.9,13]],dtype=np.float32)


# In[4]:


print(arr)


# >When the elements of a NumPy array are mixed types, then the array's type will be upcast to the highest level type.
# If Array input has mixed int and float elements, all the integers will be cast to their floating-point equivalents. If an array is mixed with int, float, and string elements, everything is cast to strings.

# In[6]:


arr1=np.array([[1,"cat",9.8],[2,-8.9,"ball"]])


# In[9]:


print(repr(arr1))


# #Copying Arrays

# >In the code example below, c is a reference to a while d is a copy. Therefore, changing c leads to the same change in a,while changing d does not change the value of b.

# In[10]:


a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))


# In[11]:


d = b.copy()
d[0] = 6
print('Array b: {}'.format(repr(b)))


# #Casting Arrays
# The code below shows an example of casting using the astype function. The dtype property returns the type of an array.

# In[12]:


arr = np.array([0, 1, 2])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)


# #Dealing with Missing Values

# >When we don't want a NumPy array to contain a value at a particular index, we can use np.nan to act as a placeholder. A common usage for np.nan is as a filler value for incomplete data.
# The code below shows an example usage of np.nan. Note that np.nan cannot take on an integer type.

# In[3]:


arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

# Will result in a ValueError
np.array([np.nan, 1, 2], dtype=np.int32)


# #Representing Infinity in Numpy Arrays

# In[15]:


print(np.inf > 1000000)


# In[16]:


arr = np.array([np.inf, 5])
print(repr(arr))


# In[17]:


arr = np.array([-np.inf, 1])
print(repr(arr))


# In[18]:


# Will result in an OverflowError
np.array([np.inf, 3], dtype=np.int32)


# In[ ]:


#Set float_arr equal to np.array applied to a list with elements 1, 5.4, and 3, in that order.


# In[ ]:


float_arr=np.array([1,5.4,3])


# In[ ]:


#Set float_arr2 equal to arr2.astype, with argument np.float32.


# In[ ]:


float_arr2=arr2.astype(dtype=np.float32)


# In[ ]:


#Creating 2-D matrix with integers 1, 2, 3 in its first row, and the integers 4, 5, 6 in its second row andmanually set its type to np.float32.


# In[ ]:


matrix=np.array([[1,2,3],[4,5,6]],dtype=np.float32)


# >Dealing with Ranged Data in Numpy Arrays

# If only a single number, n, is passed in as an argument, np.arange will return an array with all the integers in the range [0, n). Note: the lower end is inclusive while the upper end is exclusive.

# In[19]:


arr = np.arange(5)
print(repr(arr))


# In[22]:


arr = np.arange(5.1)
print(repr(arr))


# For two arguments, m and n, np.arange will return an array with all the integers in the range [m, n).

# In[23]:


arr = np.arange(-1, 4)
print(repr(arr))


# For three arguments, m, n, and s, np.arange will return an array with the integers in the range [m, n) using a step size of s.

# In[24]:


arr = np.arange(-1.5, 4, 2)
print(repr(arr))


# >To specify the number of elements in the returned array, rather than the step size, we can use the np.linspace function.

# In[30]:


arr = np.linspace(5,11,num=4)
print(repr(arr))


# >This function takes in a required first two arguments, for the start and end of the range, respectively. The end of the range is inclusive for np.linspace, unless the keyword argument endpoint is set to False. To specify the number of elements, we set the num keyword argument (its default value is 50).

# In[31]:


arr = np.linspace(5, 11, num=4, endpoint=False)
print(repr(arr))


# >The code below shows example usages of np.linspace. It also takes in the dtype keyword argument for manual casting.

# In[33]:


arr = np.linspace(5, 11, num=4, dtype=np.int32)
print(repr(arr))


# In[34]:


###Reshaping Data


# >The function we use to reshape data in NumPy is np.reshape. It takes in an array and a new shape as required arguments. The new shape must exactly contain all the elements from the input array. For example, we could reshape an array with 12 elements to (4, 3), but we can't reshape it to (4, 4).
# 
# >We are allowed to use the special value of -1 in at most one dimension of the new shape. The dimension with -1 will take on the value necessary to allow the new shape to contain all the elements of the array.
# 
# >The code below shows example usages of np.reshape.

# In[5]:


arr = np.arange(8)


# In[6]:


reshaped_arr = np.reshape(arr, (2, 4))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))


# In[7]:


reshaped_arr = np.reshape(arr, (-1, 2, 2))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))


# >While the np.reshape function can perform any reshaping utilities we need, NumPy provides an inherent function for flattening an array. Flattening an array reshapes it into a 1D array. Since we need to flatten data quite often, it is a useful function.
# 
# >The code below flattens an array using the inherent flatten function

# In[8]:


arr = np.arange(8)
arr = np.reshape(arr, (2, 4))
flattened = arr.flatten()
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(flattened))
print('flattened shape: {}'.format(flattened.shape))


# >Similar to how it is common to reshape data, it is also common to transpose data. Perhaps we have data that's supposed to be in a particular format, but some new data we get is rearranged. We can just transpose the data, using the np.transpose function, to convert it to the proper format.
# 
# >The code below shows an example usage of the np.transpose function. The matrix rows become columns after the transpose.

# In[10]:


arr = np.arange(8)
arr = np.reshape(arr, (4, 2))
transposed = np.transpose(arr)
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(transposed))
print('transposed shape: {}'.format(transposed.shape))


# >The function takes in a required first argument, which will be the array we want to transpose. It also has a single keyword argument called axes, which represents the new permutation of the dimensions.
# 
# >The permutation is a tuple/list of integers, with the same length as the number of dimensions in the array. It tells us where to switch up the dimensions. For example, if the permutation had 3 at index 1, it means the old third dimension of the data becomes the new second dimension (since index 1 represents the second dimension).
# 
# >The code below shows an example usage of the np.transpose function with the axes keyword argument. The shape property gives us the shape of an array.

# In[13]:


arr = np.arange(24)
arr = np.reshape(arr, (3, 4, 2))
transposed = np.transpose(arr, axes=(1, 2, 0))
print('arr shape: {}'.format(arr.shape))
print('transposed shape: {}'.format(transposed.shape))


# >In the above example, the old first dimension became the new third dimension, the old second dimension became the new first dimension, and the old third dimension became the new second dimension. The default value for axes is a dimension reversal (e.g. for 3-D data the default axes value is [2, 1, 0]).

# In[ ]:


#Creating Arrays with zeroes and ones.


# >Sometimes, we need to create arrays filled solely with 0 or 1. For example, since binary data is labeled with 0 and 1, we may need to create dummy datasets of strictly one label. For creating these arrays, NumPy provides the functions np.zeros and np.ones. They both take in the same arguments, which includes just one required argument, the array shape. The functions also allow for manual casting using the dtype keyword argument.

# In[14]:


arr = np.zeros(4)
print(repr(arr))


# In[15]:


arr = np.ones((2, 3))
print(repr(arr))


# In[16]:


arr = np.ones((2, 3), dtype=np.int32)
print(repr(arr))


# >If we want to create an array of 0's or 1's with the same shape as another array, we can use np.zeros_like and np.ones_like.

# In[17]:


arr = np.array([[1, 2], [3, 4]])
print(repr(np.zeros_like(arr)))


# In[18]:


arr = np.array([[0., 1.], [1.2, 4.]])
print(repr(np.ones_like(arr)))
print(repr(np.ones_like(arr, dtype=np.int32)))


# In[ ]:


###Arthimetic and Linear Algebra Operations in Numpy


# >One of the main purposes of NumPy is to perform multi-dimensional arithmetic. Using NumPy arrays, we can apply arithmetic to each element with a single operation.

# In[19]:


arr = np.array([[1, 2], [3, 4]])


# Add 1 to element values

# In[22]:


print(repr(arr + 1))


# Subtract element values by 1.2

# In[21]:


print(repr(arr - 1.2))


# Double element values

# In[23]:


print(repr(arr * 2))


# In[24]:


# Halve element values
print(repr(arr / 2))
# Integer division (half)
print(repr(arr // 2))
# Square element values
print(repr(arr**2))
# Square root element values
print(repr(arr**0.5))


# In[25]:


#The code below converts Fahrenheit to Celsius in NumPy.


# In[26]:


def f2c(temps):
    return (5/9)*(temps-32)


# In[27]:


fahrenheits = np.array([32, -4, 14, -40])
celsius = f2c(fahrenheits)
print('Celsius: {}'.format(repr(celsius)))


# In[29]:


#Non-Linear arthimetic functions in numpy


# >Apart from basic arithmetic operations, NumPy also allows us to use non-linear functions such as exponentials and logarithms.
# 
# >The function np.exp performs a base e exponential on an array, while the function np.exp2 performs a base 2 exponential. Likewise, np.log, np.log2, and np.log10 all perform logarithms on an input array, using base e, base 2, and base 10, respectively.

# In[30]:


arr = np.array([[1, 2], [3, 4]])
# Raised to power of e
print(repr(np.exp(arr)))


# In[31]:


# Raised to power of 2
print(repr(np.exp2(arr)))


# In[32]:


arr2 = np.array([[1, 10], [np.e, np.pi]])
# Natural logarithm
print(repr(np.log(arr2)))


# In[33]:


# Base 10 logarithm
print(repr(np.log10(arr2)))


# >To do a regular power operation with any base, we use np.power. The first argument to the function is the base, while the second is the power. If the base or power is an array rather than a single number, the operation is applied to every element in the array.

# In[36]:


arr = np.array([[1, 2], [3, 4]])
# Raise 3 to power of each number in arr
print(repr(np.power(3, arr)))


# In[37]:


arr2 = np.array([[10.2, 4], [3, 5]])
# Raise arr2 to power of each number in arr
print(repr(np.power(arr2, arr)))


# <a> "https://docs.scipy.org/doc/numpy/reference/routines.math.html" Link to other Functions in numpy</a>

# >Since NumPy arrays are basically vectors and matrices, it makes sense that there are functions for dot products and matrix multiplication. Specifically, the main function to use is np.matmul, which takes two vector/matrix arrays as input and produces a dot product or matrix multiplication.
# 
# >The code below shows various examples of matrix multiplication. When both inputs are 1-D, the output is the dot product.
# 
# >Note that the dimensions of the two input matrices must be valid for a matrix multiplication. Specifically, the second dimension of the first matrix must equal the first dimension of the second matrix, otherwise np.matmul will result in a ValueError.

# In[40]:


arr1 = np.array([1, 2, 3])
arr2 = np.array([-3, 0, 10])
print(np.matmul(arr1, arr2))


# In[41]:


arr3 = np.array([[1, 2], [3, 4], [5, 6]])
arr4 = np.array([[-1, 0, 1], [3, 2, -4]])
print(repr(np.matmul(arr3, arr4)))
print(repr(np.matmul(arr4, arr3)))
# This will result in ValueError
print(repr(np.matmul(arr3, arr3)))


# In[1]:


#Random Operations in Numpy


# >Similar to the Python random module, NumPy has its own submodule for pseudo-random number generation called np.random. It provides all the necessary randomized operations and extends it to multi-dimensional arrays. To generate pseudo-random integers, we use the np.random.randint function.

# In[6]:


print(np.random.randint(5))


# >The np.random.randint function takes in a single required argument, which actually depends on the high keyword argument. If high=None (which is the default value), then the required argument represents the upper (exclusive) end of the range, with the lower end being 0. Specifically, if the required argument is n, then the random integer is chosen uniformly from the range [0, n).

# In[12]:


print(np.random.randint(5,high=None))


# >If high is not None, then the required argument will represent the lower (inclusive) end of the range, while high represents the upper (exclusive) end.

# In[13]:


print(np.random.randint(5, high=6))


# >The size keyword argument specifies the size of the output array, where each integer in the array is randomly drawn from the specified range. As a default, np.random.randint returns a single integer.

# In[14]:


random_arr = np.random.randint(-3, high=14,
                               size=(2, 2))


# In[15]:


print(random_arr)


# In[16]:


print(np.random.randint(2, 10, (2, 3, 4)))


# In[17]:


#Utility Functions-seed and Shuffle


# In[18]:


np.random.seed(1)
print(np.random.randint(10))
random_arr = np.random.randint(3, high=100,
                               size=(2, 2))
print(repr(random_arr))


# In[19]:


# New seed
np.random.seed(2)
print(np.random.randint(10))
random_arr = np.random.randint(3, high=100,
                               size=(2, 2))
print(repr(random_arr))


# In[20]:


# Original seed
np.random.seed(1)
print(np.random.randint(10))
random_arr = np.random.randint(3, high=100,
                               size=(2, 2))
print(repr(random_arr))


# >The np.random.shuffle function allows us to randomly shuffle an array. Note that the shuffling happens in place (i.e. no return value), and shuffling multi-dimensional arrays only shuffles the first dimension.

# In[21]:


vec = np.array([1, 2, 3, 4, 5])
np.random.shuffle(vec)
print(repr(vec))


# >Note that only the rows of matrix are shuffled (i.e. shuffling along first dimension only).

# In[22]:


matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
np.random.shuffle(matrix)
print(repr(matrix))


# In[23]:


#Random Sampling from Uniform Distributions.


# >Using np.random we can also draw samples from probability distributions. For example, we can use np.random.uniform to draw pseudo-random real numbers from a uniform distribution.

# In[24]:


print(np.random.uniform())


# >The function np.random.uniform actually has no required arguments. The keyword arguments, low and high, represent the inclusive lower end and exclusive upper end from which to draw random samples. Since they have default values of 0.0 and 1.0, respectively, the default outputs of np.random.uniform come from the range [0.0, 1.0).

# In[25]:


print(np.random.uniform(low=-1.5, high=2.2))


# >The size keyword argument is the same as the one for np.random.randint, i.e. it represents the output size of the array.

# In[26]:


print(repr(np.random.uniform(size=3)))


# In[27]:


print(repr(np.random.uniform(low=-3.4, high=5.9,
                             size=(2, 2))))


# >Another popular distribution we can sample from is the normal (Gaussian) distribution. The function we use is np.random.normal.

# In[28]:


print(np.random.normal())


# >Like np.random.uniform, np.random.normal has no required arguments. The loc and scale keyword arguments represent the mean and standard deviation, respectively, of the normal distribution we sample from.

# In[29]:


print(np.random.normal(loc=1.5, scale=3.5))


# <a> "https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.random.html" Link to access other random functions in numpy</a>

# In[31]:


#Custom Sampling in Numpy


# >While NumPy provides built-in distributions to sample from, we can also sample from a custom distribution with the np.random.choice function.

# In[32]:


colors = ['red', 'blue', 'green']
print(np.random.choice(colors))


# In[33]:


print(repr(np.random.choice(colors, size=2)))


# >The required argument for np.random.choice is the custom distribution we sample from. The p keyword argument denotes the probabilities given to each element in the input distribution. Note that the list of probabilities for p must sum to 1.
# 
# >In the example, we set p such that 'red' has a probability of 0.8 of being chosen, 'blue' has a probability of 0.19, and 'green' has a probability of 0.01. When p is not set, the probabilities are equal for each element in the distribution (and sum to 1).

# In[34]:


print(repr(np.random.choice(colors, size=(2, 2),
                            p=[0.8, 0.19, 0.01])))


# In[ ]:


#Indexing and Slicing Numpy arrays


# In[35]:


#Accessing elements in array
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])
print(arr[4])

arr = np.array([[6, 3], [0, 2]])
# Subarray
print(repr(arr[0]))


# In[36]:


#Slicing Numpy Arrays
arr = np.array([1, 2, 3, 4, 5])


# In[37]:


print(repr(arr[:]))


# In[38]:


print(repr(arr[1:]))


# In[39]:


print(repr(arr[2:4]))


# In[40]:


print(repr(arr[:-1]))


# In[41]:


print(repr(arr[-2:]))


# For multi-dimensional arrays, we can use a comma to separate slices across each dimension.

# In[43]:


arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])


# In[44]:


print(repr(arr[:]))


# In[45]:


print(repr(arr[1:]))


# In[46]:


print(repr(arr[:, -1]))


# In[47]:


print(repr(arr[:, 1:]))


# In[48]:


print(repr(arr[0:1, 1:]))


# In[49]:


print(repr(arr[0, 1:]))


# >In addition to accessing and slicing arrays, it is useful to figure out the actual indexes of the minimum and maximum elements. To do this, we use the np.argmin and np.argmax functions.

# In[50]:


arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [-3, 9, 1]])


# In[51]:


print(np.argmin(arr[0]))


# In[52]:


print(np.argmax(arr[2]))


# In[53]:


print(np.argmin(arr))


# >The np.argmin and np.argmax functions take the same arguments. The required argument is the input array and the axis keyword argument specifies which dimension to apply the operation on.

# In[54]:


arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [-3, 9, 1]])


# >axis=0 meant the function found the index of the minimum row element for each column.

# In[55]:


print(repr(np.argmin(arr, axis=0)))


# >axis=1 meant the function found the index of the minimum column element for each row.

# In[56]:


print(repr(np.argmin(arr, axis=1)))


# Setting axis to -1 just means we apply the function across the last dimension.

# In[57]:


print(repr(np.argmax(arr, axis=-1)))


# In[59]:


### Filtering Data in Numpy


# In[60]:


arr = np.array([[0, 2, 3],
                [1, 3, -6],
                [-3, -2, 1]])
print(repr(arr == 3))


# In[61]:


print(repr(arr > 0))


# In[62]:


print(repr(arr != 1))


# In[63]:


# Negated from the previous step
print(repr(~(arr != 1)))


# >Something to note is that np.nan can't be used with any relation operation. Instead, we use np.isnan to filter for the location of np.nan.
# 
# >The code below uses np.isnan to determine which locations of the array contain np.nan values.

# In[64]:


arr = np.array([[0, 2, np.nan],
                [1, np.nan, -6],
                [np.nan, -2, 1]])
print(repr(np.isnan(arr)))


# >The np.where function takes in a required first argument, which is a boolean array where True represents the locations of the elements we want to filter for. When the function is applied with only the first argument, it returns a tuple of 1-D arrays.
# 
# >The tuple will have size equal to the number of dimensions in the data, and each array represents the True indices for the corresponding dimension. Note that the arrays in the tuple will all have the same length, equal to the number of True elements in the input argument.

# In[65]:


print(repr(np.where([True, False, True])))


# In[66]:


arr = np.array([0, 3, 5, 3, 1])
print(repr(np.where(arr == 3)))


# In[67]:


arr = np.array([[0, 2, 3],
                [1, 0, 0],
                [-3, 0, 0]])
x_ind, y_ind = np.where(arr != 0)


# In[68]:


print(repr(x_ind)) # x indices of non-zero elements
print(repr(y_ind)) # y indices of non-zero elements
print(repr(arr[x_ind, y_ind]))


# >The interesting thing about np.where is that it must be applied with exactly 1 or 3 arguments. When we use 3 arguments, the first argument is still the boolean array. However, the next two arguments represent the True replacement values and the False replacement values, respectively. The output of the function now becomes an array with the same shape as the first argument.

# In[70]:


np_filter = np.array([[True, False], [False, True]])
positives = np.array([[1, 2], [3, 4]])
negatives = np.array([[-2, -5], [-1, -8]])
print(repr(np.where(np_filter, positives, negatives)))


# In[71]:



np_filter = positives > 2
print(repr(np.where(np_filter, positives, negatives)))

np_filter = negatives > 0
print(repr(np.where(np_filter, positives, negatives)))


# >However, if we wanted to use a constant replacement value, e.g. -1, we could incorporate broadcasting. Rather than using an entire array of the same value, we can just use the value itself as an argument.
# 
# >The code below showcases broadcasting with np.where.

# In[72]:


np_filter = np.array([[True, False], [False, True]])
positives = np.array([[1, 2], [3, 4]])
print(repr(np.where(np_filter, positives, -1)))


# In[73]:


##Axis-Wise Filtering


# >The np.any function is equivalent to performing a logical OR (||), while the np.all function is equivalent to a logical AND (&&) on the first argument. np.any returns true if even one of the elements in the array meets the condition and np.all returns true only if all the elements meet the condition. When only a single argument is passed in, the function is applied across the entire input array, so the returned value is a single boolean.

# In[74]:


arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [3, 9, 1]])


# In[76]:


print(repr(arr > 0))
print(repr(np.any(arr > 0, axis=0)))
print(repr(np.any(arr > 0, axis=1)))
print(repr(np.all(arr > 0, axis=-1)))


# In[77]:


arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [3, 9, 1]])
has_positive = np.any(arr > 0, axis=1)
print(has_positive)
print(repr(arr[np.where(has_positive)]))


# #Creating a function The function replaces each of the non-positive elements in data with 0.

# In[78]:


def replace_zeros(data):
  # CODE HERE
  zeros=np.zeros_like(data)
  zero_replace=np.where(data>0,data,zeros)
  return zero_replace
  pass


# In[ ]:


#Creating Basic Statistical Analysis in Numpy


# >The axis keyword argument is identical to how it was used in np.argmin and np.argmax from the chapter on Indexing. In our example, we use axis=0 to find an array of the minimum values in each column of arr and axis=1 to find an array of the maximum values in each row of arr.

# In[79]:


arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(arr.min())
print(arr.max())

print(repr(arr.min(axis=0)))
print(repr(arr.max(axis=-1)))


# <a> "https://docs.scipy.org/doc/numpy/reference/routines.statistics.html" </a>
# More Statistical Functions in Numpy

# NumPy also provides basic statistical functions such as np.mean, np.var, and np.median, to calculate the mean, variance, and median of the data, respectively.

# In[82]:


arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(np.mean(arr))
print(np.var(arr))
print(np.median(arr))
print(repr(np.median(arr, axis=-1)))


# In[83]:


def basic_stats(data):
  # CODE HERE
  mean=np.mean(data)
  median=np.median(data)
  var=np.var(data)
  return (mean,median,var)
  pass


# In[84]:


#Aggregation Techniques in Numpy


# >To sum the values within a single array, we use the np.sum function.
# 
# >The function takes in a NumPy array as its required argument, and uses the axis keyword argument in the same way as described in previous chapters. If the axis keyword argument is not specified, np.sum returns the overall sum of the array.

# In[85]:


arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(np.sum(arr))
print(repr(np.sum(arr, axis=0)))
print(repr(np.sum(arr, axis=1)))


# >In addition to regular sums, NumPy can perform cumulative sums using np.cumsum. Like np.sum, np.cumsum also takes in a NumPy array as a required argument and uses the axis argument. If the axis keyword argument is not specified, np.cumsum will return the cumulative sums for the flattened array.
# 
# >The code below shows how to use np.cumsum. For a 2-D NumPy array, setting axis=0 returns an array with cumulative sums across each column, while axis=1 returns the array with cumulative sums across each row. Not setting axis returns a cumulative sum across all the values of the flattened array.

# In[86]:


arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(repr(np.cumsum(arr)))
print(repr(np.cumsum(arr, axis=0)))
print(repr(np.cumsum(arr, axis=1)))


# In[ ]:


#Concatenation Operation in Numpy


# >Like the summation functions, np.concatenate uses the axis keyword argument. However, the default value for axis is 0 (i.e. dimension 0). Furthermore, the required argument for np.concatenate is a list of arrays, which the function combines into a single array.
# 
# >The code below shows how to use np.concatenate, which aggregates arrays by joining them along a specific dimension. For 2-D arrays, not setting the axis argument (defaults to axis=0) concatenates the arrays vertically. When we set axis=1, the arrays are concatenated horizontally.

# In[88]:


arr1 = np.array([[0, 72, 3],
                 [1, 3, -60],
                 [-3, -2, 4]])
arr2 = np.array([[-15, 6, 1],
                 [8, 9, -4],
                 [5, -21, 18]])
print(repr(np.concatenate([arr1, arr2])))
print(repr(np.concatenate([arr1, arr2], axis=1)))


# In[89]:


def get_sums(data):
  # CODE HERE
  total_sum=np.sum(data)
  col_sum=np.sum(data,axis=0)
  return (total_sum,col_sum)
  pass


# In[90]:


def concat_arrays(data1, data2):
  # CODE HERE
  col_concat=np.concatenate([data1,data2])
  row_concat=np.concatenate([data1,data2],axis=1)
  return col_concat,row_concat
  pass


# In[ ]:




