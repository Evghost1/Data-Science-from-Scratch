import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Callable, TypeVar


# Create Vector Type
Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), 'vectors musth be the same length'
# use zip to pair elements together, then unpack them. Used when more variables have same length and on elementwise
    return [v_i + w_i for v_i, w_i in zip(v,w)] 

assert add([1,2,3],[4,5,6]) == [5,7,9]





def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), 'vectors musth be the same length'

    return [v_i - w_i for v_i, w_i in zip(v,w)]

assert subtract([1,2,3],[4,5,6]) == [-3,-3,-3]




def scalar_multiply(c: float, v:Vector) -> Vector:
    # Multiplies every element by c
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1,2,3,4]) == [2,4,6,8]




# The dot product of two vectors is the sum of their componentwise products
# if w has magnitude 1, the dot product measures how far the vector v extends in the w direction.
# another way of saying this is that it's thelength of the vector you'd get if you projected v onto w
def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), 'Vectors must be same length'

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1,2,3],[4,5,6]) == 32




def vector_sum(vectors: List[Vector]) -> Vector:
    # Check that vectors is not empty
    assert vectors, 'no vectors provided!'

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), 'different sizes!' #all, any used when check all or each elements

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1,2,3],[4,5,6]]) == [5, 7, 9]




def vector_mean(vectors: List[Vector]) -> Vector:
    # Computes the element-wise average
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1,2],[3,4],[5,6]]) == [3, 4]




def sum_of_squares(v: Vector) ->float:
    return dot(v, v)

assert sum_of_squares([1,2,3]) == 14




def magnitude(v: Vector) -> float:
    #return the magnitude (or length) of v
    return math.sqrt(sum_of_squares(v))

assert magnitude([3,4]) == 5




def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))


# define the type Matrix
## first - we can use a matrix to represent a dataset consisting of multiple vectors, simply by considering each vector as a row of the matrix
## Secod - we can use a n X k matrix to represent a linear function that maps k-dimensional vectors to n-dimensioanl vectors
## Third - matrices can be used to represent binary relationships.
Matrix = List[List[float]]




def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    return num_rows, num_cols

assert shape([[1,2,3],[4,5,6]]) == (2,3)




def get_row(A: Matrix, i: int) -> Vector:
    return A[i]




def get_column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]




def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)










