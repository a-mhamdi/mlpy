import marimo

__generated_with = "0.10.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Programming with Python
        ---
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Goals**

        1. Learn the basics of programming in *Python*;
        2. Get familiar with *Marimo Notebook*;
        3. Use the modules of scientific computing.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Numerical variables & types""")
    return


@app.cell
def _():
    a = 1 # An integer
    print('The variable a = {} is of type {}'.format(a, type(a)))
    return (a,)


@app.cell
def _():
    b = -1.25 # A floating number
    print('The variable b = {} is of type {}'.format(b, type(b)))
    return (b,)


@app.cell
def _():
    c = 1+0.5j # A complex number 
    print('The variable c = {} is of type {}'.format(c, type(c)))
    return (c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Strings""")
    return


@app.cell
def _():
    msg = "My 1st lab!"
    print(msg, type(msg), sep = '\n***\n') # \n: Carriage Return & Line Feed
    print(msg + 3* '\nPython is awesome')
    return (msg,)


@app.cell
def _():
    longMsg = """This is a long message,
    spanned over multiple lines"""
    print(longMsg)
    return (longMsg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Indexing and slicing_""")
    return


@app.cell
def _(msg):
    # Positive indexing
    print(msg, msg[1:5], sep = ' -----> ')
    # Negative indexing
    print(msg, msg[-5:-1], sep = ' -----> ')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_String transformations_""")
    return


@app.cell
def _():
    msg_1 = 'A message'
    print(len(msg_1))
    print(msg_1.lower())
    print(msg_1.upper())
    print(msg_1.split(' '))
    print(msg_1.replace('mes', 'MES'))
    print('a' in msg_1)
    return (msg_1,)


@app.cell
def _():
    price, number, perso = 300, 7, 'A customer'
    print('{} asks for {} pieces. They cost {} TND!'.format(perso, number, price))
    print('{1} demande {2} pièces. They cost {0} TND!'.format(price, perso, number))
    return number, perso, price


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Binary, octal & hexadecimal""")
    return


@app.cell
def _():
    x = 0b0101 # 0b : binary
    print(x, type(x), sep = '\t----\t') # \t : tabular
    y = 0xAF # Ox : hexadecimal
    print(y, type(y), sep = '\t' + '---'*5 + '\t')
    z = 0o010 # 0o : octal
    print(z, type(z), sep = ', ')
    return x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Boolean_""")
    return


@app.cell
def _():
    a_1 = True
    b_1 = False
    print(a_1)
    print(b_1)
    return a_1, b_1


@app.cell
def _():
    print("50 > 20 ? : {} \n50 < 20 ? : {} \n50 = 20 ? : {}\n50 /= 20 ? : {}"
          .format(50 > 20, 50 < 20, 50 == 20, 50 != 20)
         )
    return


@app.cell
def _():
    print(bool(123), bool(0), bool('Lab'), bool())
    return


@app.cell
def _():
    var1 = 100
    print(isinstance(var1, int))
    var2 = -100.35
    print(isinstance(var2, int))
    print(isinstance(var2, float))
    return var1, var2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Lists, tuples & dictionaries""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In Python, a list is an ordered collection of items that can be of any data type (including other lists). Lists are defined using square brackets, with items separated by commas. For example:""")
    return


@app.cell
def _():
    shopping_list = ['milk', 'eggs', 'bread', 'apples']
    return (shopping_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A tuple is also an ordered collection of items, but it is immutable, meaning that the items it contains cannot be modified once the tuple is created. Tuples are defined using parentheses, with items separated by commas. For example:""")
    return


@app.cell
def _():
    point = (3, 5)
    return (point,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A dictionary is a collection of key-value pairs, where the keys are unique and used to look up the corresponding values. Dictionaries are defined using curly braces, with the key-value pairs separated by commas. The keys and values are separated by a colon. For example:""")
    return


@app.cell
def _():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Eve': '555-9101'}
    return (phonebook,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can access the items in a list or tuple using an index, and you can access the values in a dictionary using the corresponding keys. For example:""")
    return


@app.cell
def _(phonebook, point, shopping_list):
    # Accessing the second item in a list
    print(shopping_list[1])  # prints 'eggs'

    # Accessing the first item in a tuple
    print(point[0])  # prints 3

    # Accessing the phone number for 'Bob' in the phonebook dictionary
    print(phonebook['Bob'])  # prints '555-5678'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### List""")
    return


@app.cell
def _():
    lst = ['a', 'b', 'c', 1, True] # An aggregate of various types
    print(lst)
    return (lst,)


@app.cell
def _(lst):
    print(len(lst)) # Length of `lst` variable
    print(lst[1:3]) # Accessing elements of `lst`
    lst[0] = ['1', 0] # Combined list
    print(lst)
    print(lst[3:])
    print(lst[:3])
    return


@app.cell
def _(lst):
    lst.append('etc') # Insert 'etc' at the end
    print(lst)
    return


@app.cell
def _(lst):
    lst.insert(1, 'xyz') # Inserting 'xyz'
    print(lst)
    return


@app.cell
def _(lst):
    lst.pop(1)
    print(lst)
    return


@app.cell
def _(lst):
    lst.pop()
    print(lst)
    return


@app.cell
def _(lst):
    del lst[0]
    print(lst)
    return


@app.cell
def _(lst):
    lst.append('b')
    print(lst)
    lst.remove('b')
    print(lst)
    return


@app.cell
def _(lst):
    # Loop
    for k in lst:
        print(k)
    return (k,)


@app.cell
def _(lst):
    lst.clear()
    print(lst)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        | **_Method_**  | **_Description_** |
        | ------------- | ------------- |
        | **copy()**    | Returns a copy of the list |
        | **list()**    | Transforms into a list |
        | **extend ()** | Extends a list by adding elements at its end |
        | **count()**   | Returns the occurrences of the specified value |
        | **index()**   | Returns the index of the first occurrence of a specified value |
        | **reverse()** | Reverse a list |
        | **sort()**    | Sort a list |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Tuples""")
    return


@app.cell
def _():
    tpl = (1, 2, 3)
    print(tpl)
    return (tpl,)


@app.cell
def _():
    tpl_1 = (1, '1', 2, 'text')
    print(tpl_1)
    return (tpl_1,)


@app.cell
def _(tpl_1):
    print(len(tpl_1))
    return


@app.cell
def _(tpl_1):
    print(tpl_1[1:])
    return


@app.cell
def _(tpl_1):
    try:
        tpl_1.append('xyz')
    except Exception as err:
        print(err)
    return


@app.cell
def _(tpl_1):
    try:
        tpl_1.insert(1, 'xyz')
    except Exception as err:
        print(err)
    return


@app.cell
def _(tpl_1):
    my_lst = list(tpl_1)
    my_lst.append('xyz')
    print(my_lst, type(my_lst), sep=', ')
    return (my_lst,)


@app.cell
def _(my_lst):
    nv_tpl = tuple(my_lst) # Convert 'my_lst' into a tuple 'nv_tpl'
    print(nv_tpl, type(nv_tpl), sep = ', ')
    return (nv_tpl,)


@app.cell
def _(nv_tpl):
    for k_1 in nv_tpl:
        print(k_1)
    return (k_1,)


@app.cell
def _(nv_tpl, tpl_1):
    rs_tpl = tpl_1 + nv_tpl
    print(rs_tpl)
    return (rs_tpl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Dictionaries""")
    return


@app.cell
def _():
    # dct = {"key": "value"}
    dct = {
        "Term" : "GM",
        "Speciality" : "ElnI",
        "Sem" : "4"
    }
    print(dct, type(dct), sep = ', ')
    return (dct,)


@app.cell
def _(dct):
    print(dct["Sem"])
    sem = dct.get("Sem")
    print(sem)
    return (sem,)


@app.cell
def _(dct):
    dct["Term"] = "GE"
    print(dct)
    return


@app.cell
def _(dct):
    # Loop
    for el in dct:
        print(el, dct[el], sep = '\t|\t')
    return (el,)


@app.cell
def _(dct):
    for k_2 in dct.keys():
        print(k_2)
    return (k_2,)


@app.cell
def _(dct):
    for v in dct.values():
        print(v)
    return (v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## NumPy""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *NumPy* is a *Python* library that is used for scientific computing and data analysis. It provides support for large, multi-dimensional arrays and matrices of numerical data, and a large library of mathematical functions to operate on these arrays.

        One of the main features of *NumPy* is its $N$-dimensional array object, which is used to store and manipulate large arrays of homogeneous data (_i.e._, data of the same type, such as integers or floating point values). The array object provides efficient operations for performing element-wise calculations, indexing, slicing, and reshaping.

        *NumPy* also includes a number of functions for performing statistical and mathematical operations on arrays, such as mean, standard deviation, and dot product. It also includes functions for linear algebra, random number generation, and Fourier transforms.

        Official documentation can be found at [https://numpy.org/](https://numpy.org/)
        """
    )
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_NumPy vs List_""")
    return


@app.cell
def _(np):
    a_np = np.arange(6) # NumPy
    print("a_np = ", a_np)
    print(type(a_np))
    a_lst = list(range(0,6)) # List
    print("a_lst = ", a_lst)
    print(type(a_lst))
    # Comparison
    print("2 * a_np = ", a_np * 2)
    print("2 * a_lst = ", a_lst * 2)
    return a_lst, a_np


@app.cell
def _(np):
    v_np = np.array([1, 2, 3, 4, 5, 6]) # NB : parentheses then brackets, i.e, ([])
    print(v_np)
    return (v_np,)


@app.cell
def _(np):
    v_np_1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(v_np_1)
    return (v_np_1,)


@app.cell
def _(v_np_1):
    print(type(v_np_1))
    return


@app.cell
def _(v_np_1):
    print(v_np_1[0])
    return


@app.cell
def _(v_np_1):
    v_np_1.ndim
    return


@app.cell
def _(v_np_1):
    v_np_1.shape
    return


@app.cell
def _(v_np_1):
    v_np_1.size
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we need to create a matrix $(3, 3)$, we can do as follows:""")
    return


@app.cell
def _(np):
    u = np.arange(9).reshape(3,3)
    print(u)
    return (u,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let us see some known operations to do on matrices""")
    return


@app.cell
def _(np):
    M = np.array([[1, 2], [1, 2]])
    print(M)
    return (M,)


@app.cell
def _(np):
    N = np.array([[0, 3], [4, 5]])
    print(N)
    return (N,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Addition_""")
    return


@app.cell
def _(M, N, np):
    print(M + N)
    print(np.add(M, N))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Subtraction_""")
    return


@app.cell
def _(M, N, np):
    print(M-N)
    print(np.subtract(M, N))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Element-wise Division_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \left[\begin{array}{cc}0&3\\4&5\end{array}\right]
        ./
        \left[\begin{array}{cc}1&2\\1&2\end{array}\right]
        \quad =\quad
        \left[\begin{array}{cc}0:1&3:2\\4:1&5:2\end{array}\right]
        $$
        """
    )
    return


@app.cell
def _(M, N, np):
    print(N / M)
    print(np.divide(N, M))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Element-wise Product_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Element-wise multiplication, also known as **Hadamard product**, is an operation that multiplies each element of one matrix with the corresponding element of another matrix. It is denoted by the symbol $\odot$ or `.*` in some programming languages.

        For example, consider the following matrices:

        $$A = \left[\begin{array}{ccc}a_1,& a_2,& a_3\end{array}\right] \qquad\text{and}\qquad B = \left[\begin{array}{ccc}b_1,& b_2,& b_3\end{array}\right]$$


        The element-wise product of these matrices is:

        $$A \odot B = \left[\begin{array}{ccc}a_1b_1,& a_2b_2,& a_3b_3\end{array}\right]$$


        $$
        \left[\begin{array}{cc}1&2\\1&2\end{array}\right]
        .\times
        \left[\begin{array}{cc}0&3\\4&5\end{array}\right]
        \quad =\quad
        \left[\begin{array}{cc}0&6\\4&10\end{array}\right]
        $$


        We need element-wise multiplication in many applications. For example, in image processing, element-wise multiplication is used to modify the intensity values of an image by multiplying each pixel value with a scalar value. In machine learning, element-wise multiplication is used in the implementation of various neural network layers, such as convolutional layers and fully connected layers. Element-wise multiplication is also used in many other mathematical and scientific applications.
        """
    )
    return


@app.cell
def _(M, N, np):
    print(M * N)
    print(np.multiply(M, N))
    return


@app.cell
def _(mo):
    mo.md(r"""_Dot Product_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \left[\begin{array}{cc}1&2\\1&2\end{array}\right]
        \times
        \left[\begin{array}{cc}0&3\\4&5\end{array}\right]
        \quad =\quad
        \left[\begin{array}{cc}8&13\\8&13\end{array}\right]
        $$
        """
    )
    return


@app.cell
def _(M, N, np):
    print(M.dot(N))
    print(np.dot(M, N))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Kronecker Product_""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \left[\begin{array}{cccc}1&2&3&4\end{array}\right]
        \bigotimes
        \left[\begin{array}{ccc}1&2\\3&4\\5&6\\7&8\end{array}\right] \;=\; 
        \left[\begin{array}{cccccccccccc}
        1&2&2&4&3&6&4&8\\3&4&6&8&9&12&12&16\\5&6&10&12&15&18&20&24\\7&8&14&16&21&24&28&32
        \end{array}\right]
        $$
        """
    )
    return


@app.cell
def _(np):
    u_1 = np.arange(1, 5)
    v_1 = np.arange(1, 9).reshape(4, 2)
    (u_1, v_1)
    return u_1, v_1


@app.cell
def _(np, u_1, v_1):
    np.kron(u_1, v_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_Determinant of a matrix_""")
    return


@app.cell
def _(M, N, np):
    print("Determinant of M:")
    print(np.linalg.det(M))
    print("Determinant of N:")
    print(np.linalg.det(N))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Matplotlib""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *Matplotlib* is a $2$D data visualization library in *Python* that allows users to create a wide range of static, animated, and interactive visualizations in *Python*. It is one of the most widely used data visualization libraries in the *Python* data science ecosystem and is particularly useful for creating line plots, scatter plots, bar plots, error bars, histograms, bar charts, pie charts, box plots, and many other types of visualizations.

        *Matplotlib* is built on top of *NumPy* and is often used in conjunction with other libraries in the PyData ecosystem, such as *Pandas* and *Seaborn*, to create complex visualizations of data. It is also compatible with a number of different backends, such as the _Jupyter notebook_, _Qt_, and _Tkinter_, which allows it to be used in a wide range of environments and contexts.

        The full documentation and an exhaustive list of samples can be found at [https://matplotlib.org/](https://matplotlib.org/)
        """
    )
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    plt.rc('figure', figsize=(6, 4))
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We begin by creating a sinusoidal waveform denoted by $x$, period is $1$ sec. The offset is $1$.""")
    return


@app.cell
def _(np):
    t = np.arange(0.0, 2.0, 0.01)
    x_1 = 1 + np.sin(2 * np.pi * t)
    return t, x_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The set of instructions that allow to plot \(x\) are:""")
    return


@app.cell
def _(plt, t, x_1):
    plt.plot(t, x_1)
    plt.title('$x(t) = 1+\\sin\\left(2\\pi\\frac{t}{1}\\right)$')
    plt.xlabel('$t$ (sec)')
    return


@app.cell
def _(np):
    t_1 = np.arange(0.0, 2.0, 0.1)
    y_1 = np.sin(2 * np.pi * t_1)
    return t_1, y_1


@app.cell
def _(plt, t_1, y_1):
    plt.stem(t_1, y_1)
    plt.xlabel('$t$ (sec)')
    return


@app.cell
def _(np):
    x_2 = np.logspace(-2, 3, 100)
    y_2 = np.log10(x_2)
    return x_2, y_2


@app.cell
def _(np):
    np.log10.__doc__
    return


@app.cell
def _(plt, x_2, y_2):
    plt.plot(x_2, y_2)
    return


@app.cell
def _(plt, x_2, y_2):
    plt.semilogx(x_2, y_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**About distributions**""")
    return


@app.cell
def _(np):
    a_2 = np.random.randn(2 ** 16)
    b_2 = np.random.rand(2 ** 16)
    return a_2, b_2


@app.cell
def _(a_2, b_2, plt):
    (_, ax) = plt.subplots(1, 2)
    ax[0].hist(a_2, bins=16)
    ax[1].hist(b_2, bins=16)
    return (ax,)


@app.cell
def _(a_2, b_2, plt):
    (_, ax_1) = plt.subplots(2, 2)
    ax_1[0][0].hist2d(a_2, a_2, bins=32)
    ax_1[0][1].hist2d(a_2, b_2, bins=32)
    ax_1[1][0].hist2d(b_2, a_2, bins=32)
    ax_1[1][1].hist2d(b_2, b_2, bins=32)
    return (ax_1,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
