# Polypy
A simple python library for polynomials and other math tools. This README is also structured in a way that represent the history and evolution of this code.

---
# Polynomials
- You can use the Poly class to create a polynomial starting from a list of coefficients;
- You can also create polynomials using std_poly(n) to create a polynomial with the shape x^n;
- Then you can sum, multiply and divide these to create even more sophisticated polynomials;
- Thanks to print_poly() you can visualize the polynomial in a readable format;
- There are also integrals (where you can choose the constant c) and derivative of any order;
- You can plot the polynomial and transform it into a function too.

# Other tools
- You can approximate derivatives in a certain point and definite integrals of functions f: R->R;
- You can approximate the gradient and hessian matrix for a function f:R^n->R;
- There is also a function to approximate Taylor series for functions f:R->R up to a certain exponent;
- There are three different functions to create the n-th Legendre, Laguerre and Hermite polynomials.

## Example of use
After putting [this file](polypy.py) in your working directory you can import the library like this: [import polypy as plp](getting_started.png). [This](example.png) is an example of use after. This library is not yet available on PyPI, so you need to place polypy.py in your project directory to use it. Now there is also the function "test_Poly()" to see an example of code built in the library.

# Additions
## SODE class
This class represents separable ODE and solve Cauchy problem related using second-order taylor expansions for a better approximation and plotting the solution. This can be confronted with the actual solution as I've done in the "test_SODE()" function, which is also an example of use of this class.

---

Possible future updates: adding general ODE solvers and root finding algorithms (like Newton), code cleanup, other improvements.

---

Note: this is a persinal project created for learning. It can be updated during time, but this is not guaranteed.
