import inspect
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

#BASE
#----------------------------------------------------------------
def approx(n: float,lvl=3):
    '''Approximates a number to a certain decimal place (lvl). Example: approx(3.1462,lvl=2) -> returns 3.15.'''
    k=n*10**int(lvl)
    p=k-int(k)
    m=k-p
    if p>=0.5:
       m+=1
    return m/(10**int(lvl))
#----------------------------------------------------------------
def fact(n: int):
    '''Calculates the factorial of n. Example: fact(5) -> returns 120.'''
    if n<0:
        raise ValueError("n must be a positive integer.")
    elif int(n)==0:
        return 1
    else:
        k=1
        for i in range(0,n):
            k*=i+1
        return k
    raise ValueError("Something unexpected happened.")
#----------------------------------------------------------------    
def divisors(n: int):
    '''Returns a list with all the dividors for a number.'''
    try:
        n=int(n)
    except Exception:
        raise ValueError("n must be a positive integer number.")
    if n>0:
        divisors=[]
        for i in range(1,int(n/2+1)):
            if n%i==0:
                divisors.append(i)
        divisors.append(n)
        return divisors
    else:
        raise ValueError("n must be a positive integer number.")
#----------------------------------------------------------------
#POLYNOMIAL
#----------------------------------------------------------------       
class Poly:
    def __init__(self,coeff: list):
        '''Initializes the polynomial class. Needs the coefficient list in ascending order of exponents. Example: Poly([0,1,2,0,1]) -> x+2x^2+x^4.'''
        try:
            coeff=list(coeff)
        except Exception:
            raise ValueError("You are not creating a polynomial. Try using a list or something similar.")
        if len(coeff)!=0:
            while coeff[len(coeff)-1]==0 and len(coeff)>1:
                del coeff[len(coeff)-1]
            self.coeffs=coeff
            self.exponents=[]
            self.useful_coeffs=[]
            for i in range(0,len(self.coeffs)):
                if self.coeffs[i]!=0:
                    self.exponents.append(i)
                    self.useful_coeffs.append(self.coeffs[i])
        else:
            return None
        
    def print_poly(self, approximation=True):
        '''Prints the polynomial in a readable way. Example: Poly([1,2,3]).print_poly() -> prints "1 + 2 x + 3 x^2".'''
        if self!=None and self.coeffs!=[0]:
            for i in range(0,len(self.exponents)):
                if self.exponents[i]==0:
                    if approximation==True:
                        print(approx(self.useful_coeffs[i]),sep="",end="")
                    else:
                        print(self.useful_coeffs[i],sep="",end="")
                elif self.exponents[i]==1:
                    if self.useful_coeffs[i]==1:
                        print("x",sep="",end="")
                    elif self.useful_coeffs[i]==-1:
                        if i==0:
                            print("-x",sep="",end="")
                        else:
                            print("x",sep="",end="")
                    else:
                        if i==0:
                            if approximation==True:
                                print(approx(self.useful_coeffs[i])," x",sep="",end="")
                            else:
                                print(self.useful_coeffs[i]," x",sep="",end="")
                        else:
                            if approximation==True:
                                print(approx(np.abs(self.useful_coeffs[i]))," x",sep="",end="")
                            else:
                                print(np.abs(self.useful_coeffs[i])," x",sep="",end="")
                else:
                    if self.useful_coeffs[i]==1:
                        print(" x^",self.exponents[i],sep="",end="")
                    elif self.useful_coeffs[i]==-1:
                        if i==0:
                            print("-",end="")
                        print(" x^",self.exponents[i],sep="",end="")
                    else:
                        if i==0:
                            if approximation==True:
                                print(approx(self.useful_coeffs[i])," x^",self.exponents[i],sep="",end="")
                            else:
                                print(self.useful_coeffs[i]," x^",self.exponents[i],sep="",end="")
                        else:
                            if approximation==True:
                                print(approx(np.abs(self.useful_coeffs[i]))," x^",self.exponents[i],sep="",end="")
                            else:
                                print(np.abs(self.useful_coeffs[i])," x^",self.exponents[i],sep="",end="")
                if i<len(self.exponents)-1:
                    if self.useful_coeffs[i+1]>0:
                        print(" + ", end="")
                    else:
                        print(" - ", end="")
            print("")
        elif self.coeffs==[0]:
            print(0)
        else:
            raise ValueError("You are not trying to print a polynomial.")
    def value(self,x0: float):
        '''Calculates the value of the polynomial in x0. Example: Poly([1,2,3]).value(0) -> returns 1.'''
        if self!=None:
            value=0
            for i in range(0,len(self.exponents)):
                value+=self.useful_coeffs[i]*(x0**self.exponents[i])
            return value
        else:
            return 0
    def sum(self,other):
        '''Returns the sum  of two polynomials. Example: P1=Poly([1,2]) P2=Poly([0,0,1]). P1.sum(P2) -> returns 1+2x+x^2.'''
        if self!=None and other!=None:
            new_coeff=[]
            N=max(len(self.coeffs),len(other.coeffs))
            n=min(len(self.coeffs),len(other.coeffs))
            for i in range(0,n):
                new_coeff.append(self.coeffs[i]+other.coeffs[i])
            for i in range(n,N):
                if len(self.coeffs)==N:
                    new_coeff.append(self.coeffs[i])
                elif len(other.coeffs)==N:
                    new_coeff.append(other.coeffs[i])            
                else:
                    raise ValueError("Error with polynomials.")
            return Poly(new_coeff)
        else:
            return None
    def opposite(self):
        '''Returns the opposite of the polynomial. Example: Poly([1,2]).opposite() -> returns -1-2x.'''
        if self!=None:
            new_coeff=[-c for c in self.coeffs]
            return Poly(new_coeff)
        else:
            return None
    def difference(self,other):
        '''Returns the difference between two polynomials. Example: P1=Poly([1,2]), P2=Poly([2]). P1.difference(P2) -> returns 2x-1.'''
        return self.sum(other.opposite())
    def product(self,other):
        '''Returns the product between two polynomials. Example: P1=Poly([0,1]), P2=Poly([0,0,1]). P1.product(P2) -> returns x^3.'''
        if self!=None and other!=None:
            P0=Poly([0])
            for i in range(0,len(self.coeffs)):
                new_coeff=[]
                for k in range(0,i):
                    new_coeff.append(0)
                for j in range(0,len(other.coeffs)):
                    new_coeff.append(self.coeffs[i]*other.coeffs[j])
                P1=Poly(new_coeff)
                P0=P0.sum(P1)
            return P0
        else:
            return None
    def multiply(self,k: float):
        '''Returns the polynomial multiplied by a scalar. Example: Poly([0,1]).multiply(3) -> returns 3x.'''
        if self!=None:
            new_coeff=[c*k for c in self.coeffs]
            return Poly(new_coeff)
        else:
            return None
    def divide(self,other):
        '''Returns the quotient and the remainder of the division between two polynomials. Example: P1=Poly([2,2,1]), P2=Poly([1,1]). P1.divide(P2) -> returns x+1, 1 (both as a Poly object).'''
        if self!=None and other!=None and other.coeffs!=[0]:
            new_coeff=[]
            for i in range(0,len(self.coeffs)-len(other.coeffs)+1):
                new_coeff.append(0)
            P1=Poly(self.coeffs)
            while len(P1.coeffs)>=len(other.coeffs):
                param=P1.coeffs[len(P1.coeffs)-1]/other.coeffs[len(other.coeffs)-1]
                new_coeff[len(P1.coeffs)-len(other.coeffs)]=param
                P2_coeff=[]
                for i in range(0,len(P1.coeffs)-len(other.coeffs)):
                    P2_coeff.append(0)
                P2_coeff.append(param)
                P1=P1.difference(other.product(Poly(P2_coeff)))
            return Poly(new_coeff),P1
        elif other.coeffs==[0]:
            raise ValueError("Error: division by zero.")
        else:
            return None
    def pow(self,n: int):
        '''Returns the polynomial raised to an integer power. Example: Poly([1,1]).pow(2) -> returns 1+x+x^2.'''
        if n<0:
            raise ValueError("n must be a positive integer.")
        elif int(n)==0:
            return Poly([1])
        else:
            p=self
            for i in range(1,int(n)):
                p=p.product(self)
            return p
        raise ValueError("Something unexpected happened.")
    def polyderivative(self):
        '''Returns the derivative of the polynomial as a new polynomial. Example Poly([0,0,1]).polyderivative() -> returns 2x.'''
        if self!=None:
            new_coeff=[]
            for i in range(1,len(self.coeffs)):
                new_coeff.append(i*self.coeffs[i])
            if len(new_coeff)==0:
                new_coeff=[0]
            return Poly(new_coeff)
        else:
            return None
    def polyderivative_n(self,n):
        '''Returns the n-th derivative of the polynomial as a new polynomial. Example Poly([0,0,1]).polyderivative_n(2) -> returns 2, as a Poly object.'''
        if self!=None:
            p=self
            for i in range(0,n):
                p=p.polyderivative()
            return p
        else:
            return None
    def polyint(self,c=0):
        '''Returns the integral with a given value for c. Example Poly([0,2]).polyint(c=3) -> returns x^2+3.'''
        if self!=None:
            new_coeff=[c]
            for i in range(0,len(self.coeffs)):
                new_coeff.append(self.coeffs[i]/(i+1))
            return Poly(new_coeff)
        else:
            return None
    def func(self):
        '''Returns the function that evaluates the polynomial. Example: Poly([0,1]).func() -> returns f(x)=x.'''
        def poly_func(x):
            f=0
            for i in range(0,len(self.exponents)):
               f+=self.useful_coeffs[i]*x**(self.exponents[i])
            return f
        return poly_func
    def plot(self,a=-2,b=2,draw_prec=50,show=True):
        '''Plots the polynomial in a given range.'''
        x=np.linspace(a,b,draw_prec)
        f=self.func()
        plt.plot(x,f(x))
        if show==True:
            plt.show()
    def bisection(self,a,b,prec=0.0001,approximation=True):
        '''Using bisection approximates a zero between two points for the polynomial. It works only if f(a)*f(b)<0.'''
        f=self.func()
        if f(a)*f(b)>=0:
            raise ValueError("Bisection doesn't work in this case.")
        c=(a+b)/2
        while f(c)<=-prec or f(c)>=prec:
            if f(c)*f(a)>0:
                a=c
            else:
                b=c
            c=(a+b)/2
        if approximation==True:
            lvl=0
            while prec<1:
                prec*=10
                lvl+=1
            return approx(c,lvl)
        else:
            return c
#----------------------------------------------------------------
def std_poly(n: int):
    '''Creates the polynomial: x^n. Example: std_poly(3) -> returns x^3 as an element of the poly class.'''
    try:
        n=int(n)
    except Exception:
        raise ValueError("n must be a positive integer.")
    if n<0:
        raise ValueError("n must be a positive integer.")        
    coeff=[]
    for i in range(0,n):
        coeff.append(0)
    coeff.append(1)
    return Poly(coeff)
#----------------------------------------------------------------
#DERIVATIVES AND DEFINITE INTEGRALS
#----------------------------------------------------------------
def derivative(f: Callable[[float],float],x0: float,prec=0.0001,approximation=True):
    '''Approximates the derivative of a function at a given point.'''
    der_plus=(f(x0+prec)-f(x0))/prec
    der_minus=(f(x0-prec)-f(x0))/prec
    if approximation==True:
        a=0
        while prec<1:
            prec*=10
            a+=1
        return approx((der_plus-der_minus)/2,lvl=a)
    else:
        return (der_plus-der_minus)/2
#------------------------------------------------------------------------			
def derivative2(f,x0: float,prec=0.0001,approximation=True):
    '''Approximates the second derivate of a function at a given point.'''
    df=lambda x: derivative(f,x,prec,approximation=False)
    return derivative(df,x0,prec,approximation)
#------------------------------------------------------------------------
def derivative_n(f: Callable[[float],float],x0: float,n: int,prec=0.0001,approximation=True):
    '''Approximates the n-th derivative of a function at a given point. For bigger n it's recommended to lower precision (prec) to 0.1.'''
    if n==1:
        return derivative(f,x0,prec,approximation)
    elif n>1:
        g=lambda x: derivative(f,x,prec,approximation=False)
        return derivative_n(g,x0,n-1,prec,approximation)
    else:
        raise ValueError("n must be a positive integer.") 
#------------------------------------------------------------------------
def integral(f: Callable[[float],float],a: float,b: float,prec=0.0001,approximation=True):
    '''Approximates the integral of a function in a given interval.'''
    integral_minus=0
    integral_plus=0
    while a<b:
        integral_minus+=prec*min(f(a),f(a+prec))
        integral_plus+=prec*max(f(a),f(a+prec))
        a+=prec
    if approximation==True:
        a=0
        while prec<1:
            prec*=10
            a+=1
        return approx((integral_minus+integral_plus)/2,lvl=a)
    else:
        return (integral_minus+integral_plus)/2
#----------------------------------------------------------------
def grad(f: Callable[[...,float],float],xs0: list,prec=0.0001,approximation=True):
    '''Approximates the gradient of a function at a given point.'''
    lenf=len(inspect.signature(f).parameters)
    if lenf!=len(xs0):
        raise ValueError("The function and the point don't have the same size.")
    xs0=list(xs0)
    grad=[]
    for i in range(0,len(xs0)):
        g= lambda x: f(*(xs0[:i]+[x]+xs0[i+1:]))
        grad.append(derivative(g,xs0[i],prec,approximation))
    return grad
#----------------------------------------------------------------    
def hesse(f: Callable[[...,float],float],xs0: list,prec=0.0001,approximation=True):
    '''Approximates the hessian matrix of a function at a given point.'''
    lenf=len(inspect.signature(f).parameters)
    if lenf!=len(xs0):
        raise ValueError("The function and the point don't have the same size.")
    hesse=np.zeros((lenf,lenf))
    xs0=list(xs0)
    for j in range(0,lenf):
        for i in range(0,lenf):
            if i>j:
                g= lambda xj, xi: f(*(xs0[:j]+[xj]+xs0[j+1:i]+[xi]+xs0[i+1:]))
                dg= lambda xj, xi: (grad(g,[xj,xi],prec,approximation=False))[0]
                dd= (grad(dg,[xs0[j],xs0[i]],prec,approximation))[1]
                hesse[i,j]=dd
            elif i<j:
                hesse[i][j]=hesse[j][i]
            elif i==j:
                g= lambda x: f(*(xs0[:i]+[x]+xs0[i+1:]))
                dg= lambda x: derivative(g,x,prec,approximation=False)
                hesse[i][j]=derivative(dg,xs0[i],prec,approximation)
            else:
                raise ValueError("Error during execution.")
    return hesse
#----------------------------------------------------------------            
#EXPANSIONS
#----------------------------------------------------------------
def taylor(f: Callable[[float],float],x0: float,lvl=3,prec=0.01,approximation=True):
    '''Returns the Taylor expansion of a function as a poly object.'''
    if len(inspect.signature(f).parameters)!=1:
        raise ValueError("The mathematical function must be f: R->R.")
    if type(lvl)!=int:
        lvl=int(lvl)
    p=Poly([f(x0)])
    stp=std_poly(1).difference(Poly([x0]))
    for i in range(0,lvl):
        q=stp.pow(i+1).multiply(derivative_n(f,x0,i+1,prec,approximation)/fact(i+1))
        p=p.sum(q)
    return p
#----------------------------------------------------------------
def legendre_poly(n: int):
    '''Returns the n-th Legendre polynomial as a poly object.'''
    try:
        n=int(n)
    except Exception:
        raise ValueError("n must be a positive integer number.")
    if n<0:
        raise ValueError("Error: n must be a positive integer.")
    elif n==0:
        return Poly([1])
    elif n==1:
        return std_poly(1)
    else:
        return (std_poly(1).product(legendre_poly(n-1).multiply(2*n-1)).difference(legendre_poly(n-2).multiply(n-1))).multiply(1/n)
#----------------------------------------------------------------
def laguerre_poly(n: int):    
    '''Returns the n-th Laguerre polynomial as a poly object.'''
    try:
        n=int(n)
    except Exception:
        raise ValueError("n must be a positive integer number.")
    if n<0:
        raise ValueError("Error: n must be a positive integer.")
    elif n==0:
        return Poly([1])
    elif n==1:
        return Poly([1,-1])
    else:
        return (laguerre_poly(n-1).product(Poly([2*n-1]).difference(std_poly(1)))).difference(laguerre_poly(n-2).multiply(n-1)).multiply(1/n)
#----------------------------------------------------------------
def hermite_poly(n: int):    
    '''Returns the n-th Hermite polynomial as a poly object.'''
    try:
        n=int(n)
    except Exception:
        raise ValueError("n must be a positive integer number.")
    if n<0:
        raise ValueError("Error: n must be a positive integer.")
    elif n==0:
        return Poly([1])
    elif n==1:
        return Poly([0,2])
    else:
        return Poly([0,2]).product(hermite_poly(n-1)).difference(hermite_poly(n-2).multiply(2*n-2))
#----------------------------------------------------------------        