# ME2-Coding
# ME2 Maths and Computing Module Coursework
Numerical solution of a differential equation using Python

05/03/2026 start 20/03/2026 deadline

briefing see CW.Assignment

The project focuses on modelling a physical system described by differential equations and solving it numerically.

---

# Authors

Peter Zhu 02570810
 
# Project Overview

The objective of this coursework is to:

1. Identify a physical phenomenon described by differential equations.

    we chose (membrane on speaker), use wave equation.

2. Formulate the governing equations and boundary/initial conditions.

    The wave equation in cylindrical coordinates:

    ∂²u/∂t² = c²(∂²u/∂r² + (1/r)∂u/∂r + (1/r²)∂²u/∂θ²) + q(r,θ,t)/ρₛ

    where:
    - u is displacement, 
    - c = √(T/ρₛ) is the membrane wave speed
    - T is the membrane tension per unit length
    - ρₛ is the surface mass density
    - q(r,θ,t) is the external distributed forcing from the voice coil / air loading

    Boundary conditions: 

    1, At the outer rim r=R: clamped edge
    u(R, θ, t) = 0 for all θ and t.

    2, At the centre r=0: the solution must remain finite and smooth. 
    ∂u​(0, θ, t)/∂r  = 0. ∂v​(0, θ, t)/∂r  = 0. 

未完成，此后所有信息请参考report.docx
未完成，此后所有信息请参考report.docx
未完成，此后所有信息请参考report.docx

3. Select an appropriate numerical method.

    explicit finite difference method

4. Derive the discretised form of the equations. 
5. Implement the numerical solver in Python.
6. Visualise and analyse the results.
7. Extend the analysis using an additional numerical computing topic.

---

# Physical Problem

(Describe the physical system studied in this project)

- Physical phenomenon
- Key assumptions
- Simplifications
- Geometry / domain of the system


# Governing Equations

(Present the differential equation describing the system.)
Example:
∂u/∂t = α ∂²u/∂x²

or

d²x/dt² + c dx/dt + kx = 0

Explain briefly:

- variables
- parameters
- physical meaning.


# Boundary and Initial Conditions

Specify the conditions used in the simulation.

# Numerical Method

Describe the numerical method used to solve the equation.

Examples:

- Finite Difference Method
- Forward Euler
- Backward Euler
- Runge–Kutta

Explain:

- why this method was chosen
- advantages / limitations.

# Discretisation

Derive the discretised form of the equation using the notation from the course slides.

Example:


u_i^{n+1} = u_i^n + Δt ( ... )


Explain:

- spatial discretisation
- time discretisation
- grid definition.

# Implementation

The numerical solver is implemented in Python using Jupyter Notebook.

Main steps:

1. Define computational grid
2. Apply boundary conditions
3. Implement time-stepping scheme
4. Compute numerical solution
5. Store results


# Results and Visualisation

The results are visualised using multiple plotting methods.

Examples:

- Line plots
- Surface plots
- Contour plots
- Heat maps

These plots illustrate the evolution of the numerical solution.

