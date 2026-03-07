#Project purpose

    This repository is an ME2 Computing coursework project.
    The code must implement numerical methods taught in the course and follow the same mathematical notation used in the lectures.

    Focus on clarity, correctness, and traceability rather than advanced optimisation.

#Core constraints

    ##Build the project from scratch in stages

        Do not write one large script.

        Always follow this order:

        1, Propose and create the folder structure.

        2, Create module files.

        3, Implement one function at a time.

        4, Add a small test or verification after each function.

        5, Connect the functions in a main script.

        6, Add plotting and post-processing last.

        7, Prefer small commits and small changes.

    ##Only use numerical methods taught in the course

        Allowed methods listed below.

        Numerical integration:

        trapezium rule

        Simpson rule

        Gauss integration

        Interpolation:

        Lagrange interpolation

        Newton interpolation

        cubic splines

        ODE initial value problems:

        Forward Euler

        Backward Euler

        RK4

        ODE boundary value problems:

        finite difference method

        shooting method

        PDE:

        explicit finite difference

        implicit finite difference

        Do not use methods outside the course unless the user explicitly asks.

    ##Code must be beginner-friendly

        Rules:

        Python only

        keep functions small

        use clear variable names

        prefer simple loops over complex expressions

        write short docstrings for functions

        avoid unnecessary abstractions

        code must be easy for a beginner to understand

    ##Avoid advanced external libraries

        Allowed libraries:

        numpy

        matplotlib

        Not allowed unless explicitly requested:

        scipy solvers

        symbolic solvers

        finite element libraries

        machine learning tools

        black-box PDE solvers

        All numerical algorithms should be implemented directly in code, not delegated to external solvers.

    #Implementation style

        When implementing a numerical method:

        Write the continuous equation in comments.

        Write the discretised equation in comments.

        Implement the algorithm clearly step by step.

        Use variable names that match the mathematical notation where practical. avoid special symbols.

