#= Optimization problem constraint object.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington),
                   and Autonomous Systems Laboratory (Stanford University)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

import JuMP: dual

export value, name, dual, cone, lhs, jacobian, all_jacobians, kind, variables, parameters

# ..:: Globals ::..

# Specialize arguments to variables and parameters
const VariableArgument = Argument{AtomicVariable}
const VariableArgumentBlocks = ArgumentBlocks{AtomicVariable}
const ConstantArgument = Argument{AtomicConstant}
const ConstantArgumentBlocks = ArgumentBlocks{AtomicConstant}

# ..:: Data structures ::..

"""
`ProgramFunction` defines an affine or quadratic function that gets used as a
building block in the conic program. The function accepts a list of argument
blocks (variables and parameters) that it depends on, and the underlying
`DifferentiableFunction` function that performs the value and Jacobian
computation.

By wrapping `DifferentiableFunction` in this way, the aim is to store the
variables and parameters that this otherwise generic "mathematical object" (the
function) depends on. With this information, we can automatically compute
Jacobians for the KKT conditions.
"""
struct ProgramFunction
    f::DifferentiableFunction # The core computation method
    x::VariableArgumentBlocks # Variable arguments
    p::ConstantArgumentBlocks # Constant arguments
    prog::Ref{AbstractConicProgram} # The parent conic program

    """
        ProgramFunction(prog, x, p)

    General (affine or quadratic) function constructor.

    # Arguments
    - `prog`: the optimization program.
    - `x`: the variable argument blocks.
    - `p`: the parameter argument blocks.
    - `f`: the core method that can compute the function value and its
      Jacobians.

    # Returns
    - `F`: the function object.
    """
    function ProgramFunction(
        prog::AbstractConicProgram,
        x::VariableArgumentBlocks,
        p::ConstantArgumentBlocks,
        f::Function,
    )::ProgramFunction

        # Create the differentiable function wrapper
        xargs = length(x)
        pargs = length(p)
        consts = prog.pars[]
        f = DifferentiableFunction(f, xargs, pargs, consts)

        F = new(f, x, p, prog)

        return F
    end
end # struct

"""
`ConicConstraint` defines a conic constraint for the optimization program. It
is basically the following mathematical object:

```math
f(x, p)\\in \\mathcal K
```

where ``f`` is an affine function and ``\\mathcal K`` is a convex cone.
"""
struct ConicConstraint
    f::ProgramFunction              # The affine function
    K::ConvexCone                   # The convex cone set data structure
    constraint::Types.Constraint    # Underlying JuMP constraint
    prog::Ref{AbstractConicProgram} # The parent conic program
    name::String                    # A name for the constraint

    """
        ConicConstraint(f, kind, prog[; name])

    Basic constructor. The function `f` value gets evaluated.

    # Arguments
    - `f`: the affine function which must belong to some convex cone.
    - `kind`: the kind of convex cone that is to be used. All the sets allowed
      by `ConvexCone` are supported.
    - `prog`: the parent conic program to which the constraint belongs.

    # Keywords
    - `name`: (optional) a name for the constraint, which can be used to more
      easily search for it in the constraints list.

    # Returns
    - `finK`: the conic constraint.
    """
    function ConicConstraint(
        f::ProgramFunction,
        kind::Union{SupportedCone,SupportedDualCone},
        prog::AbstractConicProgram;
        name::Types.Optional{String} = nothing,
    )::ConicConstraint

        # Create the underlying JuMP constraint
        f_value = f()
        K = ConvexCone(f_value, kind)
        constraint = add!(jump_model(prog), K)

        if isnothing(name)
            # Assign a default name
            name = "constraint"
        end

        finK = new(f, K, constraint, prog, name)

        return finK
    end
end # struct

const Constraints = Vector{ConicConstraint}

# ..:: Methods ::..

"""
    ProgFunc([; jacobians, scalar])

Compute the function value and (optionally) Jacobians. This basically forwards
data to the underlying `DifferentiableFunction`, which handles the computation.

# Keywords
- `jacobians`: (optional) set to true in order to compute the Jacobians as
  well.
- `scalar`: (optional) whether to output a scalar value.

# Returns
- `f`: the function value. The Jacobians can be queried later by using the
  `jacobian` function.
"""
function (ProgFunc::ProgramFunction)(;
    jacobians::Bool = false,
    scalar::Bool = false,
)::FunctionValueOutputType

    # Compute the input argument values
    x_input = [value(blk) for blk in ProgFunc.x]
    p_input = [value(blk) for blk in ProgFunc.p]
    args = vcat(x_input, p_input)

    f_value = ProgFunc.f(args...; jacobians = jacobians, scalar = scalar) # Core call

    return f_value
end

"""
Convenience methods that pass the calls down to `DifferentiableFunction`.
"""
value(F::ProgramFunction; scalar::Bool = false) = value(F.f; scalar = scalar)
jacobian(F::ProgramFunction, key::JacobianKeys)::JacobianValueType = jacobian(F.f, key)
all_jacobians(F::ProgramFunction)::JacobianDictType = all_jacobians(F.f)

""" Convenience getters. """
program(F::ProgramFunction)::AbstractConicProgram = F.prog[]
variables(F::ProgramFunction)::VariableArgumentBlocks = F.x
parameters(F::ProgramFunction)::ConstantArgumentBlocks = F.p

""" Get the cone. """
cone(C::ConicConstraint)::ConvexCone = C.K

""" Get the kind of cone. """
kind(C::ConicConstraint)::SupportedCone = kind(cone(C))

""" Get the cone name. """
name(C::ConicConstraint)::String = C.name

""" Get the cone dual variable. """
dual(C::ConicConstraint)::Types.RealVector = dual(C.constraint)

""" Get the underlying affine function. """
lhs(C::ConicConstraint)::ProgramFunction = C.f

"""
    function_args_id(F[, args])

Get the input argument indices of `args` for the function `F`. If `args` is
ommitted then a "map" vector which is [1:nx; 1:np] and can be used to map from
the absolute argument index to the index in the list of variable or constant
arguments.

# Arguments
- `F`: the function.
- `args`: (optional) the arguments of the function, as a symbol denoting either
  the variable arguments (`:x`) or the constant arguments (`:p`).

# Returns
The index numbers of the arguments.
"""
function function_args_id(F::ProgramFunction, args::Symbol)::LocationIndices
    nargs = length(getfield(F, args))
    if args == :x
        return 1:nargs
    else
        return (1:nargs) .+ length(getfield(F, :x))
    end
end

function function_args_id(F::ProgramFunction)::LocationIndices
    idx_map = [1:length(variables(F)); 1:length(parameters(F))]
    return idx_map
end
