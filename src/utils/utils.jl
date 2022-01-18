#= Trajectory generation utilities.

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

module Utils

export Types

include("globals.jl")

function skew end
function linterp end
function zohinterp end
function diracinterp end

module Types
include("basic_types.jl")
include("quaternion.jl")
include("ellipsoid.jl")
include("hyperrectangle.jl")
include("dltv.jl")
include("trajectory.jl")
include("homotopy.jl")
include("table.jl")
include("tree.jl")
end

using .Types
export Quaternion, dcm, rpy, slerp_interpolate, rotate, Log, skew, ddq
export Ellipsoid, project, ∇
export Hyperrectangle
export DLTV
export ContinuousTimeTrajectory, sample
export Homotopy
export Table, improvement_percent

include("helper.jl")
include("plots.jl")
include("scene_3d.jl")

end
