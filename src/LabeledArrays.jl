module LabeledArrays

import SparseArrays: nnz
using LinearAlgebra
import LinearAlgebra: Adjoint
nnz(x::LinearAlgebra.Adjoint) =
    nnz(x.parent)


export LabeledArray
"""
A matrix, `AbstractArray{T,2}`, with a layer labeled rows and cols.
"""
struct LabeledMatrix{T,A <: AbstractArray{T,2},R,C} <: AbstractArray{T,2}
    row::R
    col::C
    values::A
    function LabeledMatrix(row,col,values::A) where {A<:AbstractArray}
        new{eltype(values),A,typeof(row),typeof(col)}(row,col,values)
    end
end
export collapse_observations
function collapse_observations(f::Function,x,T=Any) 
   nb = NBijection(T[])
    cols = Int32[  get!(nb,f(e))
                   for e in values(x)
                   ]
    LabeledMatrix(x,nb,
                 sparse(1:length(cols),cols,
                        ConstArray(1,length(cols)),
                        length(cols),length(nb)))
end
export row
row(x::LabeledMatrix,y) = row(x.col,y)
"""
    row(x::NBijection,y)

See also [`LabeledMatrix`](@ref).
"""
function row(x::NBijection,y)
    idxs = filter(x->x!==nothing,indexin(y,x))
    sparsevec(idxs,ConstArray(1,length(idxs)),length(x))
end


Base.getindex(x::LabeledMatrix, a...) = x.values[a...]
Base.getindex(x::LabeledMatrix, row::NumID, a...) =
    if match(x.row,row)
        x.values[get(row),a...]
    else
        error("invalid index: $row for $(x.row)")
    end

import Base: +, *, adjoint, size
Base.size(x::LabeledMatrix) =
    Base.size(x.values)
adjoint(x::LabeledMatrix) =
    LabeledMatrix(x.col, x.row, x.values')
@deprecate reverse(x::LabeledMatrix) adjoint(x)

function *(x::LabeledMatrix, y::LabeledMatrix)
    @assert x.col == y.row
    LabeledMatrix(
        x.row, y.col,
        x.values * y.values)
end
*(x::LabeledMatrix, y::AbstractArray) = x.values*y
*(x::AbstractArray, y::LabeledMatrix) = x*y.values
*(x::LabeledMatrix, y::Number) =
    LabeledMatrix(
        x.row, x.col,
        x.values*y)
*(x::Number, y::LabeledMatrix) =
    LabeledMatrix(
        y.row, y.col,
        x*y.values)

function +(x::LabeledMatrix, y::LabeledMatrix)
    @assert x.row == y.row
    @assert x.col == y.col
    LabeledMatrix(
        x.row, y.col,
        x.values + y.values)
end
+(x::LabeledMatrix, y) = x.values+y
+(x, y::LabeledMatrix) = x+y.values
+(x::LabeledMatrix, y::Number) =
    LabeledMatrix(
        x.row, x.col,
        x.values+y)
+(x::Number, y::LabeledMatrix) =
    LabeledMatrix(
        y.row, y.col,
        x+y.values)



export bool
"TODO: broadcasting"
bool(A::LabeledMatrix) =
    LabeledMatrix(A.row, A.col, A.values.>0)


import StatsBase: mean


Base.sum(x::LabeledMatrix; kw...) =
    sum(x.values; kw...)
using LinearAlgebra
import LinearAlgebra: Adjoint
Base.sum(x::Adjoint; dims=:) =
    if dims isa Colon
        sum(x.parent,dims=:)'
    elseif dims==1
        sum(x.parent,dims=2)'
    else
        sum(x.parent,dims=1)'
    end

 
import Printf: @printf
function Base.show(io::IO, x::LabeledMatrix)
    Base.show(io, MIME"text/plain"(), x)
end
compactstring(x) =
    let io = IOBuffer()
        print(IOContext(io, :compact => true),x)
        String(take!(io))
    end
function Base.show(io::IO, ::MIME"text/plain", x::LabeledMatrix)
    nzs = x.values.>0
    @printf(io,"%d x %d: %s x %s. ", size(x.values)[1], 
            size(x.values)[2], x.row, x.col)
    @printf(io, "%d observation values, %2.4f%% sparse. ",
            nnz(x.values), (1.0-nnz(x.values)/*(size(x.values)...))*100.0)
    crow = compactstring(valtype(x.row))
    ccol = compactstring(valtype(x.col))
    S=sum(x.values,dims=2)
    @printf(io, "On average %.2f observed %ss/%s (unique %.2f).",
            mean(S[S.>0]),
            ccol,
            crow,
            mean(sum(nzs,dims=2)))
    zeros=sum(iszero.(sum(nzs,dims=2)))
    if zeros > 0
        print(io," $zeros $(crow)s have no observed $ccol.")
    end
    zeros=sum(iszero.(sum(nzs,dims=1)))
    if zeros > 0
        print(io," $zeros observed $(ccol)s have no observed $crow.")
    end
end

end # module
