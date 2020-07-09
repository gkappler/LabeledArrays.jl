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




using LinearAlgebra

import Base: sum
import LinearAlgebra: Adjoint
Base.sum(x::Adjoint; dims=:) =
    if dims isa Colon
        sum(x.parent,dims=:)'
    elseif dims==1
        sum(x.parent,dims=2)'
    else
        sum(x.parent,dims=1)'
    end


import StatsBase: mean
import Printf: @printf

function Base.show(io::IO, x::LabeledArray)
    Base.show(io, MIME"text/plain"(), x)
end

compactstring(x) =
    let io = IOBuffer()
        print(IOContext(io, :compact => true),x)
        String(take!(io))
    end

function Base.show(io::IO, ::MIME"text/plain", x::LabeledArray)
    nzs = x.counts.>0
    print(io, "",
          nnz(x.counts), " observation counts")
    @printf(io," (%d %s x %d %s). ", size(x.counts)[1], x.row,
            size(x.counts)[2], x.col)
    crow = compactstring(x.row)
    ccol = compactstring(x.col)
    @printf(io, "%.2f %s/%s (unique %.2f) on average. %d %ss have no observed %s. %d observed %ss have no %s.\n",
            mean(sum(x.counts,dims=2)),
            ccol,
            crow,
            mean(sum(nzs,dims=2)),
            sum(iszero.(sum(nzs,dims=2))),
            crow,
            ccol,
            sum(iszero.(sum(nzs,dims=1))),
            ccol,
            crow
            )
end

end # module
