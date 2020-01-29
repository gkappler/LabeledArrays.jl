module LabeledArrays

import SparseArrays: nnz
using LinearAlgebra
import LinearAlgebra: Adjoint
nnz(x::LinearAlgebra.Adjoint) =
    nnz(x.parent)

export LabeledArray

struct LabeledArray{T,A <: AbstractArray{T,2},R,C} <: AbstractArray{T,2}
    row::R
    col::C
    counts::A
    function LabeledArray(row,col,counts::A) where {A<:AbstractArray}
        new{eltype(counts),A,typeof(row),typeof(col)}(row,col,counts)
    end
end

import Base: +, *, adjoint, size

Base.size(x::LabeledArray) = Base.size(x.counts)

export bool
"TODO: broadcasting"
bool(A::LabeledArray) =
    LabeledArray(A.row, A.col, A.counts.>0)

function adjoint(x::LabeledArray)
    LabeledArray(x.col, x.row, x.counts')
end

function *(x::LabeledArray, y::LabeledArray)
    @assert x.col == y.row
    LabeledArray(
        x.row, y.col,
        x.counts * y.counts)
end

function +(x::LabeledArray, y::LabeledArray)
    @assert x.row == y.row
    @assert x.col == y.col
    LabeledArray(
        x.row, y.col,
        x.counts + y.counts)
end

function Base.reverse(x::LabeledArray)
    LabeledArray(x.col, x.row, x.counts')
end



Base.getindex(x::LabeledArray, a...) = x.counts[a...]



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
