module LabeledArrays

### A beneficial type piracy, todo: pull-request in SparseArrays
using SparseArrays
import SparseArrays: nnz
using LinearAlgebra
import LinearAlgebra: Adjoint
"""
    nnz(x::LinearAlgebra.Adjoint)

Call `SparseArrays.nnz` of transpose for performance.
"""
nnz(x::LinearAlgebra.Adjoint) =
    nnz(x.parent)


############################################################
export NBijection, EnumerationDict, VectorDict
"""
    NBijection(x::Vector{T})

KEY <-> Int, bijection map. Synonym to possibly better name `EnumerationDict`, or `VectorDict`.

Store inverse indices in a `Dict{T,I}` as well as original `Vector{T}`.
- `copy`: set to false to store `x` without copying.
  (make sure that Vector will be mutated only through `NBijection` API.
"""
struct NBijection{KEY, I<:Integer}
    enumeration::Vector{KEY}
    index::Dict{KEY,I}
end
EnumerationDict{KEY, I<:Integer} = NBijection{KEY, I}
VectorDict{KEY, I<:Integer} = NBijection{KEY, I}
NBijection{KEY, I}() where {I,KEY} =
    NBijection{KEY,I}(KEY[], Dict{KEY,I}())
NBijection(x) = NBijection(collect(x), copy=false)
NBijection(x::Vector{KEY}; copy=true) where KEY =
    NBijection{KEY,Int}(
        copy ? Base.copy(x) : x,
        Dict((t => i for (t,i) = zip(x, 1:length(x)))...))

@deprecate terms(x::NBijection) keys(x)
"""
    keys(b::NBijection)

terms are keys in sequence that share position index with some other 
data structure, like a count matrix in Count.
"""
Base.keys(b::NBijection) = b.enumeration
Base.values(b::NBijection) = 1:length(b)
Base.keytype(x::NBijection{KEY,I}) where {KEY,I} = KEY
Base.valtype(x::NBijection{KEY,I}) where {KEY,I} = I
Base.eltype(x::NBijection{KEY,I}) where {KEY,I} = KEY
Base.show(io::IO,x::NBijection) =
    print(io,"NBijection{",valtype(x),"}: ",x.index)
Base.length(b::NBijection) = length(b.enumeration)
Base.iterate(b::NBijection,a...) =
    iterate(b.enumeration,a...)
"""
    Base.getindex(b::NBijection,i::Integer)

`b.enumeration[i]`.
"""
@inline Base.@propagate_inbounds Base.getindex(b::NBijection,i::Integer) =
    b.enumeration[i]

import Base: match
Base.match(x::NBijection, y) = false

import Base: indexin
Base.indexin(i,b::NBijection{KEY,I}) where {KEY,I} =
    Union{I,Nothing}[ get(b.index,e,nothing) for e in i]




import Base: get!, push!
"""
    Base.get!(f::Function,b::NBijection, key)

Calls `f(length(b),key)` hook function if `key !in b`, push `key` to `b.enumeration` and store `key=>index`.
Similar to `get!(f,b::Dict, key)`.

TODO: remove f(n,key)
```jldoctest
julia> x = NBijection(['a','b','c'])
julia> get!(x,'b')
2

julia> get!((i,k)->println("neu"), x, 'd')
neu
4

julia> get!((i,k)->println("neu"), x, 'd')
4
```
"""
Base.get!(f::Function,b::NBijection, key) =
    get!(b.index, key) do
        push!(b.enumeration, key)
        f(length(b),key)
        length(b)
    end

"""
    Base.get!(b::NBijection, key)

When `key !in b.index`, push `nothing` to `b.enumeration` and store `key=>index`.
```jldoctest
julia> x = NBijection(['a','b','c'])
julia> get!(x,'b')
2

julia> get!(x,'d')
4

julia> get!(x,'d')
4

```
"""
Base.get!(b::NBijection, key) =
    get!((_,_)->nothing,b, key)
@deprecate get_index!(b::NBijection, key) get!(b,key)

"""
    Base.push!(b::NBijection, key)

Push `key` to `b.enumeration` and store `key=>index`.
Return last index.

@asserts that key is not in index, fails otherwise.
Consider [`get!`](@ref).
"""
function Base.push!(b::NBijection, key)
    @assert !haskey(b.index,key)
    push!(b.enumeration,key)
    b.index[key] = length(b.enumeration)
end

"""
    sparse_vector(x::NBijection,sel)

Create a sparse vector representation of `sel`ected values in `x::`[`NBijection`](@ref).
"""
function SparseArrays.sparsevec(x::NBijection,sel)
    idxs = filter(
        x->x!==nothing,
        indexin(sel,x))
    sparsevec(
        idxs,
        ConstArray(1,length(idxs)),
        length(x))
end






export ConstArray
struct ConstArray{T,N,l} <: AbstractArray{T,N}
    value::T
    ConstArray(x::T, l::Integer...) where T =
        new{T,length(l),l}(x)
end
Base.size(::ConstArray{T,N,l}) where {T,N,l}	= l
Base.getindex(A::ConstArray, i::Int...) = A.value
Base.setindex!(A::ConstArray, v, i::Int) = error("create a normal array and convert")
Base.sum(A::ConstArray) = A.value * (*(size(A)...)) 

export LabeledMatrix
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
export sparsevec
SparseArrays.sparsevec(x::LabeledMatrix,y) = sparsevec(x.col,y)
@deprecate row(x,y) sparsevec(x,y)
# ? col, row ?
# SparseArrays.sparsecol(x::LabeledMatrix,y) = sparsevec(x.row,y)


Base.getindex(x::LabeledMatrix, a...) = x.values[a...]
Base.getindex(x::LabeledMatrix, row, a...) =
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


import Base: isless
Base.isless(than::Int, A::LabeledMatrix) =
    LabeledMatrix(A.row, A.col, A.values.>than)

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
            sum(S[S.>0])/length(S),
            ccol,
            crow,
            sum(sum(nzs,dims=2))/size(nzs)[2])
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
