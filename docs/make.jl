push!(LOAD_PATH,"../src")
using Documenter: Documenter, makedocs, deploydocs, doctest, DocMeta
using LabeledArrays
using Test
using Literate

docdir = joinpath(dirname(pathof(LabeledArrays)),"../docs/src/")
mandir = joinpath(docdir,"man")

DocMeta.setdocmeta!(LabeledArrays, :DocTestSetup, quote
    using LabeledArrays
end; recursive=true)

doctest(LabeledArrays; fix=true)

makedocs(;
         source = docdir,
         modules=[LabeledArrays],
         authors="Gregor Kappler",
         # repo="https://github.com/gkappler/CombinedParsers.jl/blob/{commit}{path}#L{line}",
         sitename="LabeledArrays.jl",
         # format=Documenter.HTML(;
         #                        prettyurls=get(ENV, "CI", "false") == "true",
         #                        canonical="https://gkappler.github.io/CombinedParsers.jl",
         #                        assets=String[],
         #                        ),
         pages=[
             "Home" => "index.md",
             # "Library" => Any[
             #     "Public" => "lib/public.md",
             #     ##"Internals" => "lib/internals.md"
             # ],
             # "Developer Guide" => "developer.md"
         ]
         )

deploydocs(;
    repo="github.com/gkappler/LabeledArrays.jl",
)
