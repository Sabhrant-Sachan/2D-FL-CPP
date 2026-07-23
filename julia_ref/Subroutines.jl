module Subroutines

using LinearAlgebra

export getF1W, fejer1w, newtonR2D, newtonR1D


"""
#Code to test the above function: 

f(x)  = cos(x) - x; df(x) = -sin(x) - 1

root1 = newtonR1D(f, df, 1, 50)

println("root1 = ", root1)    # ≈ 0.7390851332151607

println("|f(root1)| = ", abs(f(root1)))  

g(x)  = x^3 - 2; dg(x) = 3x^2

root2 = newtonR1D(g, dg, 1.0, 50)

println("root2 = ", root2)   # ≈ 1.2599210498948732

println("|g(root2)| = ", abs(g(root2)))  
"""

function newtonR2D(f1, f2, Jinv, t0::Float64, s0::Float64, maxi::Int; tol=1e-14)
    t = t0
    s = s0
    iter = 0
    while iter < maxi
        F1 = f1(t, s)
        F2 = f2(t, s)
        J11, J12, J21, J22 = Jinv(t, s) 

        # solve J * Δ = -F
        detJ = J11*J22 - J12*J21
        Δt   = (-F1*J22 + F2*J12) / detJ
        Δs   = (-F2*J11 + F1*J21) / detJ

        t += Δt
        s += Δs

        if max(abs(Δt), abs(Δs)) < tol
            return t, s
        end
        iter += 1
    end

    return :max, :max  
end

"""
#Code to test the above function: 

f1(x,y) = sin(y)+x+1; f2(x,y) = cos(x)+y-1;

x0 = pi/2; y0 = 0.0; maxi = 300;

detJ(x,y) = 1 .+ sin.(x).*cos.(y) 

Jfinv(x,y) = (1 / detJ(x,y)) .* [1 sin(x);-cos(y) 1]

x, y = newtonR2D(f1, f2, Jfinv, x0, y0, maxi)
"""

const F1W_PATH = joinpath(@__DIR__, "..", "F1W.bin")

"""
getF1W(n; file=F1W_PATH) -> Vector{Float64}

Read n Fejér-1 weights from packed file written by `makeF1W`.
"""
function getF1W(n::Int; file::AbstractString=F1W_PATH)
    offset_entries = n*(n - 1) ÷ 2         # 0-based start index
    buf = Vector{Float64}(undef, n)
    open(file, "r") do io
        seek(io, Int64(offset_entries) * 8) # byte offset
        read!(io, buf)
    end
    return buf
end

"""
Fejér type-1 quadrature weights for n nodes on [-1,1].
"""
function fejer1w(n::Int)
    w = zeros(Float64, n)
    m = (n - 1) ÷ 2
    for k in 1:n
        s = 0.0
        for j in 1:m
            s += cospi(2*j*(2*k - 1)/(2*n)) / (4*j^2 - 1)
        end
        w[k] = 2 * (1 - 2*s)/n
    end
    return w
end
"""
Write packed Fejér-1 weights for n=1..N into `file` (Float64).
Total length = N*(N+1)//2 doubles. 
n=1:  [w₁]
n=2:  [w₁ w₂]
n=3:  [w₁ w₂ w₃]
n=4:  [w₁ w₂ w₃ w₄] (concatenate all to give a large column vec)
File vector = [ w₁ | w₁ w₂ | w₁ w₂ w₃ | w₁ w₂ w₃ w₄ ]
"""
function makeF1W(; N::Int=2048, file::AbstractString="F1W.bin")
    open(file, "w") do io
        for n in 1:N
            w = fejer1w(n)
            write(io, w)  # writes as Float64 in column-major order
        end
    end
    return nothing
end

end