module Subroutines

using LinearAlgebra

export getF1W, fejer1w, newtonR2D, newtonR1D,
       Bis, GSS, dlineseg2D, nevill, diffpowers,
       ChebyT, ChebyT!, ChebyTN, ChebyTN!,
       wfunc, dwfunc, wfunc!, dwfunc!, wsfunc, wsfunc!,
       ddwfunc, ddwfunc!, qw1func, qw1func!,
       qw2func, qw2func!, qw3func, qw3func!,
       wfofunc, dwfofunc, winv!, winv, lineinter, ptinqua,
       tanginterp, tanginterx, extendqua!,
       ellipselinter, ellipserinter, nevill!, ptinqua!

"""
    ellipserinter(Cx,Cy, R1,R2, e1x,e1y, e2x,e2y, ux,uy, vx,vy; tol=1e-14) -> Et, Lt

Intersection of the ray `L(t) = u + t*v` (t ≥ 0) with the ellipse
`E(θ) = C + R1*cos(θ)*e1 + R2*sin(θ)*e2`, with 2D inputs.

Returns `(Et, Lt)`; if no real ray intersection, both `NaN`.

Allocation-free (no temporary vectors, no filter/minimum).
"""
function ellipserinter(Cx::Float64, Cy::Float64,
    R1::Float64, R2::Float64, e1x::Float64, e1y::Float64,
    e2x::Float64, e2y::Float64, ux::Float64, uy::Float64,
    vx::Float64, vy::Float64; tol::Float64=1e-14)::Tuple{Float64,Float64}

    invR1 = 1.0 / R1
    invR2 = 1.0 / R2

    # Projections in normalized/rotated coords
    dux = ux - Cx
    duy = uy - Cy

    a1 = (dux*e1x + duy*e1y) * invR1
    a2 = (dux*e2x + duy*e2y) * invR2
    b1 = (vx*e1x  + vy*e1y)  * invR1
    b2 = (vx*e2x  + vy*e2y)  * invR2

    # Quadratic A t^2 + B t + Cq = 0
    A  = b1*b1 + b2*b2
    B  = 2.0*(a1*b1 + a2*b2)
    Cq = a1*a1 + a2*a2 - 1.0

    # Degenerate direction
    if A == 0.0
        return (NaN, NaN)
    end

    # Discriminant
    Δ = B*B - 4.0*A*Cq
    if Δ < -tol
        return (NaN, NaN)
    elseif Δ < 0.0
        Δ = 0.0
    end

    sD = sqrt(Δ)

    # Stable (Kahan) roots
    q = -0.5 * (B + (B >= 0.0 ? +sD : -sD))
    if q == 0.0
        return (NaN, NaN)
    end

    Lt1 = q / A
    Lt2 = Cq / q

    # Choosing smallest finite t >= 0 without allocating
    t1ok = isfinite(Lt1) & (Lt1 >= 0.0)
    t2ok = isfinite(Lt2) & (Lt2 >= 0.0)

    if !(t1ok | t2ok)
        return (NaN, NaN)
    end

    Lt = if t1ok
        (t2ok ? min(Lt1, Lt2) : Lt1)
    else
        Lt2
    end

    # cos θ = a1 + t*b1
    CC = a1 + Lt*b1

    # snap near ±1
    if abs(CC + 1.0) < tol
        CC = -1.0
    elseif abs(CC - 1.0) < tol
        CC =  1.0
    end

    # safety clamp (still alloc-free)
    CC = ifelse(CC > 1.0, 1.0, ifelse(CC < -1.0, -1.0, CC))

    Et = acos(CC)

    # Disambiguate θ vs 2π-θ by comparing squared distances to the hit point X = u + t v
    Xx = ux + Lt*vx
    Xy = uy + Lt*vy

    c = cos(Et)
    s = sin(Et)

    # g(Et)
    Gpx = Cx + (R1*c)*e1x + (R2*s)*e2x
    Gpy = Cy + (R1*c)*e1y + (R2*s)*e2y
    dxp = Gpx - Xx
    dyp = Gpy - Xy
    d2p = dxp*dxp + dyp*dyp

    # g(-Et)  (cos same, sin flips)
    Gmx = Cx + (R1*c)*e1x + (R2*(-s))*e2x
    Gmy = Cy + (R1*c)*e1y + (R2*(-s))*e2y
    dxm = Gmx - Xx
    dym = Gmy - Xy
    d2m = dxm*dxm + dym*dym

    if d2p > d2m
        Et = 2.0 * π - Et
    end

    return Et, Lt
end


"""
    ellipselinter(Cx,Cy, R1,R2, e1x,e1y, e2x,e2y, ux,uy, vx,vy; tol=1e-14) -> Et1, Lt1, Et2, Lt2

Intersection between a line `L(t) = u + t*v` and an ellipse
`E(θ) = C + R1*cos(θ)*e1 + R2*sin(θ)*e2`, with 2D inputs.

Returns `(Et1, Lt1, Et2, Lt2)`; if no real intersection, all `NaN`.

Allocation-free (no temporary vectors).
"""
function ellipselinter(Cx::Float64, Cy::Float64,
    R1::Float64, R2::Float64, e1x::Float64, e1y::Float64,
    e2x::Float64, e2y::Float64, ux::Float64, uy::Float64,
    vx::Float64, vy::Float64; tol::Float64=1e-14)::Tuple{Float64,Float64,Float64,Float64}

    invR1 = 1.0 / R1
    invR2 = 1.0 / R2

    # projections in rotated/normalized coordinates
    dux = ux - Cx
    duy = uy - Cy

    a1 = (dux*e1x + duy*e1y) * invR1
    a2 = (dux*e2x + duy*e2y) * invR2
    b1 = (vx*e1x + vy*e1y)  * invR1
    b2 = (vx*e2x + vy*e2y)  * invR2

    # Quadratic A t^2 + B t + Cq = 0
    A  = b1*b1 + b2*b2
    B  = 2.0*(a1*b1 + a2*b2)
    Cq = a1*a1 + a2*a2 - 1.0

    # Degenerate line direction in normalized frame
    if A == 0.0
        return (NaN, NaN, NaN, NaN)
    end

    D = B*B - 4.0*A*Cq
    if D < -tol
        return (NaN, NaN, NaN, NaN)
    elseif D < 0.0
        D = 0.0
    end

    sD = sqrt(D)

    # Kahan trick: compute a well-conditioned numerator first
    q = -0.5 * (B + (B >= 0.0 ? +sD : -sD))
    if q == 0.0
        return (NaN, NaN, NaN, NaN)
    end

    Lt1 = q / A
    Lt2 = Cq / q

    if !(isfinite(Lt1) && isfinite(Lt2))
        return (NaN, NaN, NaN, NaN)
    end

    # cos(θ) = a1 + t*b1
    CC1 = a1 + Lt1*b1
    CC2 = a1 + Lt2*b1

    # snap near ±1 to remove acos jitter
    if abs(CC1 + 1.0) < tol
        CC1 = -1.0
    elseif abs(CC1 - 1.0) < tol
        CC1 =  1.0
    end
    if abs(CC2 + 1.0) < tol
        CC2 = -1.0
    elseif abs(CC2 - 1.0) < tol
        CC2 =  1.0
    end

    # safety clamp (still alloc-free)
    CC1 = ifelse(CC1 > 1.0, 1.0, ifelse(CC1 < -1.0, -1.0, CC1))
    CC2 = ifelse(CC2 > 1.0, 1.0, ifelse(CC2 < -1.0, -1.0, CC2))

    Et1 = acos(CC1)
    Et2 = acos(CC2)

    # Pick θ vs 2π-θ by comparing distance to the actual intersection X = u + t v.
    # We compare squared distances to avoid sqrt.

    # ---- intersection point X1 ----
    X1x = ux + Lt1*vx
    X1y = uy + Lt1*vy

    # g(Et1)
    s, c = sincos(Et1)

    Gpx = Cx + (R1*c)*e1x + (R2*s)*e2x
    Gpy = Cy + (R1*c)*e1y + (R2*s)*e2y
    dxp = Gpx - X1x
    dyp = Gpy - X1y
    d2p = dxp*dxp + dyp*dyp

    # g(-Et1)  (cos same, sin flips)
    Gmx = Cx + (R1*c)*e1x + (R2*(-s))*e2x
    Gmy = Cy + (R1*c)*e1y + (R2*(-s))*e2y
    dxm = Gmx - X1x
    dym = Gmy - X1y
    d2m = dxm*dxm + dym*dym

    if d2p > d2m
        Et1 = 2.0 * π - Et1
    end

    # ---- intersection point X2 ----
    X2x = ux + Lt2*vx
    X2y = uy + Lt2*vy

    # g(Et2)
    s, c = sincos(Et2)

    Gpx = Cx + (R1*c)*e1x + (R2*s)*e2x
    Gpy = Cy + (R1*c)*e1y + (R2*s)*e2y
    dxp = Gpx - X2x
    dyp = Gpy - X2y
    d2p = dxp*dxp + dyp*dyp

    # g(-Et2)
    Gmx = Cx + (R1*c)*e1x + (R2*(-s))*e2x
    Gmy = Cy + (R1*c)*e1y + (R2*(-s))*e2y
    dxm = Gmx - X2x
    dym = Gmy - X2y
    d2m = dxm*dxm + dym*dym

    if d2p > d2m
        Et2 = 2.0 * π - Et2
    end

    return Et1, Lt1, Et2, Lt2
end

"""
#The above two functions can be checked by following code
C = [0; 0]; tht = 0;

e1 = [cos(tht); sin(tht)];

e2 = [-sin(tht); cos(tht)];

P1 = [-1; -3]; P2 = [2; 3];

u = P1; v = P2-P1;

R1 = 2; R2 = 1;

Et1,Lt1,Et2,Lt2 = ellipselinter(C,R1,R2,e1,e2,u,v)

Et,Lt = ellipserinter(C,R1,R2,e1,e2,u,v)

gx(t) = C[1].+R1.*cos.(t).*e1[1].+R2.*sin.(t).*e2[1]

gy(t) = C[2].+R1.*cos.(t).*e1[2].+R2.*sin.(t).*e2[2]

t = range(0,2π,length=101)

x = gx(Et); y = gy(Et); 

plt = plot(; legend=false, linewidth=2, aspect_ratio=1)

plot!(plt, gx(t),gy(t))

plot!(plt, [u[1]; x],[u[2]; y], marker =:circle, arrow=true)
"""

"""
    extendqua(PQ::AbstractVector{<:Float64}, δ::Float64) -> Vector{Float64}

Extend a convex quadrilateral by distance `δ` along its outward normals.

`PQ` packs the quad vertices as an 8-vector:
    [x1, y1, x2, y2, x3, y3, x4, y4]
with edges (p1→p2→p3→p4→p1). Returns the extended quad as the same 8-vector
`[P1x, P1y, P2x, P2y, P3x, P3y, P4x, P4y]`.

Notes:
- Outward normals are taken as `[ey, -ex]/‖e‖` and flipped as needed using
  `sign(cross(e1,e2).z)` so they indeed point outward.
- Requires the helpers `lineinter` and `tanginterx` defined earlier.
"""
function extendqua!(Pout::AbstractVector{Float64},
                    PQ::AbstractVector{Float64},
                    δ::Float64)

    # Unpack vertices: p1, p2, p3, p4
    p1x, p1y = PQ[1], PQ[2]
    p2x, p2y = PQ[3], PQ[4]
    p3x, p3y = PQ[5], PQ[6]
    p4x, p4y = PQ[7], PQ[8]

    # Edge vectors
    e1x = p2x - p1x; e1y = p2y - p1y
    e2x = p3x - p2x; e2y = p3y - p2y
    e3x = p4x - p3x; e3y = p4y - p3y
    e4x = p1x - p4x; e4y = p1y - p4y

    # Lengths
    len1 = sqrt(e1x*e1x + e1y*e1y)
    len2 = sqrt(e2x*e2x + e2y*e2y)
    len3 = sqrt(e3x*e3x + e3y*e3y)
    len4 = sqrt(e4x*e4x + e4y*e4y)

    # Unit normals (initial orientation): (ey, -ex) / ||e||
    n1x =  e1y / len1;  n1y = -e1x / len1
    n2x =  e2y / len2;  n2y = -e2x / len2
    n3x =  e3y / len3;  n3y = -e3x / len3
    n4x =  e4y / len4;  n4y = -e4x / len4

    # Orientation: z-component of cross([e1x,e1y,0], [e2x,e2y,0])
    z = e1x*e2y - e1y*e2x
    if z < 0.0
        # normals were inward → flip all
        n1x = -n1x; n1y = -n1y
        n2x = -n2x; n2y = -n2y
        n3x = -n3x; n3y = -n3y
        n4x = -n4x; n4y = -n4y
    end

    # Offset each side by δ along its normal (two points per offset line)
    EQL1p1x = p1x + δ*n1x;  EQL1p1y = p1y + δ*n1y
    EQL1p2x = p2x + δ*n1x;  EQL1p2y = p2y + δ*n1y

    EQL2p1x = p2x + δ*n2x;  EQL2p1y = p2y + δ*n2y
    EQL2p2x = p3x + δ*n2x;  EQL2p2y = p3y + δ*n2y

    EQL3p1x = p3x + δ*n3x;  EQL3p1y = p3y + δ*n3y
    EQL3p2x = p4x + δ*n3x;  EQL3p2y = p4y + δ*n3y

    EQL4p1x = p4x + δ*n4x;  EQL4p1y = p4y + δ*n4y
    EQL4p2x = p1x + δ*n4x;  EQL4p2y = p1y + δ*n4y

    # Intersections (tangent-based):
    #   P2 = (offset side 1) ∩ (offset side 2)
    #   P3 = (offset side 2) ∩ (offset side 3)
    #   P1 = (offset side 4) ∩ (offset side 1)
    #   P4 = (offset side 3) ∩ (offset side 4)

    P2x, P2y, P3x, P3y = tanginterx(
        EQL1p1x, EQL1p1y, EQL1p2x, EQL1p2y,
        EQL2p1x, EQL2p1y,
        EQL2p2x - EQL2p1x, EQL2p2y - EQL2p1y,
        EQL3p1x, EQL3p1y, EQL3p2x, EQL3p2y,
    )

    P1x, P1y, P4x, P4y = tanginterx(
        EQL1p1x, EQL1p1y, EQL1p2x, EQL1p2y,
        EQL4p1x, EQL4p1y,
        EQL4p2x - EQL4p1x, EQL4p2y - EQL4p1y,
        EQL3p1x, EQL3p1y, EQL3p2x, EQL3p2y,
    )

    # Write out [P1, P2, P3, P4] in order
    Pout[1] = P1x; Pout[2] = P1y
    Pout[3] = P2x; Pout[4] = P2y
    Pout[5] = P3x; Pout[6] = P3y
    Pout[7] = P4x; Pout[8] = P4y

    return nothing 
end

"""
    tanginterx(L1p1, L1p2, Tp, Tm, L3p1, L3p2) -> (P1x, P1y, P2x, P2y)

Given three lines:
- L1 through `L1p1 -> L1p2`
- L2 the tangent line through `Tp -> Tp+Tm`
- L3 through `L3p1 -> L3p2`

Return the intersection points in real space:
- `P1 = L1 ∩ L2` as `(P1x, P1y)`
- `P2 = L3 ∩ L2` as `(P2x, P2y)`

All inputs are 2-vectors (e.g. `[x, y]`).
"""
@inline function tanginterx(
    L1x1::Float64, L1y1::Float64, L1x2::Float64, L1y2::Float64,
    Tpx::Float64, Tpy::Float64, Tmx::Float64, Tmy::Float64,
    L3x1::Float64, L3y1::Float64, L3x2::Float64, L3y2::Float64)

    # Tangent line endpoints (L2)
    L2x1 = Tpx
    L2y1 = Tpy
    L2x2 = Tpx + Tmx
    L2y2 = Tpy + Tmy

    # Intersection parameter along L1
    v, _ = lineinter(
        L1x1, L1y1, L1x2, L1y2,
        L2x1, L2y1, L2x2, L2y2)

    P1x = L1x1 + v * (L1x2 - L1x1)
    P1y = L1y1 + v * (L1y2 - L1y1)

    # Intersection parameter along L3
    u, _ = lineinter(
        L3x1, L3y1, L3x2, L3y2,
        L2x1, L2y1, L2x2, L2y2)

    P2x = L3x1 + u * (L3x2 - L3x1)
    P2y = L3y1 + u * (L3y2 - L3y1)

    return P1x, P1y, P2x, P2y
end


"""
    tanginterp(L1p1, L1p2, Tp, Tm, L3p1, L3p2) -> (u, v)

Parameter-space intersections (no coordinates):

For the same lines as `tanginterx`, return a tuple
`S = (u, v)` where:
- `v` is the parameter of the intersection on line L1
- `u` is the parameter of the intersection on line L3

All inputs are 2-vectors (e.g. `[x, y]`).
"""
@inline function tanginterp(
    L1x1::Float64, L1y1::Float64, L1x2::Float64, L1y2::Float64,
    Tpx::Float64, Tpy::Float64, Tmx::Float64, Tmy::Float64,
    L3x1::Float64, L3y1::Float64, L3x2::Float64, L3y2::Float64)
    # Tangent line endpoints (L2)
    L2x1 = Tpx
    L2y1 = Tpy
    L2x2 = Tpx + Tmx
    L2y2 = Tpy + Tmy

    # v = param on L1 at intersection with L2
    v, _ = lineinter(
        L1x1, L1y1, L1x2, L1y2,
        L2x1, L2y1, L2x2, L2y2)

    # u = param on L3 at intersection with L2
    u, _ = lineinter(
        L3x1, L3y1, L3x2, L3y2,
        L2x1, L2y1, L2x2, L2y2)

    return u, v
end

"""
    ptinqua!(inside, p, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)

For each point column `(p[1,i], p[2,i])`, set `inside[i] = true` if it lies
inside the convex quadrilateral `(p1, p2, p3, p4)` (boundary included),
else `false`.

- `p` is 2×N: first row x, second row y
- `inside` is a preallocated Bool vector of length N

This function does **no heap allocation**.
"""
function ptinqua!(inside::Vector{Bool}, p::AbstractArray{Float64},
    p1x::Float64, p1y::Float64, p2x::Float64, p2y::Float64,
    p3x::Float64, p3y::Float64, p4x::Float64, p4y::Float64)

    N = size(p,2)

    # Precompute edge differences from p1
    dx21 = p2x - p1x; dy21 = p2y - p1y
    dx31 = p3x - p1x; dy31 = p3y - p1y
    dx41 = p4x - p1x; dy41 = p4y - p1y

    # Signed double areas (orientation) for triangles (p1,p2,p3) and (p1,p3,p4)
    s1 = dx21*dy31 - dx31*dy21       # triangle (p1,p2,p3)
    s2 = dx41*dy31 - dx31*dy41       # triangle (p1,p3,p4)

    @inbounds  for i in 1:N
        x = p[1,i]
        y = p[2,i]

        # Triangle (p1,p2,p3)
        det2 = (x - p1x)*dy31 - dx31*(y - p1y)      # β
        det3 = dx21*(y - p1y) - (x - p1x)*dy21      # γ
        det1 = s1 - (det2 + det3)                   # α

        b1 = (s1*det1 >= 0.0) & (s1*det2 >= 0.0) & (s1*det3 >= 0.0)

        # Triangle (p1,p3,p4), reuse (x - p1), (y - p1)
        det3b = dx41*(y - p1y) - (x - p1x)*dy41

        det1b = s2 - (det2 + det3b)

        b2 = (s2*det1b >= 0.0) & (s2*det2 >= 0.0) & (s2*det3b >= 0.0)

        inside[i] = b1 | b2
    end

    return inside
end

function ptinqua(px::Float64, py::Float64,
    p1x::Float64, p1y::Float64, p2x::Float64, p2y::Float64,
    p3x::Float64, p3y::Float64, p4x::Float64, p4y::Float64)::Bool

    # Precompute edge differences from p1
    dx21 = p2x - p1x; dy21 = p2y - p1y
    dx31 = p3x - p1x; dy31 = p3y - p1y
    dx41 = p4x - p1x; dy41 = p4y - p1y

    # Signed double areas (orientation) for triangles (p1,p2,p3) and (p1,p3,p4)
    s1 = dx21*dy31 - dx31*dy21       # triangle (p1,p2,p3)
    s2 = dx41*dy31 - dx31*dy41       # triangle (p1,p3,p4)


    # Triangle (p1,p2,p3)
    det2 = (px - p1x) * dy31 - dx31 * (py - p1y)      # β
    det3 = dx21 * (py - p1y) - (px - p1x) * dy21      # γ
    det1 = s1 - (det2 + det3)                   # α

    b1 = (s1 * det1 >= 0.0) & (s1 * det2 >= 0.0) & (s1 * det3 >= 0.0)

    # Triangle (p1,p3,p4), reuse (px - p1), (py - p1)
    det3b = dx41 * (py - p1y) - (px - p1x) * dy41

    det1b = s2 - (det2 + det3b)

    b2 = (s2 * det1b >= 0.0) & (s2 * det2 >= 0.0) & (s2 * det3b >= 0.0)

    return b1 | b2
end

function ptinqua(p::AbstractArray{Float64},
    p1x::Float64, p1y::Float64, p2x::Float64, p2y::Float64,
    p3x::Float64, p3y::Float64, p4x::Float64, p4y::Float64)

    inside = Vector{Bool}(undef, size(p, 2))

    ptinqua!(inside, p, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)

    return inside
end

"""
#Code to check the above function:
using Random
using Plots
Random.seed!(1234)               # reproducible
N = 1000
x = 4 .* rand(N) .- 2            # uniform in [-2,2]
y = 4 .* rand(N) .- 2

# diamond-shaped convex quad
p1 = [-1.0, 0.0]
p2 = [ 0.0,-1.0]
p3 = [ 1.0, 0.0]
p4 = [ 0.0, 1.0]

# pack points as 2*N (row1=x, row2=y)
P = [x'; y']

mask = ptinqua(P, p1, p2, p3, p4) 

# colors: black outside, blue inside
cols = fill(RGB(0,0,0), N)
cols[mask] .= RGB(0,0,1)

# plot quad boundary + points
plt = plot(; legend=false, xlabel="x", ylabel="y", title="ptinqua test")
plot!(plt,
      [p1[1], p2[1], p3[1], p4[1], p1[1]],
      [p1[2], p2[2], p3[2], p4[2], p1[2]],
      linewidth=2, color=:black)
scatter!(plt, x, y; ms=3.5, markersize=0.1, markeralpha = 0.75, 
                markerstrokewidth=0.0, color=cols)

"""


"""
    lineinter(L1x, L1y, L2x, L2y) -> (v, u)

Given two parametric lines

- Line 1: (x1,y1) + v*(x2-x1, y2-y1)
- Line 2: (x3,y3) + u*(x4-x3, y4-y3)

return the parameters `v` (for Line 1) and `u` (for Line 2) at their intersection.

Inputs (each length-2):
- `L1x = [x1, x2]`, `L1y = [y1, y2]`
- `L2x = [x3, x4]`, `L2y = [y3, y4]`

Notes:
- This computes the intersection of the **infinite lines**. If you need the
  segments to intersect, check that `0 ≤ v ≤ 1` and `0 ≤ u ≤ 1`.
- If the lines are parallel (or nearly so), denominators can be ~0; may
  get `Inf`/`NaN`. 
"""
@inline function lineinter(x1::Float64, y1::Float64, x2::Float64, y2::Float64,
    x3::Float64, y3::Float64, x4::Float64, y4::Float64)
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    dx3 = x3 - x1
    dy3 = y3 - y1

    D   = dx2 * dy1 - dx1 * dy2

    invD= 1.0 / D

    s = (-dx3 * dy2 + dx2 * dy3) * invD   # param on L1
    t = ( dx1 * dy3 - dy1 * dx3) * invD   # param on L2

    return s, t
end

"""
# Code to check the above function
# Line 1: horizontal through y=0 from (-2,0) to (2,0)
L1x = [-2.0, 2.0];  L1y = [0.0, 0.0]
# Line 2: vertical through x=0 from (0,-1) to (0,1)
L2x = [0.0, 0.0];   L2y = [-1.0, 1.0]

v, u = lineinter(L1x, L1y, L2x, L2y)
xi = L1x[1] + v*(L1x[2] - L1x[1])   # = L2x[1] + u*(L2x[2] - L2x[1])
yi = L1y[1] + v*(L1y[2] - L1y[1])   # = L2y[1] + u*(L2y[2] - L2y[1])

println("line_inter: v = \$v, u = \$u, intersection = (\$xi, \$yi)")

# quick sanity checks (should intersect at (0,0))
@assert isapprox(xi, 0.0; atol=1e-16)
@assert isapprox(yi, 0.0; atol=1e-16)
"""


"""
Stable mutating evaluation of:
   w(t) = 2 * v(t)^p / (v(t)^p + v(-t)^p)
 with
   v(t)  = (1+t) * (1/2 - (1/2 - 1/p) * t * (1-t))
   v(-t) = (1-t) * (1/2 + (1/2 - 1/p) * t * (1+t))

 - Computes v and vm together in a cancellation-free way.
 - Avoids 0/0 by never forming vp + vmp.
 - Only one ratio r = (min/max)^p ∈ [0,1].
 - Branchless selection via `ifelse`.
 - out .= β * wfunc(w, α .* t + γ)
"""

function wfunc!(out::AbstractArray, p::Int, t::AbstractArray; 
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 

    c = 0.5 - 1.0 / p              # c = 1/2 - 1/p
    c₂ = 2 * β
    @inbounds  for i in eachindex(t)
        τ = α * t[i] + γ
        tp = 1 + τ                 # 1 + τ
        tm = 1 - τ                 # 1 - τ
        u = c * τ

        # Cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm = tm * muladd(u, tp, 0.5)

        # Compare to decide which is larger (no t-branching)
        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)       # max(v, vm)
        b = ifelse(v_ge, vm, v)       # min(v, vm)

        r = (b / a)^p

        inv1pr = inv(1 + r)        # 1/(1+r)

        # If v≥vm: w = 1/(1+r); else w = r/(1+r)
        out[i] = c₂ * ifelse(v_ge, inv1pr, r * inv1pr)
    end

    return nothing
end

function wfunc(p::Int, t::AbstractArray; 
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 
    Z = similar(t)
    wfunc!(Z, p, t; α=α, β=β, γ=γ)
    return Z
end

"""
Stable mutating evaluation of:
   w^(P/Q)(t) = 2^(P/Q) * v(t)^(p*P/Q) / (v(t)^p + v(-t)^p)^(P/Q)
 with
   v(t)  = (1+t) * (1/2 - (1/2 - 1/p) * t * (1-t))
   v(-t) = (1-t) * (1/2 + (1/2 - 1/p) * t * (1+t))

 - Computes v and vm together in a cancellation-free way.
 - Avoids 0/0 by never forming vp + vmp.
 - Branchless selection via `ifelse`.
 - out .= β * wsfunc(w, α .* t + γ)
"""
function wsfunc!(out::AbstractArray, p::Int, P::Int, Q::Int, t::AbstractArray;
    α::Float64=1.0, β::Float64=1.0, γ::Float64=0.0)

    c = 0.5 - 1.0 / p
    s = P / Q

    num, rem = divrem(p * P, Q)
    k_is_int = iszero(rem)

    c2s = β * 2.0^s

    @inbounds for i in eachindex(t)
        τ = α * t[i] + γ
        tp = 1 + τ
        tm = 1 - τ
        u = c * τ

        # cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm = tm * muladd(u, tp, 0.5)

        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)   # max(v, vm)
        b = ifelse(v_ge, vm, v)   # min(v, vm)

        ρ = b / a
        r = ρ^p
        den = (1 + r)^(-s)

        fac = if k_is_int
            ρ^num
        else
            ρ^(p * s)
        end

        out[i] = c2s * ifelse(v_ge, den, fac * den)
    end

    return nothing
end

function wsfunc(p::Int, P::Int, Q::Int, t::AbstractArray;
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0)

    Z = similar(t)
    wsfunc!(Z, p, P, Q, t; α=α, β=β, γ=γ)
    return Z
end

"""
Stable mutating evaluation of:
   w^(s)(t) = 2^(s) * v(t)^(p*s) / (v(t)^p + v(-t)^p)^(s)
 with
   v(t)  = (1+t) * (1/2 - (1/2 - 1/p) * t * (1-t))
   v(-t) = (1-t) * (1/2 + (1/2 - 1/p) * t * (1+t))

 - Computes v and vm together in a cancellation-free way.
 - Avoids 0/0 by never forming vp + vmp.
 - Branchless selection via `ifelse`.
 - out .= β * wsfunc(w, α .* t + γ)
"""
function wsfunc!(out::AbstractArray, p::Int, s::Float64, t::AbstractArray;
    α::Float64=1.0, β::Float64=1.0, γ::Float64=0.0)

    c = 0.5 - 1.0 / p

    c2s = β * 2.0^s

    @inbounds for i in eachindex(t)
        τ = α * t[i] + γ
        tp = 1 + τ
        tm = 1 - τ
        u = c * τ

        # cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm = tm * muladd(u, tp, 0.5)

        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)   # max(v, vm)
        b = ifelse(v_ge, vm, v)   # min(v, vm)

        ρ = b / a
        r = ρ^p
        den = (1 + r)^(-s)

        fac =  ρ^(p * s)
    
        out[i] = c2s * ifelse(v_ge, den, fac * den)
    end

    return nothing
end

function wsfunc(p::Int, s::Float64, t::AbstractArray;
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0)

    Z = similar(t)
    wsfunc!(Z, p, s, t; α=α, β=β, γ=γ)
    return Z
end


"""
Stable mutating derivative:
   w'(t) = (3(p-2)t^2 + 2) * [ v(t)^(p-1) v(-t)^(p-1) ] / (v(t)^p + v(-t)^p)^2

 Uses the same cancellation-free v/vm computation as wfunc.
 Let a = max(v, vm), b = min(v, vm), ρ = b/a ∈ [0,1], r = ρ^(p-1).
 Then the fraction is r / (a * (1 + ρ*r))^2.
 out .= β * dwfunc(w, α .* t + γ)
"""
function dwfunc!(out::AbstractArray, p::Int, t::AbstractArray; 
    α::Float64=1.0, β::Float64=1.0, γ::Float64 = 0.0)

    c = 0.5 - 1.0 / p              # c = 1/2 - 1/p
    @inbounds  for i in eachindex(t)
        τ = α * t[i] + γ
        tp = 1 + τ                 # 1 + τ
        tm = 1 - τ                 # 1 - τ
        u = c * τ
        # Cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm = tm * muladd(u, tp, 0.5)

        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)       # max(v, vm)
        b = ifelse(v_ge, vm, v)       # min(v, vm)

        ρ = b / a                     # ∈ [0,1]
        r = ρ^(p - 1)

        prefac = 3.0 * (p - 2.0) * (τ * τ) + 2.0  # (3(p-2)τ^2 + 2)

        out[i] = β * prefac * (r / (a * (1.0 + ρ * r))^2)
    end


    return nothing
end

function dwfunc(p::Int, t::AbstractArray; 
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 
    Z = similar(t)
    dwfunc!(Z, p, t; α=α, β=β, γ=γ)
    return Z
end

"""
Stable mutating derivative:
   w''(t) = h(t) * [ v(t)^(p-2) v(-t)^(p-2) ] / (v(t)^p + v(-t)^p)^2

   h(t) = 6(p-2)v(t)v(-t)t + 2p * (3*(1/2-1/p)t^2+1/p)^2 * (2v(t)-1+p*(1-w(t)))

 Uses the same cancellation-free v/vm computation as wfunc.
 Let a = max(v, vm), b = min(v, vm), ρ = b/a ∈ [0,1], r = ρ^(p-2).
 Then the fraction is h(t) * r / (a^2 * (1 + ρ^2 *r) )^2.
 out .= β * ddwfunc(w, α .* t + γ)
"""
function ddwfunc!(out::AbstractArray, p::Int, t::AbstractArray; 
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 

    c = 0.5 - 1.0 / p              # c = 1/2 - 1/p

    @inbounds  for i in eachindex(t)
        τ  = α * t[i] + γ
        tp = 1 + τ                 # 1 + τ
        tm = 1 - τ                 # 1 - τ
        u  = c * τ

        # Cancellation-free v(t), v(-t)
        v  = tp * muladd(-u, tm, 0.5)
        vm = tm * muladd( u, tp, 0.5)

        # Compare to decide which is larger (no t-branching)
        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)       # max(v, vm)
        b = ifelse(v_ge, vm, v)       # min(v, vm)
        ρ = b/a
        r = ρ^(p-2)
        rp = r*ρ*ρ

        inv1pr = inv(1 + rp)

        w = 2.0 * ifelse(v_ge, inv1pr, rp * inv1pr)

        h = β * (6.0 * (p - 2.0) * v * vm * τ +
                 2.0 * p * (3.0 * c * τ^2 + 1.0 / p)^2
                 * (2.0 * v - 1.0 + p * (1.0 - w)))

        out[i] = h * (r / (a^2 * (1.0 + rp))^2)
    end

    return nothing
end

function ddwfunc(p::Int, t::AbstractArray; 
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 
    Z = similar(t)
    ddwfunc!(Z, p, t; α=α, β=β, γ=γ)
    return Z
end

"""
Stable mutating evaluation of:

   dw(t)/w(-t)^(2s-1) = 2^(1-2*s) * (3(p-2)t^2 + 2) *
    [ v(t)^(p-1) v(-t)^(2p(1-s)-1) ] / (v(t)^p + v(-t)^p)^(3-2s)

 with
   v(t)  = (1+t) * (1/2 - (1/2 - 1/p) * t * (1-t))
   v(-t) = (1-t) * (1/2 + (1/2 - 1/p) * t * (1+t))

   For v(t) >= v(-t):

    (v(-t)/v(t))^(2p(1-s)-1) ] / ( v(t)^2 * (1 + (v(-t)/v(t))^p)^(3-2s) ).

   For v(-t) > v(t):

    [ (v(t)/v(-t))^(p-1) ] / ( v(-t)^2 * (1 + (v(t)/v(-t))^p)^(3-2s) ).

 - Computes v and vm together in a cancellation-free way.
 - Avoids 0/0 by never forming vp + vmp.
 - out .= β * qw1func(α .* t + γ)
"""
function qw1func!(out::AbstractArray, p::Int, t::AbstractArray, 
    s::Float64; α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0)

    c = 0.5 - 1.0 / p              # c = 1/2 - 1/p
    ℓ = 2.0 * p * (1.0 - s) - 1.0
    pe = ifelse(abs(ℓ) < 1e-8, 0.0, ℓ)
    c₁ = 2^(1.0 - 2.0*s) * 3.0 * (p - 2.0)
    c₂ = 2^(2.0*(1.0 - s))
    @inbounds  for i in eachindex(t)
        τ = α * t[i] + γ
        tp = 1 + τ                 # 1 + τ
        tm = 1 - τ                 # 1 - τ
        u = c * τ

        # Cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm = tm * muladd(u, tp, 0.5)

        # Compare to decide which is larger (no t-branching)
        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)       # max(v, vm)
        b = ifelse(v_ge, vm, v)       # min(v, vm)

        # 2^(1-2*s) * (3(p-2)τ^2 + 2)
        prefac = muladd(c₁ , τ * τ, c₂) 

        ρ = b / a
        r = ρ^p
        ℓ = ifelse(v_ge, pe, p - 1)

        out[i] = β * prefac * (ρ^ℓ / (a^2 * (1.0 + r)^(3.0 - 2.0 * s)))
    end

    return nothing
end

function qw1func(p::Int, t::AbstractArray, s::Float64;
    α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 
    Z = similar(t)
    qw1func!(Z, p, t, s; α=α, β=β, γ=γ)
    return Z
end


"""
Stable mutating evaluation of:

   dw(t)/w(-t)^(1-s) = 2^(s-1) * (3(p-2)t^2 + 2) *
    [ v(t)^(p-1) v(-t)^(ps-1) ] / (v(t)^p + v(-t)^p)^(1+s)

 with
   v(t)  = (1+t) * (1/2 - (1/2 - 1/p) * t * (1-t))
   v(-t) = (1-t) * (1/2 + (1/2 - 1/p) * t * (1+t))

   For v(t) >= v(-t):

    (v(-t)/v(t))^(ps-1)  / ( v(t)^2 * (1 + (v(-t)/v(t))^p)^(1+s) ).

   For v(-t) > v(t):

    (v(t)/v(-t))^(p-1) / ( v(-t)^2 * (1 + (v(t)/v(-t))^p)^(1+s) ).

 - Computes v and vm together in a cancellation-free way.
 - Avoids 0/0 by never forming vp + vmp.
 - out .= β * qw2func(α .* t + γ)
"""
function qw2func!(out::AbstractArray, p::Int, t::AbstractArray, 
    s::Float64; α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0)

    c = 0.5 - 1.0 / p              # c = 1/2 - 1/p
    ℓ = p * s - 1.0
    pe = ifelse(abs(ℓ) < 1e-8, 0.0, ℓ)
    c₂ = 2^s
    c₁ = c₂ * 3.0 * (p - 2.0)/2.0
    @inbounds  for i in eachindex(t)
        τ = α * t[i] + γ
        tp = 1 + τ                 # 1 + τ
        tm = 1 - τ                 # 1 - τ
        u = c * τ

        # Cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm= tm * muladd( u, tp, 0.5)

        # Compare to decide which is larger (no t-branching)
        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)       # max(v, vm)
        b = ifelse(v_ge, vm, v)       # min(v, vm)

        # 2^(s-1) * (3(p-2)τ^2 + 2)
        prefac = muladd(c₁ , τ * τ, c₂) 

        ρ = b / a
        r = ρ^p
        ℓ = ifelse(v_ge, pe, p - 1)

        out[i] = β * prefac * (ρ^ℓ / (a^2 * (1.0 + r)^(1.0 + s)))
    end

    return nothing
end

function qw2func(p::Int, t::AbstractArray, s::Float64
    ; α::Float64 = 1.0, β::Float64 = 1.0, γ::Float64 = 0.0) 
    Z = similar(t)
    qw2func!(Z, p, t, s; α=α, β=β, γ=γ)
    return Z
end

"""
Stable mutating evaluation of:

   dw(t)/(w(-t)^(2s-1) w(t)^(1-s)) = 2^(-s) * (3(p-2)t^2 + 2) *
    [ v(t)^(ps-1) v(-t)^(2p(1-s)-1) ] / (v(t)^p + v(-t)^p)^(2-s)

 with
   v(t)  = (1+t) * (1/2 - (1/2 - 1/p) * t * (1-t))
   v(-t) = (1-t) * (1/2 + (1/2 - 1/p) * t * (1+t))

   For v(t) >= v(-t):

    (v(-t)/v(t))^(2p(1-s)-1)  / ( v(t)^2 * (1 + (v(-t)/v(t))^p)^(2-s) ).

   For v(-t) > v(t):

    (v(t)/v(-t))^(ps-1)  / ( v(-t)^2 * (1 + (v(t)/v(-t))^p)^(2-s) ).

 - Computes v and vm together in a cancellation-free way.
 - Avoids 0/0 by never forming vp + vmp.
"""
function qw3func!(out::AbstractArray, p::Int, t::AbstractArray, 
    s::Float64; α::Float64=1.0)

    c = 0.5 - 1.0 / p              # c = 1/2 - 1/p
    ℓ = 2.0 * p * (1.0 - s) - 1.0
    pe = ifelse(abs(ℓ) < 1e-8, 0.0, ℓ)
    c₂ = 2^(1.0 - s)
    c₁ = c₂ * 3.0 * (p - 2.0)/2.0
    @inbounds  for i in eachindex(t)
        τ = α * t[i]
        tp = 1 + τ                 # 1 + τ
        tm = 1 - τ                 # 1 - τ
        u = c * τ

        # Cancellation-free v(t), v(-t)
        v = tp * muladd(-u, tm, 0.5)
        vm= tm * muladd( u, tp, 0.5)

        # Compare to decide which is larger (no t-branching)
        v_ge = v ≥ vm
        a = ifelse(v_ge, v, vm)       # max(v, vm)
        b = ifelse(v_ge, vm, v)       # min(v, vm)

        # 2^(-s) * (3(p-2)τ^2 + 2)
        prefac = muladd(c₁ , τ * τ, c₂) 

        ρ = b / a
        r = ρ^p
        ℓ = ifelse(v_ge, pe, p*s - 1)

        out[i] = prefac * (ρ^ℓ / (a^2 * (1.0 + r)^(2.0 - s)))
    end

    return nothing
end

function qw3func(p::Int, t::AbstractArray, s::Float64; α::Float64=1.0) 
    Z = similar(t)
    qw3func!(Z, p, t, s; α=α)
    return Z
end


"""
    wfofunc(p::Int, t::AbstractArray)

Evaluate `w(t) / (t+1)^p` 
"""
function wfofunc(p::Int, t::AbstractArray)
    vfo = ((1 .- 2/p) .* t .* (t .- 1) .+ 1) ./ 2
    v   = (t .+ 1) .* vfo
    vm  = 1 .- v
    return 2 .* (vfo .^ p) ./ (v.^p .+ vm.^p)
end

"""
    dwfofunc(p::Int, t::AbstractArray)

Evaluate `dw(t) / ((1+t)(1-t))^(p-1)` (broadcasted).
"""
function dwfofunc(p::Int, t::AbstractArray)
    vr  = ((1 .- 2/p) .* t .* (t .- 1) .+ 1) ./ 2
    vmr = ((1 .- 2/p) .* t .* (t .+ 1) .+ 1) ./ 2
    v   = (t .+ 1) .* vr
    vm  = (1 .- t) .* vmr
    num = (3*(p-2) .* t.^2 .+ 2) .* (vr .^(p-1) .* vmr .^(p-1))
    den = (v.^p .+ vm.^p).^2
    return num ./ den
end

"""
    winv(p::Int, x::Float64) -> Float64
    winv(p::Int, X::AbstractArray) -> Array{Float64}

Inverse of the w-map `wfunc` on `[-1,1] → [0,2]`.

Given `x ∈ [0,2]`, returns `t ∈ [-1,1]` such that `wfunc(w, t) = x`.

1) Symmetry: the map satisfies `w(-t) = 2 - w(t)`. Hence for `x > 1`, we can
   use `winv(x) = -winv(2 - x)` and only solve on `[0,1]`.
2) Odds transform: for `x ∈ [0,1]`, set
       v = x^(1/p) / (x^(1/p) + (2 - x)^(1/p))  ∈ [0,1].
   With the specific polynomial pre-map used in `wfunc`, `v` is a cubic in `t`,
   leading to a depressed cubic of the form
       t^3 + P*t + Q = 0,
   with `P = 2/(p - 2) > 0` and `Q = (1 - 2v) / (1 - 2/p)`.
3) Closed form: because `P > 0` the discriminant is positive, so there is a
   single Float64 root. We apply Cardano’s formula to get `t` and then clamp to
   `[−1,1]` to protect against tiny roundoff.
4) Special case `p = 2`: there is a simple closed form.
    t = (√x - √(2 - x)) / (√x + √(2 - x))

Notes:
- Inputs are assumed in-range; 0<=x<=2
- The array dispatch applies the scalar inverse element-wise and preserves the
  input shape.
"""
function winv!(out::Vector{Float64}, p::Int, x::Vector{Float64})

    if p == 2
        @inbounds for i in eachindex(x)
            xi = x[i]
            a = sqrt(xi)
            b = sqrt(2.0 - xi)
            out[i] = (a - b) / (a + b)
        end
    else
        invp = 1.0 / p
        P = 2.0 / (p - 2.0)
        inv_den = 1 - 2 / p
        P3 = (P / 3)^3

        @inbounds for i in eachindex(x)
            xi = x[i]

            if xi <= 1.0
                x_eff = xi
                sign = 1.0
            else
                x_eff = 2.0 - xi
                sign = -1.0
            end

            xp = x_eff^invp
            xm = (2.0 - x_eff)^invp
            v = xp / (xm + xp)

            Q = (1 - 2 * v) / inv_den
            halfQ = 0.5 * Q
            Δ = halfQ^2 + P3
            √Δ = sqrt(Δ)

            t = cbrt(-halfQ + √Δ) + cbrt(-halfQ - √Δ)

            out[i] = sign * clamp(t, -1.0, 1.0)
        end
    end

    return nothing
end

function winv(p::Int, x::Float64)::Float64

    # p==2 closed form
    if p == 2
        a, b = sqrt(x), sqrt(2.0 - x)
        return (a - b) / (a + b)
    end
    if x <= 1
        # Step 1: x -> v (odds transform)
        v = x^(1 / p) / ((2.0-x)^(1 / p) + x^(1 / p))

        # Step 2: solve t^3 + P t + Q = 0 with P>0 => Δ>0 => one real root
        P = 2.0 / (p - 2.0)     #  > 0
        Q = (1 - 2*v) / (1 - 2 / p)
        Δ = (Q / 2)^2 + (P / 3)^3 # > 0
        √Δ = sqrt(Δ)
        u = cbrt(-Q / 2 + √Δ)
        w = cbrt(-Q / 2 - √Δ)
        t = u + w
        return clamp(t, -1.0, 1.0)
    else
        return -winv(p, 2.0 - x)
    end
end

function winv(p::Int, x::AbstractArray)

    if p == 2
        a = sqrt.(x)
        b = sqrt.(2.0 .- x)
        return (a .- b) ./ (a .+ b)
    end

    t = similar(x)

    M = x .<= 1

    # --- Branch 1: x ≤ 1 (direct Cardano root) ---
    if any(M)
        x1 = x[M]
        v  = x1 .^(1 / p) ./ ((2 .- x1) .^(1 / p) .+ x1 .^(1 / p))
        P  = 2 / (p - 2)                          # > 0
        Q  = (1 .- 2 .* v) ./ (1 - 2 / p)
        Δ  = (Q ./ 2).^2 .+ (P / 3)^3             # > 0
        sD = sqrt.(Δ)
        u1 = cbrt.(-Q ./ 2 .+ sD)
        v1 = cbrt.(-Q ./ 2 .- sD)
        t1 = clamp.(u1 .+ v1, -1.0, 1.0)          # guard roundoff
        t[M] .= t1
    end

    if any(.!M)
        @views t[.!M] .= -winv(w,2 .- x[.!M])
    end

    return t

end


"""
    nevill(xs::AbstractVector, fs::AbstractVector, x::Float64) -> Float64

Evaluate the interpolating polynomial at a single point `x` using
**Neville's algorithm**, given nodes `xs` and function values `fs`.

- `xs` and `fs` must have the same length and `xs` must contain distinct points.

Returns the interpolated value at `x` and "overwrites fs values". 
"""
function nevill!(xs::Vector{Float64}, fs::Vector{Float64}, x::Float64)
    n = length(xs)

    @inbounds for m in 1:n-1
        for j in n:-1:m+1
            num = (x - xs[j-m]) * fs[j] - (x - xs[j]) * fs[j-1]
            den =  xs[j] - xs[j-m]
            fs[j] = num / den
        end
    end

    return fs[n]
end

function nevill(xs::Vector{Float64}, fs::Vector{Float64}, x::Float64)
    fsw = copy(fs)              # single allocation, fs doesn't mutate 
    return nevill!(xs, fsw, x)
end

"""
    ChebyTN!(N, x)

Chebyshev polynomials of the first kind **T₀ … T_{N-1}** evaluated at `x`.
It is assumed that N ≥ 2, as N=1 case is trivial, and ChebyT! can be 
used in that case.
"""
function ChebyTN!(out::Vector{Float64}, N::Int, x::Float64)

    @inbounds begin
        out[1] = 1.0            # T₀
        if N ≥ 2
            out[2] = x          # T₁
            for n in 3:N
                out[n] = 2 * x * out[n-1] - out[n-2]
            end
        end
    end

    return nothing
end

function ChebyTN!(out::Matrix{Float64}, N::Int, x::Vector{Float64})

    @views out[:, 1] .= 1.0            # T₀
    if N ≥ 2
        @views out[:, 2] .= x          # T₁
        @inbounds for n in 3:N
            col_n = @view out[:, n]
            col_n1 = @view out[:, n-1]
            col_n2 = @view out[:, n-2]

            @inbounds for i in eachindex(x)
                col_n[i] = 2 * x[i] * col_n1[i] - col_n2[i]
            end
        end
    end

    return nothing
end

function ChebyTN(N::Int, x::Vector{Float64})

    Z = Matrix{Float64}(undef, length(x), N)

    ChebyTN!(Z, N, x)

    return Z
end

"""
    ChebyT(n, x)

Evaluate the Chebyshev polynomial of the first kind **Tₙ(x)**.

- `n::Int` — degree (n ≥ 0)
- `x` — a Float64 number 
Returns a value of the same shape as `x`.  
Uses the identities:
- T₀(x) = 1, T₁(x) = x, T₂(x) = 2x²-1, T₃(x) = x(4x²-3)
- T_{2m}(x)   = 2·T_m(x)² - 1
- T_{2m+1}(x) = 2·T_m(x)·T_{m+1}(x) - x
"""
# Scalar version
function ChebyT(n::Int, x::Float64)::Float64
    if n ≤ 3
        n == 3 && return x * (4*x*x - 3)
        n == 2 && return 2*x*x - 1
        n == 1 && return x
        return one(x)  # n == 0
    end
    if iseven(n)
        Tm = ChebyT(n ÷ 2, x)
        return 2*Tm*Tm - 1
    else
        m  = (n - 1) ÷ 2
        Tm = ChebyT(m, x)
        Tp = ChebyT(m + 1, x)
        return 2*Tm*Tp - x
    end
end

"""
 Fast-doubling Chebyshev T_n evaluation (per element)

 If n = sum_k=0^MSB b_k 2^k, then in Horner fashion 

 n= ( (  ( (b_MSB)*2 + b_{MSB-1} )*2 + b_{MSB−2}) ⋯ )

 Idea (binary Horner on the index):
   Write n in binary and process its bits from most-significant to least.
   If we maintain the pair (a, b) = (T_m(x), T_{m+1}(x)), then for each bit:
     1) "Double" the index m → 2m using Chebyshev doubling identities
        T_{2m}   = 2*T_m^2 - 1
        T_{2m+1} = 2*T_m*T_{m+1} - x
        This gives d0 = T_{2m}, d1 = T_{2m+1}.
     2) "Add the bit": if the current bit is 1, advance one more step
        to (T_{2m+1}, T_{2m+2}) using the 3-term recurrence
        T_{k+1} = 2*x*T_k - T_{k-1}.
        Concretely:
          bit = 0 → (a, b) ← (d0, d1)           # becomes (T_{2m},   T_{2m+1})
          bit = 1 → (a, b) ← (d1, 2*x*d1 - d0)  # becomes (T_{2m+1}, T_{2m+2})

 Start state:
   (a, b) = (T_1(x), T_2(x)) = (x, 2x^2-1).

 Bit iteration:
   Let u = UInt(n). Find msb = index of the top 1-bit:
     msb = (sizeof(u)*8 - 1) - leading_zeros(u)
   Iterate k = msb-1 down to 0. At each step test the k-th bit:
     ((u >> k) & 0x1) == 0x1  ⇔ k-th bit is 1 (bits are 0-based from LSB).

 After consuming all bits, a = T_n(x) (and b = T_{n+1}(x), unused).

 Why this works:
   Reading bits from MSB→LSB, the index m is updated as m ← 2*m + bit,
   which reconstructs n by Horner’s rule in base 2. The pair (T_m, T_{m+1})
   lets us realize “×2” via the doubling identities and “+1” via the 
   recurrence. This yields O(log n) work per element.

 Tiny example (n = 13 = 0b1101):
   Start (T1, T2). Bits MSB→LSB: 1,1,0,1
     1 → (T3, T4)
     0 → (T6, T7)
     1 → (T13, T14)
   Result: a = T13(x).
"""
function ChebyT!(out::AbstractArray{<:Float64}, 
    n::Int, X::AbstractArray{<:Float64}) 

    if n == 0
        fill!(out, 1.0)
        return nothing
    elseif n == 1
        copyto!(out, X)
        return nothing
    elseif n == 2
        @inbounds  for i in eachindex(X)
            x = X[i]
            out[i] = muladd(2*x, x, -1.0)    # 2x^2 - 1
        end
        return nothing
    end

    # Convert n::Int to an unsigned Int so bit-operations 
    # like shifts and leading_zeros can be computed.
    u = UInt(n)

    #MSB =  Most significant bit (1st  bit position)
    #LSB = Least significant bit (Last bit position)
    #leading_zeros(u) tells how many zeros precede the first 1
    msb = 63 - leading_zeros(u)

    # Maintain (a, b) = (T_m(x), T_{m+1}(x)); start at m=0 → (1, x)
    # For each bit of n (MSB→LSB): double -> (T_{2m}, T_{2m+1});
    # if bit==1, advance once more -> (T_{2m+1}, T_{2m+2}).
    @inbounds for i in eachindex(X)
        x = X[i]
        a, b = x, 2*x^2 - 1    # T_1(x),  T_2(x)

        # iterate bits of n from MSB down to LSB
        # skip the topmost 1 bit as we are already representing 
        # m=0 → then build up to n
        for k in msb-1:-1:0  # works up to 64-bit Int
            # doubling step: (T_m, T_{m+1}) -> (T_{2m}, T_{2m+1})
            d0 = muladd(2.0*a, a, -1.0)           # 2*a^2 - 1
            d1 = muladd(2.0*a, b, -x)             # 2*a*b - x
            #The if condition below checks if k-th bit is 1
            #For instance, if n=13, then msb = 3 and
            #(u >> 0) & 0x1 == 0x1  # true  (bit0 = 1)
            #(u >> 1) & 0x1 == 0x1  # false (bit1 = 0)  
            #(u >> 2) & 0x1 == 0x1  # true  (bit2 = 1)
            #(u >> 3) & 0x1 == 0x1  # true  (bit3 = 1)
            if (u >> k) & 0x1 == 0x1          
                # bit = 1: (a,b) = (T_{2m+1}, T_{2m+2})
                a = d1
                b = muladd(2.0*x, d1, -d0)           # 2*x*d1 - d0
            else
                # bit = 0: (a,b) = (T_{2m}, T_{2m+1})
                a = d0
                b = d1
            end
        end

        out[i] = a  # a = T_n(x)
    end

    return nothing
end

# Array version 
function ChebyT(n::Int, X::AbstractVector{<:Float64})
    Z = similar(X)
    ChebyT!(Z, n, X)
    return Z
end


"""
This function computes distance of a point p
from the line segment [p1,p2]. Equation of 
line = p1+t(p2-p1);

"""
function dlineseg2D(px::Float64, py::Float64, 
    p1x::Float64, p1y::Float64, 
    p2x::Float64, p2y::Float64)
    # e = p2 - p1
    ex = p2x - p1x
    ey = p2y - p1y

    # d = p - p1
    dx = px - p1x
    dy = py - p1y

    e2 = ex*ex + ey*ey       # dot(e, e)

    # Degenerate segment (p1 == p2)
    if e2 == 0.0
        return sqrt(dx*dx + dy*dy)
    end

    # Projection parameter t ∈ (-∞, +∞)
    t = (dx*ex + dy*ey) / e2

    if t <= 0.0
        # Closest point is p1
        return sqrt(dx*dx + dy*dy)

    elseif t >= 1.0
        # Closest point is p2
        rx = px - p2x
        ry = py - p2y
        return sqrt(rx*rx + ry*ry)

    else
        # Closest point is projection p1 + t*(p2 - p1)
        qx = p1x + t*ex
        qy = p1y + t*ey
        rx = px - qx
        ry = py - qy
        return sqrt(rx*rx + ry*ry)
    end
end

"""
    diffpowers(x::Float64, y::AbstractVector, p::Int)

Compute `(x + 1)^p - (x - y + 1)^p` in a numerically stable way.

- `x` : Float64 scalar (typically in [-1, 0])
- `y` : Float64 vector (each entry typically in (-1, 1)); fully vectorized
- `p` : Int ≥ 1  

Returns a vector the same length as `y`.

Method:
Let `A = x + 1`. Then `(x+1)^p - (x-y+1)^p = A^p - (A - y)^p = y * Q(y)`,
where `Q(y)` is evaluated via a Horner-like scheme with a stable
binomial-coefficient recurrence (no large `nchoosek` intermediates).
"""
function diffpowers(x::Float64, y::AbstractVector, p::Int)

    # common floating type for stable arithmetic
    T = promote_type(typeof(x), eltype(y))
    A  = T(x) + one(T)
    y = T.(y)

    # Trivial case: A == 0 (i.e., x == -1) → (0)^p - (-y)^p
    if A == 0
        return -((-y) .^ p)
    end

    # Horner init at j = p-1:
    # c_{p-1} = (-1)^(p-1) * C(p, p) * A^0 = (-1)^(p-1)
    sign = isodd(p - 1) ? -one(T) : one(T)   # tracks (-1)^j
    binom = one(T)                            # C(p, p)
    Apow  = one(T)                            # A^0
    Q     = sign * binom * Apow               # starts scalar

    # Descend j = p-2, ..., 0 with stable binomial recurrence:
    # C(p, j+1) from C(p, j+2):  C(p, j+1) = C(p, j+2) * (j+2)/(p-j-1)
    for j in (p-2):-1:0
        binom *= (j + 2) / (p - j - 1)       # now C(p, j+1)
        sign  = -sign
        Apow *= A                             # A^(p-j-1)
        coef  = sign * binom * Apow           # c_j
        Q     = coef .+ (y .* Q)             # Horner step
    end

    return y .* Q
end
"""
#Code to test the above function:
#compare with direct expression (should match, but is stable near y≈0)
x = -0.8; y = range(-0.9, 0.9; length=7) |> collect
stable = diffpowers(x, y, 6)
direct = (x + 1)^p .- (x .- y .+ 1).^p
maximum(abs.(stable .- direct))
"""

function GSS(f::F, a::Float64, b::Float64, 
    tol::Float64)::Tuple{Float64,Float64} where {F}

    τ = (sqrt(5) - 1) / 2       # ≈ 0.6180339887498949
    ρ = 1 - τ                   # ≈ 0.3819660112501051
    h = b - a
    # Initial interior points and function values
    x1 = a + ρ * h       # == b - τ*(b - a)
    x2 = a + τ * h
    f1 = f(x1)
    f2 = f(x2)

    L = τ

    iter = 0
    while (b - a) > tol && iter < 128
        if f1 > f2
            # minimum in [x1, b]
            a  = x1
            x1 = x2;  f1 = f2
            x2 = a + τ * L * h
            f2 = f(x2)
        else
            # minimum in [a, x2]
            b  = x2
            x2 = x1;  f2 = f1
            x1 = a + ρ * L * h
            f1 = f(x1)
        end
       L *= τ 
       iter += 1
    end

    #println("length ",b-a," and iter ",iter)

    # Pick the better interior point 
    if f1 <= f2
        return f1, x1
    else
        return f2, x2
    end
end

function Bis(f::F, a::Float64, b::Float64,
             maxi::Int; tol::Float64 = 1e-14) where {F}
    left  = a
    right = b
    fL    = f(a)
    fR    = f(b)

    for _ in 1:maxi
        mid = (left + right) / 2
        fM  = f(mid)

        if abs(fM) < tol
            return mid
        end

        if fL * fM < 0
            right = mid
            fR    = fM
        else
            left = mid
            fL   = fM
        end
    end

    return (left + right) / 2
end

"""
    newtonR1D(f, df, x0, maxi; tol=1e-15)

Newton's method for a Float64 1D root of `f` with derivative `df`, starting at `x0`.
Returns the approximate root.Throws an error if a zero/NaN derivative is 
encountered or the iterate becomes non-finite.
"""
function newtonR1D(f::F, df::DF, x0::Float64, maxi::Int; tol::Float64=1e-15) where {F,DF}
    
    x = x0;

    for _ in 1:maxi
        fx  = f(x)
        abs(fx) < tol && return x

        dfx = df(x)
        if !isfinite(dfx) || abs(dfx) <= 1e-16
            error("Derivative is zero or non-finite at x = $x.")
        end

        x -= fx / dfx
        isfinite(x) || error("Iterate became Inf/NaN.")
    end

    return x
end


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