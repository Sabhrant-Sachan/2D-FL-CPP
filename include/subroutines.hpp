#pragma once

#include <tuple>

namespace subroutines
{

std::tuple<double, double>
ellipserinter(
    double Cx, double Cy,
    double R1, double R2,
    double e1x, double e1y,
    double e2x, double e2y,
    double ux, double uy,
    double vx, double vy,
    double tol = 1.0e-14
);

std::tuple<double, double, double, double>
ellipselinter(
    double Cx, double Cy,
    double R1, double R2,
    double e1x, double e1y,
    double e2x, double e2y,
    double ux, double uy,
    double vx, double vy,
    double tol = 1.0e-14
);

}