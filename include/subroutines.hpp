#pragma once

#include <Eigen/Dense>
#include <array>
#include <tuple>
#include <vector>
#include <numeric>

#include <cmath>
#include <stdexcept>
#include <string>

namespace subroutines
{

   // Finds the first forward intersection between a ray and an ellipse.
   //
   // The ellipse is centered at (Cx, Cy), has semi-axis lengths R1 and R2,
   // and axis directions e1 and e2. The ray is
   //     (ux, uy) + Lt * (vx, vy),  Lt >= 0.
   //
   // Returns:
   //     (Et, Lt)
   // where Et is the ellipse angle and Lt is the ray parameter.
   // Returns NaN values when no forward intersection exists.
   std::tuple<double, double>
   ellipserinter(
       double Cx, double Cy,
       double R1, double R2,
       double e1x, double e1y,
       double e2x, double e2y,
       double ux, double uy,
       double vx, double vy,
       double tol = 1.0e-14);

   // Finds both intersections between an infinite line and an ellipse.
   //
   // The ellipse is centered at (Cx, Cy), has semi-axis lengths R1 and R2,
   // and axis directions e1 and e2. The line is
   //     (ux, uy) + Lt * (vx, vy).
   //
   // Returns:
   //     (Et1, Lt1, Et2, Lt2)
   // containing the ellipse angle and line parameter for each intersection.
   // Returns NaN values when no intersection exists.
   std::tuple<double, double, double, double>
   ellipselinter(
       double Cx, double Cy,
       double R1, double R2,
       double e1x, double e1y,
       double e2x, double e2y,
       double ux, double uy,
       double vx, double vy,
       double tol = 1.0e-14);

   // Intersects a tangent line with two other infinite lines.
   //
   // The tangent line passes through (Tpx, Tpy) and has direction (Tmx, Tmy).
   // L1 and L3 are each defined by two endpoints.
   //
   // Returns:
   //     (P1x, P1y, P2x, P2y)
   // where P1 is the intersection with L1 and P2 is the intersection with L3.
   std::tuple<double, double, double, double> tanginterx(
       double L1x1, double L1y1, double L1x2, double L1y2,
       double Tpx, double Tpy, double Tmx, double Tmy,
       double L3x1, double L3y1, double L3x2, double L3y2);

   // Expands a convex quadrilateral outward by distance delta.
   //
   // PQ stores the four input vertices as
   //     {p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y}.
   //
   // Pout receives the four vertices of the expanded quadrilateral
   // in the same format.
   void extendqua(
       std::array<double, 8> &Pout,
       const std::array<double, 8> &PQ,
       double delta);

   // Finds the intersection parameters of two infinite parametric lines.
   //
   // Line 1:
   //     (x1, y1) + s * ((x2, y2) - (x1, y1))
   //
   // Line 2:
   //     (x3, y3) + t * ((x4, y4) - (x3, y3))
   //
   // Returns:
   //     (s, t)
   //
   // Parallel lines may produce Inf or NaN values.
   std::tuple<double, double> lineinter(
       double x1, double y1, double x2, double y2,
       double x3, double y3, double x4, double y4);

   // Finds only the line parameters produced by tanginterx.
   //
   // The tangent line passes through (Tpx, Tpy) and has direction (Tmx, Tmy).
   //
   // Returns:
   //     (u, v)
   // where v is the parameter on L1 and u is the parameter on L3.
   std::tuple<double, double> tanginterp(
       double L1x1, double L1y1, double L1x2, double L1y2,
       double Tpx, double Tpy, double Tmx, double Tmy,
       double L3x1, double L3y1, double L3x2, double L3y2);

   // Tests whether one point lies inside a quadrilateral by splitting it
   // into triangles (p1,p2,p3) and (p1,p3,p4).
   //
   // The vertices must be supplied in boundary order, and the diagonal
   // from p1 to p3 must form a valid internal triangulation.
   // Points on the boundary are considered inside.
   bool ptinqua(
       double px, double py,
       double p1x, double p1y, double p2x, double p2y,
       double p3x, double p3y, double p4x, double p4y);

   // Tests multiple points against a quadrilateral.
   //
   // Each entry in points is stored as {x, y}. Results are written into
   // the preallocated inside vector. inside and points must have equal sizes.
   // Boundary points are considered inside.
   void ptinqua(
       std::vector<bool> &inside,
       const std::vector<std::array<double, 2>> &points,
       double p1x, double p1y, double p2x, double p2y,
       double p3x, double p3y, double p4x, double p4y);

   // Tests multiple points against a quadrilateral.
   //
   // Each entry in points is stored as {x, y}. Returns a newly allocated
   // Boolean vector containing one result per point.
   // Boundary points are considered inside.
   std::vector<bool> ptinqua(
       const std::vector<std::array<double, 2>> &points,
       double p1x, double p1y, double p2x, double p2y,
       double p3x, double p3y, double p4x, double p4y);

   // Stable vector evaluation of w(alpha*t + gamma), scaled by beta.
   // The mutating overload requires out.size() == t.size().
   void wfunc(std::vector<double> &out, int p, const std::vector<double> &t,
              double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

   // Allocating overload.
   std::vector<double> wfunc(int p, const std::vector<double> &t,
                             double alpha = 1.0, double beta = 1.0,
                             double gamma = 0.0);

   // Stable vector evaluation of w(alpha*t + gamma)^(P/Q), scaled by beta.
   // The mutating overload requires out.size() == t.size().
   void wsfunc(std::vector<double> &out, int p, int P, int Q,
               const std::vector<double> &t,
               double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

   // Allocating rational-power overload.
   std::vector<double> wsfunc(int p, int P, int Q,
                              const std::vector<double> &t,
                              double alpha = 1.0, double beta = 1.0,
                              double gamma = 0.0);

   // Stable vector evaluation of w(alpha*t + gamma)^s, scaled by beta.
   // The mutating overload requires out.size() == t.size().
   void wsfunc(std::vector<double> &out, int p, double s,
               const std::vector<double> &t,
               double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

   // Allocating real-power overload.
   std::vector<double> wsfunc(int p, double s,
                              const std::vector<double> &t,
                              double alpha = 1.0, double beta = 1.0,
                              double gamma = 0.0);

   // Stable vector evaluation of w'(alpha*t + gamma), scaled by beta.
   // Requires p >= 2 and out.size() == t.size().
   void dwfunc(std::vector<double> &out, int p, const std::vector<double> &t,
               double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

   // Allocating overload.
   std::vector<double> dwfunc(int p, const std::vector<double> &t,
                              double alpha = 1.0, double beta = 1.0,
                              double gamma = 0.0);

   // Stable vector evaluation of w''(alpha*t + gamma), scaled by beta.
   // Requires p >= 2 and out.size() == t.size().
   void ddwfunc(std::vector<double> &out, int p, const std::vector<double> &t,
                double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

   // Allocating overload.
   std::vector<double> ddwfunc(int p, const std::vector<double> &t,
                               double alpha = 1.0, double beta = 1.0,
                               double gamma = 0.0);

   // Stable evaluation of dw(t)/w(-t)^(2s-1).
   void qw1func(std::vector<double> &out, int p, const std::vector<double> &t,
                double s, double alpha = 1.0, double beta = 1.0,
                double gamma = 0.0);

   std::vector<double> qw1func(int p, const std::vector<double> &t,
                               double s, double alpha = 1.0,
                               double beta = 1.0, double gamma = 0.0);

   // Stable evaluation of dw(t)/w(-t)^(1-s).
   void qw2func(std::vector<double> &out, int p, const std::vector<double> &t,
                double s, double alpha = 1.0, double beta = 1.0,
                double gamma = 0.0);

   std::vector<double> qw2func(int p, const std::vector<double> &t,
                               double s, double alpha = 1.0,
                               double beta = 1.0, double gamma = 0.0);

   // Stable evaluation of dw(t)/(w(-t)^(2s-1)w(t)^(1-s)).
   void qw3func(std::vector<double> &out, int p, const std::vector<double> &t,
                double s, double alpha = 1.0);

   std::vector<double> qw3func(int p, const std::vector<double> &t,
                               double s, double alpha = 1.0);

   // Inverse of wfunc on [0,2] -> [-1,1].
   double winv(int p, double x);

   // Mutating vector overload. Requires out.size() == x.size().
   void winv(std::vector<double> &out, int p, const std::vector<double> &x);

   // Allocating vector overload.
   std::vector<double> winv(int p, const std::vector<double> &x);

   // Neville interpolation. The mutating overload overwrites fs.
   double nevill(const std::vector<double> &xs, std::vector<double> &fs, double x);

   // Non-mutating overload.
   double nevill(const std::vector<double> &xs, const std::vector<double> &fs, double x);

   // Chebyshev polynomials T0,...,T(N-1) at one point x.
   // Requires out.size() >= N and N >= 1.
   void ChebyTN(std::vector<double> &out, int N, double x);

   // Chebyshev polynomials T0,...,T(N-1) for multiple x values.
   void ChebyTN(Eigen::MatrixXd &out, int N, const std::vector<double> &x);

   // Allocating multiple-point overload.
   // Rows correspond to x values and columns to T0,...,T(N-1).
   Eigen::MatrixXd ChebyTN(int N, const std::vector<double> &x);

   // Scalar Chebyshev polynomial T_n(x).
   double ChebyT(int n, double x);

   // Mutating vector evaluation. Requires out.size() == X.size().
   void ChebyT(std::vector<double> &out, int n, const std::vector<double> &X);

   // Allocating vector overload.
   std::vector<double> ChebyT(int n, const std::vector<double> &X);

   // Euclidean distance from point p to the line segment [p1,p2].
   double dlineseg2D(double px, double py,
                     double p1x, double p1y, double p2x, double p2y);

   // Golden-section search for the minimum of f on [a,b].
   // Returns {minimum function value, minimizer}.
   template <class F>
   std::tuple<double, double> GSS(
       F &&f, double a, double b, double tol)
   {
      constexpr double tau = 0.6180339887498948482;
      constexpr double rho = 1.0 - tau;

      const double h = b - a;

      double x1 = a + rho * h;
      double x2 = a + tau * h;
      double f1 = f(x1);
      double f2 = f(x2);

      double L = tau;
      int iter = 0;

      while ((b - a) > tol && iter < 128)
      {
         if (f1 > f2)
         {
            a = x1;
            x1 = x2;
            f1 = f2;

            x2 = a + tau * L * h;
            f2 = f(x2);
         }
         else
         {
            b = x2;
            x2 = x1;
            f2 = f1;

            x1 = a + rho * L * h;
            f1 = f(x1);
         }

         L *= tau;
         ++iter;
      }

      return f1 <= f2
                 ? std::tuple<double, double>{f1, x1}
                 : std::tuple<double, double>{f2, x2};
   }

   // Bisection method for a root of f on [a,b].
   // The interval should bracket a root.
   template <class F>
   double Bis(
       F &&f, double a, double b, int maxi, double tol = 1.0e-14)
   {
      double left = a;
      double right = b;
      double fL = f(left);
      double fR = f(right);

      if (std::abs(fL) < tol)
         return left;

      if (std::abs(fR) < tol)
         return right;

      for (int iter = 0; iter < maxi; ++iter)
      {
         const double mid = std::midpoint(left, right);
         const double fM = f(mid);

         if (std::abs(fM) < tol)
            return mid;

         if ((fL < 0.0 && fM > 0.0) || (fL > 0.0 && fM < 0.0))
         {
            right = mid;
            fR = fM;
         }
         else
         {
            left = mid;
            fL = fM;
         }
      }

      return std::midpoint(left, right);
   }

   // Newton's method for a scalar root of f.
   // Throws std::runtime_error if the derivative or iterate becomes invalid.
   template <class F, class DF>
   double newtonR1D(
       F &&f, DF &&df,
       double x0, int maxi,
       double tol = 1.0e-15)
   {
      double x = x0;

      for (int iter = 0; iter < maxi; ++iter)
      {
         const double fx = f(x);

         if (std::abs(fx) < tol)
            return x;

         const double dfx = df(x);

         if (!std::isfinite(dfx) || std::abs(dfx) <= 1.0e-16)
         {
            throw std::runtime_error(
                "newtonR1D: derivative is zero or non-finite at x = " +
                std::to_string(x));
         }

         x -= fx / dfx;

         if (!std::isfinite(x))
         {
            throw std::runtime_error(
                "newtonR1D: iterate became Inf or NaN");
         }
      }

      return x;
   }
}