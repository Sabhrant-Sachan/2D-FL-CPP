#include "subroutines.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
 
namespace subroutines
{
   namespace
   {
      constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

      double ellipse_angle(
          double CC, double Lt,
          double Cx, double Cy, double R1, double R2,
          double e1x, double e1y, double e2x, double e2y,
          double ux, double uy, double vx, double vy, double tol)
      {
         if (std::abs(CC + 1.0) < tol)
            CC = -1.0;
         else if (std::abs(CC - 1.0) < tol)
            CC = 1.0;

         CC = std::clamp(CC, -1.0, 1.0);
         double Et = std::acos(CC);

         const double Xx = ux + Lt * vx, Xy = uy + Lt * vy;
         const double c = std::cos(Et), s = std::sin(Et);

         const double Gpx = Cx + R1 * c * e1x + R2 * s * e2x;
         const double Gpy = Cy + R1 * c * e1y + R2 * s * e2y;
         const double dxp = Gpx - Xx, dyp = Gpy - Xy;
         const double d2p = dxp * dxp + dyp * dyp;

         const double Gmx = Cx + R1 * c * e1x - R2 * s * e2x;
         const double Gmy = Cy + R1 * c * e1y - R2 * s * e2y;
         const double dxm = Gmx - Xx, dym = Gmy - Xy;
         const double d2m = dxm * dxm + dym * dym;

         if (d2p > d2m)
            Et = 2.0 * std::numbers::pi - Et;
         return Et;
      }
   }

   std::tuple<double, double> ellipserinter(
       double Cx, double Cy, double R1, double R2,
       double e1x, double e1y, double e2x, double e2y,
       double ux, double uy, double vx, double vy, double tol)
   {
      const double invR1 = 1.0 / R1, invR2 = 1.0 / R2;
      const double dux = ux - Cx, duy = uy - Cy;

      const double a1 = (dux * e1x + duy * e1y) * invR1;
      const double a2 = (dux * e2x + duy * e2y) * invR2;
      const double b1 = (vx * e1x + vy * e1y) * invR1;
      const double b2 = (vx * e2x + vy * e2y) * invR2;

      const double A = b1 * b1 + b2 * b2;
      const double B = 2.0 * (a1 * b1 + a2 * b2);
      const double Cq = a1 * a1 + a2 * a2 - 1.0;

      if (A == 0.0)
         return {NaN, NaN};

      double D = B * B - 4.0 * A * Cq;
      if (D < -tol)
         return {NaN, NaN};
      if (D < 0.0)
         D = 0.0;

      const double sD = std::sqrt(D);
      const double q = -0.5 * (B + (B >= 0.0 ? sD : -sD));
      if (q == 0.0)
         return {NaN, NaN};

      const double Lt1 = q / A, Lt2 = Cq / q;
      const bool ok1 = std::isfinite(Lt1) && Lt1 >= 0.0;
      const bool ok2 = std::isfinite(Lt2) && Lt2 >= 0.0;

      if (!ok1 && !ok2)
         return {NaN, NaN};

      const double Lt = ok1 ? (ok2 ? std::min(Lt1, Lt2) : Lt1) : Lt2;
      const double Et = ellipse_angle(
          a1 + Lt * b1, Lt, Cx, Cy, R1, R2,
          e1x, e1y, e2x, e2y, ux, uy, vx, vy, tol);

      return {Et, Lt};
   }

   std::tuple<double, double, double, double> ellipselinter(
       double Cx, double Cy, double R1, double R2,
       double e1x, double e1y, double e2x, double e2y,
       double ux, double uy, double vx, double vy, double tol)
   {
      const double invR1 = 1.0 / R1, invR2 = 1.0 / R2;
      const double dux = ux - Cx, duy = uy - Cy;

      const double a1 = (dux * e1x + duy * e1y) * invR1;
      const double a2 = (dux * e2x + duy * e2y) * invR2;
      const double b1 = (vx * e1x + vy * e1y) * invR1;
      const double b2 = (vx * e2x + vy * e2y) * invR2;

      const double A = b1 * b1 + b2 * b2;
      const double B = 2.0 * (a1 * b1 + a2 * b2);
      const double Cq = a1 * a1 + a2 * a2 - 1.0;

      if (A == 0.0)
         return {NaN, NaN, NaN, NaN};

      double D = B * B - 4.0 * A * Cq;
      if (D < -tol)
         return {NaN, NaN, NaN, NaN};
      if (D < 0.0)
         D = 0.0;

      const double sD = std::sqrt(D);
      const double q = -0.5 * (B + (B >= 0.0 ? sD : -sD));
      if (q == 0.0)
         return {NaN, NaN, NaN, NaN};

      const double Lt1 = q / A, Lt2 = Cq / q;
      if (!std::isfinite(Lt1) || !std::isfinite(Lt2))
         return {NaN, NaN, NaN, NaN};

      const double Et1 = ellipse_angle(
          a1 + Lt1 * b1, Lt1, Cx, Cy, R1, R2,
          e1x, e1y, e2x, e2y, ux, uy, vx, vy, tol);

      const double Et2 = ellipse_angle(
          a1 + Lt2 * b1, Lt2, Cx, Cy, R1, R2,
          e1x, e1y, e2x, e2y, ux, uy, vx, vy, tol);

      return {Et1, Lt1, Et2, Lt2};
   }

} // namespace subroutines