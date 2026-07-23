#include "subroutines.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <array>
#include <tuple>
#include <cassert>
#include <bit>

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

   std::tuple<double, double, double, double> tanginterx(
       double L1x1, double L1y1, double L1x2, double L1y2,
       double Tpx, double Tpy, double Tmx, double Tmy,
       double L3x1, double L3y1, double L3x2, double L3y2)
   {
      const double L2x2 = Tpx + Tmx;
      const double L2y2 = Tpy + Tmy;

      const double v = std::get<0>(lineinter(
          L1x1, L1y1, L1x2, L1y2,
          Tpx, Tpy, L2x2, L2y2));

      const double P1x = L1x1 + v * (L1x2 - L1x1);
      const double P1y = L1y1 + v * (L1y2 - L1y1);

      const double u = std::get<0>(lineinter(
          L3x1, L3y1, L3x2, L3y2,
          Tpx, Tpy, L2x2, L2y2));

      const double P2x = L3x1 + u * (L3x2 - L3x1);
      const double P2y = L3y1 + u * (L3y2 - L3y1);

      return {P1x, P1y, P2x, P2y};
   }

   void extendqua(
       std::array<double, 8> &Pout,
       const std::array<double, 8> &PQ,
       double delta)
   {
      const double p1x = PQ[0], p1y = PQ[1];
      const double p2x = PQ[2], p2y = PQ[3];
      const double p3x = PQ[4], p3y = PQ[5];
      const double p4x = PQ[6], p4y = PQ[7];

      const double e1x = p2x - p1x, e1y = p2y - p1y;
      const double e2x = p3x - p2x, e2y = p3y - p2y;
      const double e3x = p4x - p3x, e3y = p4y - p3y;
      const double e4x = p1x - p4x, e4y = p1y - p4y;

      const double len1 = std::hypot(e1x, e1y);
      const double len2 = std::hypot(e2x, e2y);
      const double len3 = std::hypot(e3x, e3y);
      const double len4 = std::hypot(e4x, e4y);

      double n1x = e1y / len1, n1y = -e1x / len1;
      double n2x = e2y / len2, n2y = -e2x / len2;
      double n3x = e3y / len3, n3y = -e3x / len3;
      double n4x = e4y / len4, n4y = -e4x / len4;

      if (e1x * e2y - e1y * e2x < 0.0)
      {
         n1x = -n1x;
         n1y = -n1y;
         n2x = -n2x;
         n2y = -n2y;
         n3x = -n3x;
         n3y = -n3y;
         n4x = -n4x;
         n4y = -n4y;
      }

      const double L1x1 = p1x + delta * n1x, L1y1 = p1y + delta * n1y;
      const double L1x2 = p2x + delta * n1x, L1y2 = p2y + delta * n1y;

      const double L2x1 = p2x + delta * n2x, L2y1 = p2y + delta * n2y;
      const double L2x2 = p3x + delta * n2x, L2y2 = p3y + delta * n2y;

      const double L3x1 = p3x + delta * n3x, L3y1 = p3y + delta * n3y;
      const double L3x2 = p4x + delta * n3x, L3y2 = p4y + delta * n3y;

      const double L4x1 = p4x + delta * n4x, L4y1 = p4y + delta * n4y;
      const double L4x2 = p1x + delta * n4x, L4y2 = p1y + delta * n4y;

      const auto [P2x, P2y, P3x, P3y] = tanginterx(
          L1x1, L1y1, L1x2, L1y2,
          L2x1, L2y1, L2x2 - L2x1, L2y2 - L2y1,
          L3x1, L3y1, L3x2, L3y2);

      const auto [P1x, P1y, P4x, P4y] = tanginterx(
          L1x1, L1y1, L1x2, L1y2,
          L4x1, L4y1, L4x2 - L4x1, L4y2 - L4y1,
          L3x1, L3y1, L3x2, L3y2);

      Pout = {P1x, P1y, P2x, P2y, P3x, P3y, P4x, P4y};
   }

   std::tuple<double, double> lineinter(
       double x1, double y1, double x2, double y2,
       double x3, double y3, double x4, double y4)
   {
      const double dx1 = x2 - x1, dy1 = y2 - y1;
      const double dx2 = x4 - x3, dy2 = y4 - y3;
      const double dx3 = x3 - x1, dy3 = y3 - y1;

      const double D = dx2 * dy1 - dx1 * dy2;
      const double invD = 1.0 / D;

      const double s = (-dx3 * dy2 + dx2 * dy3) * invD;
      const double t = (dx1 * dy3 - dy1 * dx3) * invD;

      return {s, t};
   }

   std::tuple<double, double> tanginterp(
       double L1x1, double L1y1, double L1x2, double L1y2,
       double Tpx, double Tpy, double Tmx, double Tmy,
       double L3x1, double L3y1, double L3x2, double L3y2)
   {
      const double L2x2 = Tpx + Tmx;
      const double L2y2 = Tpy + Tmy;

      const double v = std::get<0>(lineinter(
          L1x1, L1y1, L1x2, L1y2,
          Tpx, Tpy, L2x2, L2y2));

      const double u = std::get<0>(lineinter(
          L3x1, L3y1, L3x2, L3y2,
          Tpx, Tpy, L2x2, L2y2));

      return {u, v};
   }

   bool ptinqua(
       double px, double py,
       double p1x, double p1y, double p2x, double p2y,
       double p3x, double p3y, double p4x, double p4y)
   {
      const double dx21 = p2x - p1x, dy21 = p2y - p1y;
      const double dx31 = p3x - p1x, dy31 = p3y - p1y;
      const double dx41 = p4x - p1x, dy41 = p4y - p1y;

      const double s1 = dx21 * dy31 - dx31 * dy21;
      const double s2 = dx41 * dy31 - dx31 * dy41;

      const double dx = px - p1x, dy = py - p1y;

      const double det2 = dx * dy31 - dx31 * dy;
      const double det3 = dx21 * dy - dx * dy21;
      const double det1 = s1 - det2 - det3;

      const bool b1 =
          s1 * det1 >= 0.0 &&
          s1 * det2 >= 0.0 &&
          s1 * det3 >= 0.0;

      const double det3b = dx41 * dy - dx * dy41;
      const double det1b = s2 - det2 - det3b;

      const bool b2 =
          s2 * det1b >= 0.0 &&
          s2 * det2 >= 0.0 &&
          s2 * det3b >= 0.0;

      return b1 || b2;
   }

   void ptinqua(
       std::vector<bool> &inside,
       const std::vector<std::array<double, 2>> &points,
       double p1x, double p1y, double p2x, double p2y,
       double p3x, double p3y, double p4x, double p4y)
   {
      assert(inside.size() == points.size());

      const double dx21 = p2x - p1x, dy21 = p2y - p1y;
      const double dx31 = p3x - p1x, dy31 = p3y - p1y;
      const double dx41 = p4x - p1x, dy41 = p4y - p1y;

      const double s1 = dx21 * dy31 - dx31 * dy21;
      const double s2 = dx41 * dy31 - dx31 * dy41;

      for (std::size_t i = 0; i < points.size(); ++i)
      {
         const double dx = points[i][0] - p1x;
         const double dy = points[i][1] - p1y;

         const double det2 = dx * dy31 - dx31 * dy;
         const double det3 = dx21 * dy - dx * dy21;
         const double det1 = s1 - det2 - det3;

         const bool b1 =
             s1 * det1 >= 0.0 &&
             s1 * det2 >= 0.0 &&
             s1 * det3 >= 0.0;

         const double det3b = dx41 * dy - dx * dy41;
         const double det1b = s2 - det2 - det3b;

         const bool b2 =
             s2 * det1b >= 0.0 &&
             s2 * det2 >= 0.0 &&
             s2 * det3b >= 0.0;

         inside[i] = b1 || b2;
      }
   }

   std::vector<bool> ptinqua(
       const std::vector<std::array<double, 2>> &points,
       double p1x, double p1y, double p2x, double p2y,
       double p3x, double p3y, double p4x, double p4y)
   {
      std::vector<bool> inside(points.size());

      ptinqua(
          inside, points,
          p1x, p1y, p2x, p2y,
          p3x, p3y, p4x, p4y);

      return inside;
   }

   void wfunc(
       std::vector<double> &out,
       int p, const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p != 0);

      const double c = 0.5 - 1.0 / static_cast<double>(p);
      const double c2 = 2.0 * beta;

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         // Equivalent to Julia's muladd operations.
         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double r = std::pow(b / a, p);
         const double inv1pr = 1.0 / (1.0 + r);

         out[i] = c2 * (v_ge ? inv1pr : r * inv1pr);
      }
   }

   std::vector<double> wfunc(
       int p, const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      wfunc(out, p, t, alpha, beta, gamma);
      return out;
   }

   void wsfunc(
       std::vector<double> &out,
       int p, int P, int Q,
       const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p != 0);
      assert(Q != 0);

      const double c = 0.5 - 1.0 / static_cast<double>(p);
      const double s =
          static_cast<double>(P) / static_cast<double>(Q);

      const int product = p * P;
      const int num = product / Q;
      const int rem = product % Q;
      const bool exponent_is_integer = rem == 0;

      const double c2s = beta * std::pow(2.0, s);

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double rho = b / a;
         const double r = std::pow(rho, p);
         const double den = std::pow(1.0 + r, -s);

         const double fac = exponent_is_integer
                                ? std::pow(rho, num)
                                : std::pow(rho, static_cast<double>(p) * s);

         out[i] = c2s * (v_ge ? den : fac * den);
      }
   }

   std::vector<double> wsfunc(
       int p, int P, int Q,
       const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      wsfunc(out, p, P, Q, t, alpha, beta, gamma);
      return out;
   }

   void wsfunc(
       std::vector<double> &out,
       int p, double s,
       const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p != 0);

      const double c = 0.5 - 1.0 / static_cast<double>(p);
      const double c2s = beta * std::pow(2.0, s);

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double rho = b / a;
         const double r = std::pow(rho, p);
         const double den = std::pow(1.0 + r, -s);
         const double fac = std::pow(rho, static_cast<double>(p) * s);

         out[i] = c2s * (v_ge ? den : fac * den);
      }
   }

   std::vector<double> wsfunc(
       int p, double s,
       const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      wsfunc(out, p, s, t, alpha, beta, gamma);
      return out;
   }

   void dwfunc(
       std::vector<double> &out,
       int p, const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p >= 2);

      const double c = 0.5 - 1.0 / static_cast<double>(p);

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double rho = b / a;
         const double r = std::pow(rho, p - 1);

         const double prefactor =
             3.0 * (static_cast<double>(p) - 2.0) * tau * tau + 2.0;

         const double denominator = a * (1.0 + rho * r);

         out[i] = beta * prefactor * r / (denominator * denominator);
      }
   }

   std::vector<double> dwfunc(
       int p, const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      dwfunc(out, p, t, alpha, beta, gamma);
      return out;
   }

   void ddwfunc(
       std::vector<double> &out, int p,
       const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p >= 2);

      const double pd = static_cast<double>(p);
      const double c = 0.5 - 1.0 / pd;

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double rho = b / a;
         const double r = std::pow(rho, p - 2);
         const double rp = r * rho * rho;
         const double inv1pr = 1.0 / (1.0 + rp);

         const double w =
             2.0 * (v_ge ? inv1pr : rp * inv1pr);

         const double q = 3.0 * c * tau * tau + 1.0 / pd;

         const double h =
             beta * (6.0 * (pd - 2.0) * v * vm * tau +
                     2.0 * pd * q * q * (2.0 * v - 1.0 + pd * (1.0 - w)));

         const double denominator = a * a * (1.0 + rp);

         out[i] = h * r / (denominator * denominator);
      }
   }

   std::vector<double> ddwfunc(
       int p, const std::vector<double> &t,
       double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      ddwfunc(out, p, t, alpha, beta, gamma);
      return out;
   }

   void qw1func(
       std::vector<double> &out,
       int p, const std::vector<double> &t,
       double s, double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p != 0);

      const double pd = static_cast<double>(p);
      const double c = 0.5 - 1.0 / pd;

      const double exponent = 2.0 * pd * (1.0 - s) - 1.0;
      const double pe = std::abs(exponent) < 1.0e-8 ? 0.0 : exponent;

      const double c1 =
          std::pow(2.0, 1.0 - 2.0 * s) * 3.0 * (pd - 2.0);

      const double c2 =
          std::pow(2.0, 2.0 * (1.0 - s));

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double prefactor =
             std::fma(c1, tau * tau, c2);

         const double rho = b / a;
         const double r = std::pow(rho, p);
         const double ell = v_ge ? pe : pd - 1.0;

         out[i] = beta * prefactor * std::pow(rho, ell) /
             (a * a * std::pow(1.0 + r, 3.0 - 2.0 * s));
      }
   }

   std::vector<double> qw1func(
       int p, const std::vector<double> &t,
       double s, double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      qw1func(out, p, t, s, alpha, beta, gamma);
      return out;
   }

   void qw2func(
       std::vector<double> &out, int p, 
       const std::vector<double> &t,
       double s, double alpha, double beta, double gamma)
   {
      assert(out.size() == t.size());
      assert(p != 0);

      const double pd = static_cast<double>(p);
      const double c = 0.5 - 1.0 / pd;

      const double exponent = pd * s - 1.0;
      const double pe = std::abs(exponent) < 1.0e-8 ? 0.0 : exponent;

      const double c2 = std::pow(2.0, s);
      const double c1 = c2 * 3.0 * (pd - 2.0) / 2.0;

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i] + gamma;
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double prefactor =
             std::fma(c1, tau * tau, c2);

         const double rho = b / a;
         const double r = std::pow(rho, p);
         const double ell = v_ge ? pe : pd - 1.0;

         out[i] = beta * prefactor *
                  std::pow(rho, ell) /
                  (a * a * std::pow(1.0 + r, 1.0 + s));
      }
   }

   std::vector<double> qw2func(
       int p, const std::vector<double> &t,
       double s, double alpha, double beta, double gamma)
   {
      std::vector<double> out(t.size());
      qw2func(out, p, t, s, alpha, beta, gamma);
      return out;
   }

   void qw3func(
       std::vector<double> &out,
       int p, const std::vector<double> &t,
       double s, double alpha)
   {
      assert(out.size() == t.size());
      assert(p != 0);

      const double pd = static_cast<double>(p);
      const double c = 0.5 - 1.0 / pd;

      const double exponent = 2.0 * pd * (1.0 - s) - 1.0;
      const double pe = std::abs(exponent) < 1.0e-8 ? 0.0 : exponent;

      const double c2 = std::pow(2.0, 1.0 - s);
      const double c1 = c2 * 3.0 * (pd - 2.0) / 2.0;

      for (std::size_t i = 0; i < t.size(); ++i)
      {
         const double tau = alpha * t[i];
         const double tp = 1.0 + tau;
         const double tm = 1.0 - tau;
         const double u = c * tau;

         const double v = tp * std::fma(-u, tm, 0.5);
         const double vm = tm * std::fma(u, tp, 0.5);

         const bool v_ge = v >= vm;
         const double a = v_ge ? v : vm;
         const double b = v_ge ? vm : v;

         const double prefactor =
             std::fma(c1, tau * tau, c2);

         const double rho = b / a;
         const double r = std::pow(rho, p);

         const double ell =
             v_ge ? pe : pd * s - 1.0;

         out[i] = prefactor * std::pow(rho, ell) /
                  (a * a * std::pow(1.0 + r, 2.0 - s));
      }
   }

   std::vector<double> qw3func(
       int p, const std::vector<double> &t,
       double s, double alpha)
   {
      std::vector<double> out(t.size());
      qw3func(out, p, t, s, alpha);
      return out;
   }

   double winv(int p, double x)
   {
      assert(p >= 2);
      assert(x >= 0.0 && x <= 2.0);

      if (p == 2)
      {
         const double a = std::sqrt(x);
         const double b = std::sqrt(2.0 - x);
         return (a - b) / (a + b);
      }

      if (x > 1.0)
         return -winv(p, 2.0 - x);

      const double pd = static_cast<double>(p);
      const double invp = 1.0 / pd;

      const double xp = std::pow(x, invp);
      const double xm = std::pow(2.0 - x, invp);
      const double v = xp / (xp + xm);

      const double P = 2.0 / (pd - 2.0);
      const double Q = (1.0 - 2.0 * v) / (1.0 - 2.0 / pd);

      const double halfQ = 0.5 * Q;
      const double discriminant =
          halfQ * halfQ + std::pow(P / 3.0, 3);

      const double sqrtD = std::sqrt(discriminant);

      const double t =
          std::cbrt(-halfQ + sqrtD) +
          std::cbrt(-halfQ - sqrtD);

      return std::clamp(t, -1.0, 1.0);
   }

   void winv(
       std::vector<double> &out,
       int p, const std::vector<double> &x)
   {
      assert(out.size() == x.size());
      assert(p >= 2);

      if (p == 2)
      {
         for (std::size_t i = 0; i < x.size(); ++i)
         {
            assert(x[i] >= 0.0 && x[i] <= 2.0);

            const double a = std::sqrt(x[i]);
            const double b = std::sqrt(2.0 - x[i]);

            out[i] = (a - b) / (a + b);
         }

         return;
      }

      const double pd = static_cast<double>(p);
      const double invp = 1.0 / pd;
      const double P = 2.0 / (pd - 2.0);
      const double invDen = 1.0 - 2.0 / pd;
      const double P3 = std::pow(P / 3.0, 3);

      for (std::size_t i = 0; i < x.size(); ++i)
      {
         assert(x[i] >= 0.0 && x[i] <= 2.0);

         const bool lowerHalf = x[i] <= 1.0;
         const double xEff = lowerHalf ? x[i] : 2.0 - x[i];
         const double sign = lowerHalf ? 1.0 : -1.0;

         const double xp = std::pow(xEff, invp);
         const double xm = std::pow(2.0 - xEff, invp);
         const double v = xp / (xm + xp);

         const double Q = (1.0 - 2.0 * v) / invDen;
         const double halfQ = 0.5 * Q;
         const double discriminant = halfQ * halfQ + P3;
         const double sqrtD = std::sqrt(discriminant);

         const double t =
             std::cbrt(-halfQ + sqrtD) +
             std::cbrt(-halfQ - sqrtD);

         out[i] = sign * std::clamp(t, -1.0, 1.0);
      }
   }

   std::vector<double> winv(
       int p, const std::vector<double> &x)
   {
      std::vector<double> out(x.size());
      winv(out, p, x);
      return out;
   }

   double nevill(
       const std::vector<double> &xs,
       std::vector<double> &fs,
       double x)
   {
      assert(xs.size() == fs.size());
      assert(!xs.empty());

      const std::size_t n = xs.size();

      for (std::size_t m = 1; m < n; ++m)
      {
         for (std::size_t j = n - 1; j >= m; --j)
         {
            const double numerator =
                (x - xs[j - m]) * fs[j] -
                (x - xs[j]) * fs[j - 1];

            const double denominator =  xs[j] - xs[j - m];

            fs[j] = numerator / denominator;

            if (j == m)
               break;
         }
      }

      return fs[n - 1];
   }

   double nevill(
       const std::vector<double> &xs,
       const std::vector<double> &fs,
       double x)
   {
      std::vector<double> work = fs;
      return nevill(xs, work, x);
   }

   void ChebyTN(
       std::vector<double> &out,
       int N, double x)
   {
      assert(N >= 1);
      assert(out.size() >= static_cast<std::size_t>(N));

      out[0] = 1.0;

      if (N >= 2)
      {
         out[1] = x;

         for (int n = 2; n < N; ++n)
            out[n] = 2.0 * x * out[n - 1] - out[n - 2];
      }
   }

   void ChebyTN(
       Eigen::MatrixXd &out,
       int N, const std::vector<double> &x)
   {
      assert(N >= 1);
      const Eigen::Index rows = static_cast<Eigen::Index>(x.size());
      assert(out.rows() == rows);
      assert(out.cols() == N);

      out.col(0).setOnes();

      if (N >= 2)
      {
         for (std::size_t i = 0; i < x.size(); ++i)
            out(static_cast<Eigen::Index>(i), 1) = x[i];

         for (int n = 2; n < N; ++n)
         {
            for (std::size_t i = 0; i < x.size(); ++i)
            {
               const Eigen::Index row =
                   static_cast<Eigen::Index>(i);

               out(row, n) =
                   2.0 * x[i] * out(row, n - 1) -
                   out(row, n - 2);
            }
         }
      }
   }

   Eigen::MatrixXd ChebyTN(
       int N, const std::vector<double> &x)
   {
      assert(N >= 1);

      Eigen::MatrixXd out(
          static_cast<Eigen::Index>(x.size()), N);

      ChebyTN(out, N, x);
      return out;
   }

   double ChebyT(int n, double x)
   {
      assert(n >= 0);

      if (n == 0)
         return 1.0;
      if (n == 1)
         return x;
      if (n == 2)
         return std::fma(2.0 * x, x, -1.0);
      if (n == 3)
         return x * (4.0 * x * x - 3.0);

      if (n % 2 == 0)
      {
         const double Tm = ChebyT(n / 2, x);
         return std::fma(2.0 * Tm, Tm, -1.0);
      }

      const int m = (n - 1) / 2;
      const double Tm = ChebyT(m, x);
      const double Tp = ChebyT(m + 1, x);

      return std::fma(2.0 * Tm, Tp, -x);
   }

   void ChebyT(
       std::vector<double> &out,
       int n, const std::vector<double> &X)
   {
      assert(n >= 0);
      assert(out.size() == X.size());

      if (n == 0)
      {
         std::fill(out.begin(), out.end(), 1.0);
         return;
      }

      if (n == 1)
      {
         out = X;
         return;
      }

      if (n == 2)
      {
         for (std::size_t i = 0; i < X.size(); ++i)
            out[i] = std::fma(2.0 * X[i], X[i], -1.0);

         return;
      }

      const unsigned int u = static_cast<unsigned int>(n);
      const int msb =
          31 - static_cast<int>(std::countl_zero(u));

      for (std::size_t i = 0; i < X.size(); ++i)
      {
         const double x = X[i];

         double a = x;
         double b = std::fma(2.0 * x, x, -1.0);

         for (int k = msb - 1; k >= 0; --k)
         {
            const double d0 = std::fma(2.0 * a, a, -1.0);
            const double d1 = std::fma(2.0 * a, b, -x);

            if (((u >> k) & 1U) != 0U)
            {
               a = d1;
               b = std::fma(2.0 * x, d1, -d0);
            }
            else
            {
               a = d0;
               b = d1;
            }
         }

         out[i] = a;
      }
   }

   std::vector<double> ChebyT(
       int n, const std::vector<double> &X)
   {
      std::vector<double> out(X.size());
      ChebyT(out, n, X);
      return out;
   }

   double dlineseg2D(
       double px, double py,
       double p1x, double p1y,
       double p2x, double p2y)
   {
      const double ex = p2x - p1x;
      const double ey = p2y - p1y;

      const double dx = px - p1x;
      const double dy = py - p1y;

      const double e2 = ex * ex + ey * ey;

      // Degenerate segment: p1 == p2.
      if (e2 == 0.0)
         return std::hypot(dx, dy);

      const double t = (dx * ex + dy * ey) / e2;

      if (t <= 0.0)
         return std::hypot(dx, dy);

      if (t >= 1.0)
         return std::hypot(px - p2x, py - p2y);

      const double qx = p1x + t * ex;
      const double qy = p1y + t * ey;

      return std::hypot(px - qx, py - qy);
   }

} // namespace subroutines