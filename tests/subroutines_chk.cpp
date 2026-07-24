#include "subroutines.hpp"

#include <filesystem>

#include <mkl_cblas.h>
#include <numeric>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
#include <limits>
#include <tuple>

namespace
{
    int failures = 0;

    bool approx(double a, double b, double tol = 1.0e-12)
    {
        return std::abs(a - b) <= tol;
    }

    void check(bool condition, const std::string &name)
    {
        if (condition)
            std::cout << "[PASS] " << name << '\n';
        else
        {
            std::cout << "[FAIL] " << name << '\n';
            ++failures;
        }
    }

    template <int N>
    bool array_approx(
        const std::array<double, N> &a,
        const std::array<double, N> &b,
        double tol = 1.0e-12)
    {
        for (int i = 0; i < N; ++i)
            if (!approx(a[i], b[i], tol))
                return false;

        return true;
    }
}

int main()
{
    using namespace subroutines;

    constexpr double pi = std::numbers::pi;

    // ------------------------------------------------------------
    // lineinter
    // ------------------------------------------------------------
    {
        const auto [v, u] = lineinter(
            -2.0, 0.0, 2.0, 0.0,
            0.0, -1.0, 0.0, 1.0);

        check(approx(v, 0.5) && approx(u, 0.5),
              "lineinter: horizontal and vertical");
    }

    {
        const auto [v, u] = lineinter(
            0.0, 0.0, 2.0, 2.0,
            0.0, 2.0, 2.0, 0.0);

        check(approx(v, 0.5) && approx(u, 0.5),
              "lineinter: crossing diagonals");
    }

    {
        const auto [v, u] = lineinter(
            0.0, 1.0, 4.0, 1.0,
            3.0, 0.0, 3.0, 2.0);

        check(approx(v, 0.75) && approx(u, 0.5),
              "lineinter: unequal line parameterizations");
    }

    // ------------------------------------------------------------
    // tanginterx
    // ------------------------------------------------------------
    {
        const auto [P1x, P1y, P2x, P2y] = tanginterx(
            0.0, 0.0, 2.0, 0.0,
            1.0, -1.0, 0.0, 2.0,
            0.0, 2.0, 4.0, 2.0);

        check(
            approx(P1x, 1.0) && approx(P1y, 0.0) &&
                approx(P2x, 1.0) && approx(P2y, 2.0),
            "tanginterx: vertical tangent");
    }

    {
        const auto [P1x, P1y, P2x, P2y] = tanginterx(
            0.0, 0.0, 0.0, 2.0,
            -1.0, 1.0, 4.0, 0.0,
            2.0, 0.0, 2.0, 4.0);

        check(
            approx(P1x, 0.0) && approx(P1y, 1.0) &&
                approx(P2x, 2.0) && approx(P2y, 1.0),
            "tanginterx: horizontal tangent");
    }

    // ------------------------------------------------------------
    // tanginterp
    // ------------------------------------------------------------
    {
        const auto [u, v] = tanginterp(
            0.0, 0.0, 2.0, 0.0,
            1.0, -1.0, 0.0, 2.0,
            0.0, 2.0, 4.0, 2.0);

        check(approx(u, 0.25) && approx(v, 0.5),
              "tanginterp: vertical tangent");
    }

    {
        const auto [u, v] = tanginterp(
            0.0, 0.0, 0.0, 2.0,
            -1.0, 1.0, 4.0, 0.0,
            2.0, 0.0, 2.0, 4.0);

        check(approx(u, 0.25) && approx(v, 0.5),
              "tanginterp: horizontal tangent");
    }

    // ------------------------------------------------------------
    // extendqua
    // ------------------------------------------------------------
    {
        const std::array<double, 8> quad{
            -1.0, -1.0, 1.0, -1.0,
            1.0, 1.0, -1.0, 1.0};

        const std::array<double, 8> expected{
            -1.2, -1.2, 1.2, -1.2,
            1.2, 1.2, -1.2, 1.2};

        std::array<double, 8> result;
        extendqua(result, quad, 0.2);

        check(array_approx(result, expected),
              "extendqua: counterclockwise square");
    }

    {
        const std::array<double, 8> rectangle{
            0.0, 0.0, 2.0, 0.0,
            2.0, 1.0, 0.0, 1.0};

        const std::array<double, 8> expected{
            -0.5, -0.5, 2.5, -0.5,
            2.5, 1.5, -0.5, 1.5};

        std::array<double, 8> result;
        extendqua(result, rectangle, 0.5);

        check(array_approx(result, expected),
              "extendqua: rectangle");
    }

    {
        const std::array<double, 8> clockwise{
            0.0, 0.0, 0.0, 1.0,
            2.0, 1.0, 2.0, 0.0};

        const std::array<double, 8> expected{
            -0.5, -0.5, -0.5, 1.5,
            2.5, 1.5, 2.5, -0.5};

        std::array<double, 8> result;
        extendqua(result, clockwise, 0.5);

        check(array_approx(result, expected),
              "extendqua: clockwise orientation");
    }

    // ------------------------------------------------------------
    // Scalar ptinqua
    // Diamond: (-1,0), (0,-1), (1,0), (0,1)
    // ------------------------------------------------------------
    {
        check(ptinqua(
                  0.0, 0.0,
                  -1.0, 0.0, 0.0, -1.0,
                  1.0, 0.0, 0.0, 1.0),
              "ptinqua scalar: interior point");
    }

    {
        check(ptinqua(
                  1.0, 0.0,
                  -1.0, 0.0, 0.0, -1.0,
                  1.0, 0.0, 0.0, 1.0),
              "ptinqua scalar: boundary point");
    }

    {
        check(!ptinqua(
                  1.0, 1.0,
                  -1.0, 0.0, 0.0, -1.0,
                  1.0, 0.0, 0.0, 1.0),
              "ptinqua scalar: exterior point");
    }

    // ------------------------------------------------------------
    // Mutating and allocating ptinqua
    // ------------------------------------------------------------
    {
        const std::vector<std::array<double, 2>> points{
            {0.0, 0.0},
            {1.0, 0.0},
            {1.0, 1.0},
            {0.0, -0.5}};

        std::vector<bool> inside(points.size());

        ptinqua(
            inside, points,
            -1.0, 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0, 1.0);

        check(
            inside[0] && inside[1] &&
                !inside[2] && inside[3],
            "ptinqua mutating vector version");

        const std::vector<bool> allocated = ptinqua(
            points,
            -1.0, 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0, 1.0);

        check(allocated == inside,
              "ptinqua allocating vector version");
    }

    // ------------------------------------------------------------
    // ellipselinter
    // Ellipse: x = 2 cos(theta), y = sin(theta)
    // ------------------------------------------------------------
    {
        const auto [Et1, Lt1, Et2, Lt2] = ellipselinter(
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            -3.0, 0.0, 1.0, 0.0);

        check(
            approx(Et1, 0.0) && approx(Lt1, 5.0) &&
                approx(Et2, pi) && approx(Lt2, 1.0),
            "ellipselinter: horizontal secant");
    }

    {
        const auto [Et1, Lt1, Et2, Lt2] = ellipselinter(
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            -1.0, 1.0, 1.0, 0.0);

        check(
            approx(Et1, pi / 2.0) && approx(Lt1, 1.0) &&
                approx(Et2, pi / 2.0) && approx(Lt2, 1.0),
            "ellipselinter: tangent line");
    }

    {
        const auto [Et1, Lt1, Et2, Lt2] = ellipselinter(
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            0.0, 2.0, 1.0, 0.0);

        check(
            std::isnan(Et1) && std::isnan(Lt1) &&
                std::isnan(Et2) && std::isnan(Lt2),
            "ellipselinter: no intersection");
    }

    // ------------------------------------------------------------
    // ellipserinter
    // ------------------------------------------------------------
    {
        const auto [Et, Lt] = ellipserinter(
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            -3.0, 0.0, 1.0, 0.0);

        check(approx(Et, pi) && approx(Lt, 1.0),
              "ellipserinter: first forward intersection");
    }

    {
        const auto [Et, Lt] = ellipserinter(
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0);

        check(approx(Et, 0.0) && approx(Lt, 2.0),
              "ellipserinter: ray starting inside");
    }

    {
        const auto [Et, Lt] = ellipserinter(
            0.0, 0.0, 2.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            0.0, 2.0, 1.0, 0.0);

        check(std::isnan(Et) && std::isnan(Lt),
              "ellipserinter: no intersection");
    }
    // ------------------------------------------------------------
    // wfunc
    // ------------------------------------------------------------
    {
        const std::vector<double> t{-1.0, 0.0, 1.0};
        const std::vector<double> result = wfunc(4, t);

        check(
            approx(result[0], 0.0) &&
                approx(result[1], 1.0) &&
                approx(result[2], 2.0),
            "wfunc: endpoint and midpoint values");
    }

    {
        const std::vector<double> t{-0.5, 0.0, 0.5};

        std::vector<double> mutating(t.size());
        wfunc(mutating, 4, t);

        const std::vector<double> allocating = wfunc(4, t);

        check(mutating == allocating,
              "wfunc: mutating and allocating versions agree");
    }

    {
        const std::vector<double> t{-1.0, 0.0, 1.0};

        // beta scales the output by 3.
        const std::vector<double> result =
            wfunc(4, t, 1.0, 3.0, 0.0);

        check(
            approx(result[0], 0.0) &&
                approx(result[1], 3.0) &&
                approx(result[2], 6.0),
            "wfunc: beta scaling");
    }

    // ------------------------------------------------------------
    // wsfunc with rational exponent P/Q
    // ------------------------------------------------------------
    {
        const std::vector<double> t{-1.0, 0.0, 1.0};
        const std::vector<double> result =
            wsfunc(4, 1, 2, t);

        check(
            approx(result[0], 0.0) &&
                approx(result[1], 1.0) &&
                approx(result[2], std::sqrt(2.0)),
            "wsfunc rational: square root");
    }

    {
        const std::vector<double> t{-0.5, 0.0, 0.5};

        std::vector<double> mutating(t.size());
        wsfunc(mutating, 4, 3, 2, t);

        const std::vector<double> allocating =
            wsfunc(4, 3, 2, t);

        bool same = true;
        for (int i = 0; i < t.size(); ++i)
            same = same && approx(mutating[i], allocating[i]);

        check(same,
              "wsfunc rational: mutating and allocating agree");
    }

    // ------------------------------------------------------------
    // wsfunc with real exponent s
    // ------------------------------------------------------------
    {
        const std::vector<double> t{-1.0, 0.0, 1.0};
        const std::vector<double> result = wsfunc(4, 0.5, t);

        check(
            approx(result[0], 0.0) &&
                approx(result[1], 1.0) &&
                approx(result[2], std::sqrt(2.0)),
            "wsfunc real: square root");
    }

    {
        const std::vector<double> t{-0.75, 0.0, 0.75};

        const std::vector<double> rational = wsfunc(4, 3, 2, t);

        const std::vector<double> real = wsfunc(4, 1.5, t);

        bool same = true;
        for (int i = 0; i < t.size(); ++i)
            same = same && approx(rational[i], real[i]);

        check(same,
              "wsfunc: rational and real exponents agree");
    }

    {
        const std::vector<double> t{-0.5, 0.0, 0.5};

        const std::vector<double> w = wfunc(4, t);
        const std::vector<double> ws = wsfunc(4, 2.0, t);

        bool correct_power = true;
        for (int i = 0; i < t.size(); ++i)
            correct_power = correct_power && approx(ws[i], w[i] * w[i]);

        check(correct_power,
              "wsfunc real: equals wfunc raised to s");
    }
    // ------------------------------------------------------------
    // dwfunc
    // ------------------------------------------------------------
    {
        const std::vector<double> t{-1.0, 0.0, 1.0};
        const std::vector<double> result = dwfunc(4, t);

        check(
            approx(result[0], 0.0) &&
                approx(result[1], 2.0) &&
                approx(result[2], 0.0),
            "dwfunc: endpoint and midpoint identities");
    }

    {
        const std::vector<double> t{-0.75, -0.25, 0.25, 0.75};
        const std::vector<double> result = dwfunc(6, t);

        check(
            approx(result[0], result[3]) &&
                approx(result[1], result[2]),
            "dwfunc: even symmetry");
    }

    {
        const std::vector<double> t{-0.5, 0.0, 0.5};
        std::vector<double> mutating(t.size());

        dwfunc(mutating, 4, t);
        const std::vector<double> allocating = dwfunc(4, t);

        bool same = true;
        for (int i = 0; i < t.size(); ++i)
            same = same && approx(mutating[i], allocating[i]);

        check(same, "dwfunc: mutating and allocating agree");
    }

    // ------------------------------------------------------------
    // ddwfunc
    // ------------------------------------------------------------
    {
        const std::vector<double> t{-1.0, 0.0, 1.0};
        const std::vector<double> result = ddwfunc(2, t);

        check(
            approx(result[0], 1.0) &&
                approx(result[1], 0.0) &&
                approx(result[2], -1.0),
            "ddwfunc: p=2 endpoint and midpoint identities");
    }

    {
        const std::vector<double> t{-0.75, -0.25, 0.25, 0.75};
        const std::vector<double> result = ddwfunc(6, t);

        check(
            approx(result[0], -result[3]) &&
                approx(result[1], -result[2]),
            "ddwfunc: odd symmetry");
    }

    {
        const std::vector<double> t{-0.5, 0.0, 0.5};
        std::vector<double> mutating(t.size());

        ddwfunc(mutating, 4, t);
        const std::vector<double> allocating = ddwfunc(4, t);

        bool same = true;
        for (int i = 0; i < t.size(); ++i)
            same = same && approx(mutating[i], allocating[i]);

        check(same, "ddwfunc: mutating and allocating agree");
    }

    // ------------------------------------------------------------
    // qw1func, qw2func and qw3func
    // ------------------------------------------------------------
    {
        const std::vector<double> t{0.0};

        check(approx(qw1func(4, t, 0.75)[0], 2.0),
              "qw1func: value at zero");

        check(approx(qw2func(4, t, 0.75)[0], 2.0),
              "qw2func: value at zero");

        check(approx(qw3func(4, t, 0.75)[0], 2.0),
              "qw3func: value at zero");
    }
    {
        const std::vector<double> t{-0.5, 0.0, 0.5};
        constexpr int p = 4;
        constexpr double s = 0.75;

        std::vector<double> q1_mut(t.size());
        std::vector<double> q2_mut(t.size());
        std::vector<double> q3_mut(t.size());

        qw1func(q1_mut, p, t, s);
        qw2func(q2_mut, p, t, s);
        qw3func(q3_mut, p, t, s);

        const auto q1 = qw1func(p, t, s);
        const auto q2 = qw2func(p, t, s);
        const auto q3 = qw3func(p, t, s);

        bool same1 = true;
        bool same2 = true;
        bool same3 = true;

        for (int i = 0; i < t.size(); ++i)
        {
            same1 = same1 && approx(q1_mut[i], q1[i]);
            same2 = same2 && approx(q2_mut[i], q2[i]);
            same3 = same3 && approx(q3_mut[i], q3[i]);
        }

        check(same1, "qw1func: mutating and allocating agree");
        check(same2, "qw2func: mutating and allocating agree");
        check(same3, "qw3func: mutating and allocating agree");
    }
    {
        constexpr int p = 4;
        constexpr double s = 0.75;

        const std::vector<double> t{-0.6, -0.2, 0.2, 0.6};
        const auto q1 = qw1func(p, t, s);
        const auto q2 = qw2func(p, t, s);
        const auto q3 = qw3func(p, t, s);

        bool correct1 = true;
        bool correct2 = true;
        bool correct3 = true;

        for (int i = 0; i < t.size(); ++i)
        {
            const std::vector<double> positive{t[i]};
            const std::vector<double> negative{-t[i]};

            const double wp = wfunc(p, positive)[0];
            const double wm = wfunc(p, negative)[0];
            const double dw = dwfunc(p, positive)[0];

            const double expected1 =
                dw / std::pow(wm, 2.0 * s - 1.0);

            const double expected2 =
                dw / std::pow(wm, 1.0 - s);

            const double expected3 =
                dw / (std::pow(wm, 2.0 * s - 1.0) *
                      std::pow(wp, 1.0 - s));

            correct1 = correct1 && approx(q1[i], expected1, 1.0e-10);
            correct2 = correct2 && approx(q2[i], expected2, 1.0e-10);
            correct3 = correct3 && approx(q3[i], expected3, 1.0e-10);
        }

        check(correct1, "qw1func: agrees with direct formula");
        check(correct2, "qw2func: agrees with direct formula");
        check(correct3, "qw3func: agrees with direct formula");
    }
    // ------------------------------------------------------------
    // winv
    // ------------------------------------------------------------
    {
        check(
            approx(winv(4, 0.0), -1.0) &&
                approx(winv(4, 1.0), 0.0) &&
                approx(winv(4, 2.0), 1.0),
            "winv: endpoint and midpoint identities");
    }

    {
        check(
            approx(winv(2, 0.0), -1.0) &&
                approx(winv(2, 1.0), 0.0) &&
                approx(winv(2, 2.0), 1.0),
            "winv: p=2 closed form");
    }

    {
        constexpr int p = 4;
        const std::vector<double> t{
            -1.0, -0.75, -0.25, 0.0, 0.3, 0.8, 1.0};

        const std::vector<double> x = wfunc(p, t);
        const std::vector<double> recovered = winv(p, x);

        bool correct = true;
        for (int i = 0; i < t.size(); ++i)
            correct = correct && approx(recovered[i], t[i], 1.0e-11);

        check(correct, "winv: inverse of wfunc");
    }

    {
        constexpr int p = 6;
        const std::vector<double> x{0.1, 0.5, 1.0, 1.5, 1.9};

        std::vector<double> mutating(x.size());
        winv(mutating, p, x);

        const std::vector<double> allocating = winv(p, x);

        bool same = true;
        for (int i = 0; i < x.size(); ++i)
            same = same && approx(mutating[i], allocating[i]);

        check(same, "winv: mutating and allocating agree");
    }

    {
        constexpr int p = 5;
        const std::vector<double> x{0.2, 0.6, 1.4, 1.8};

        bool symmetric = true;

        for (double xi : x)
            symmetric = symmetric &&
                        approx(winv(p, xi), -winv(p, 2.0 - xi));

        check(symmetric, "winv: symmetry");
    }
    // ------------------------------------------------------------
    // nevill
    // ------------------------------------------------------------
    {
        // f(x) = x^2 + 2x + 1
        const std::vector<double> xs{0.0, 1.0, 2.0};
        const std::vector<double> fs{1.0, 4.0, 9.0};

        check(
            approx(nevill(xs, fs, 1.5), 6.25),
            "nevill: quadratic interpolation");
    }

    {
        // f(x) = 2x - 3
        const std::vector<double> xs{-1.0, 0.0, 2.0};
        const std::vector<double> fs{-5.0, -3.0, 1.0};

        check(
            approx(nevill(xs, fs, 0.5), -2.0),
            "nevill: linear interpolation");
    }

    {
        const std::vector<double> xs{0.0, 1.0, 2.0};
        const std::vector<double> original{1.0, 4.0, 9.0};

        std::vector<double> work = original;

        const double mutating = nevill(xs, work, 1.5);
        const double nonmutating = nevill(xs, original, 1.5);

        check(
            approx(mutating, nonmutating) &&
                work != original,
            "nevill: mutating and non-mutating agree");
    }
    // ------------------------------------------------------------
    // ChebyTN: scalar output
    // ------------------------------------------------------------
    {
        std::vector<double> out(6);
        ChebyTN(out, 6, 1.0);

        bool correct = true;
        for (double value : out)
            correct = correct && approx(value, 1.0);

        check(correct, "ChebyTN scalar: x=1");
    }

    {
        std::vector<double> out(6);
        ChebyTN(out, 6, -1.0);

        const std::vector<double> expected{
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0};

        bool correct = true;
        for (int i = 0; i < out.size(); ++i)
            correct = correct && approx(out[i], expected[i]);

        check(correct, "ChebyTN scalar: x=-1");
    }

    {
        std::vector<double> out(7);
        ChebyTN(out, 7, 0.0);

        const std::vector<double> expected{
            1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0};

        bool correct = true;
        for (int i = 0; i < out.size(); ++i)
            correct = correct && approx(out[i], expected[i]);

        check(correct, "ChebyTN scalar: x=0");
    }

    const auto at = [](const std::vector<double> &matrix,
                       int row,
                       int col,
                       int columnCount)
    {
        return matrix[row * columnCount + col];
    };
    // ------------------------------------------------------------
    // ChebyTN: flat row-major matrix output
    // ------------------------------------------------------------
    {
        const std::vector<double> x{-1.0, 0.0, 1.0};
        constexpr int N = 6;

        const std::vector<double> out = ChebyTN(N, x);

        bool correct = true;

        for (int n = 0; n < N; ++n)
        {

            correct =
                correct &&
                approx(
                    at(out, 0, n, N),
                    n % 2 == 0 ? 1.0 : -1.0);

            const double zeroExpected =
                n % 2 == 0
                    ? (n % 4 == 0 ? 1.0 : -1.0)
                    : 0.0;

            correct =
                correct &&
                approx(at(out, 1, n, N), zeroExpected);

            correct =
                correct &&
                approx(at(out, 2, n, N), 1.0);
        }

        check(correct, "ChebyTN matrix: x=-1,0,1");
    }

    {
        const std::vector<double> x{-0.75, 0.0, 0.4, 1.0};
        constexpr int N = 8;

        std::vector<double> mutating(x.size() * N);
        ChebyTN(mutating, N, x);

        const std::vector<double> allocating = ChebyTN(N, x);

        bool agree = true;

        for (int i = 0; i < mutating.size(); ++i)
        {
            agree =
                agree &&
                approx(mutating[i], allocating[i], 1.0e-12);
        }

        check(
            agree,
            "ChebyTN matrix: mutating and allocating agree");
    }

    {
        const std::vector<double> x{-0.8, -0.2, 0.3, 0.9};
        constexpr int N = 10;

        const std::vector<double> matrix = ChebyTN(N, x);

        bool agreesWithScalar = true;

        for (int i = 0; i < x.size(); ++i)
        {
            std::vector<double> scalarRow(N);
            ChebyTN(scalarRow, N, x[i]);

            for (int n = 0; n < N; ++n)
            {

                agreesWithScalar =
                    agreesWithScalar &&
                    approx(
                        at(matrix, i, n, N),
                        scalarRow[n]);
            }
        }

        check(
            agreesWithScalar,
            "ChebyTN matrix: rows agree with scalar version");
    }
    // ------------------------------------------------------------
    // ChebyTN: MKL matrix-vector multiplication
    // ------------------------------------------------------------
    {
        const std::vector<double> x{-1.0, -0.5, 0.0, 0.5, 1.0};
        constexpr int N = 3;

        const std::vector<double> matrix = ChebyTN(N, x);

        // p(x) = 2*T0(x) + 3*T1(x) - T2(x)
        const std::vector<double> coefficients{
            2.0,
            3.0,
            -1.0};

        std::vector<double> result(x.size(), 0.0);

        const MKL_INT rows =
            static_cast<MKL_INT>(x.size());

        const MKL_INT columns =
            static_cast<MKL_INT>(N);

        cblas_dgemv(
            CblasRowMajor,
            CblasNoTrans,
            rows,
            columns,
            1.0,
            matrix.data(),
            columns,
            coefficients.data(),
            1,
            0.0,
            result.data(),
            1);

        bool correct = true;

        for (int i = 0; i < x.size(); ++i)
        {
            // T0(x) = 1
            // T1(x) = x
            // T2(x) = 2*x*x - 1
            const double expected =
                2.0 +
                3.0 * x[i] -
                (2.0 * x[i] * x[i] - 1.0);

            correct =
                correct &&
                approx(result[i], expected, 1.0e-12);
        }

        check(
            correct,
            "ChebyTN matrix: MKL matrix-vector multiplication");
    }

    // ------------------------------------------------------------
    // ChebyT: scalar
    // ------------------------------------------------------------
    {
        bool correct = true;

        for (int n = 0; n <= 20; ++n)
        {
            correct = correct &&
                      approx(ChebyT(n, 1.0), 1.0);

            correct = correct &&
                      approx(
                          ChebyT(n, -1.0),
                          n % 2 == 0 ? 1.0 : -1.0);

            const double zeroExpected =
                n % 2 == 0
                    ? (n % 4 == 0 ? 1.0 : -1.0)
                    : 0.0;

            correct = correct &&
                      approx(ChebyT(n, 0.0), zeroExpected);
        }

        check(correct, "ChebyT scalar: exact values at -1,0,1");
    }

    {
        const double x = 0.37;

        check(
            approx(ChebyT(0, x), 1.0) &&
                approx(ChebyT(1, x), x) &&
                approx(ChebyT(2, x), 2.0 * x * x - 1.0) &&
                approx(ChebyT(3, x), x * (4.0 * x * x - 3.0)),
            "ChebyT scalar: low-degree formulas");
    }

    {
        const double x = 0.41;

        bool recurrenceHolds = true;

        for (int n = 1; n < 20; ++n)
        {
            const double lhs = ChebyT(n + 1, x);
            const double rhs =
                2.0 * x * ChebyT(n, x) - ChebyT(n - 1, x);

            recurrenceHolds =
                recurrenceHolds && approx(lhs, rhs, 1.0e-12);
        }

        check(
            recurrenceHolds,
            "ChebyT scalar: three-term recurrence");
    }
    // ------------------------------------------------------------
    // ChebyT: vector
    // ------------------------------------------------------------
    {
        const std::vector<double> X{-1.0, 0.0, 1.0};
        constexpr int n = 7;

        const std::vector<double> out = ChebyT(n, X);

        check(
            approx(out[0], -1.0) &&
                approx(out[1], 0.0) &&
                approx(out[2], 1.0),
            "ChebyT vector: exact values at -1,0,1");
    }

    {
        const std::vector<double> X{
            -0.95, -0.5, -0.1, 0.0, 0.25, 0.7, 1.0};

        constexpr int n = 13;

        const std::vector<double> out = ChebyT(n, X);

        bool agreesWithScalar = true;

        for (int i = 0; i < X.size(); ++i)
        {
            agreesWithScalar =
                agreesWithScalar &&
                approx(out[i], ChebyT(n, X[i]), 1.0e-12);
        }

        check(
            agreesWithScalar,
            "ChebyT vector: agrees with scalar version");
    }

    {
        const std::vector<double> X{-0.8, -0.3, 0.2, 0.9};
        constexpr int n = 18;

        std::vector<double> mutating(X.size());
        ChebyT(mutating, n, X);

        const std::vector<double> allocating = ChebyT(n, X);

        bool same = true;

        for (int i = 0; i < X.size(); ++i)
            same = same && approx(mutating[i], allocating[i]);

        check(
            same,
            "ChebyT vector: mutating and allocating agree");
    }

    {
        const std::vector<double> X{-0.9, -0.2, 0.4, 0.85};
        constexpr int N = 15;

        const std::vector<double> allDegrees = ChebyTN(N, X);
        const std::vector<double> degreeNMinus1 = ChebyT(N - 1, X);

        bool same = true;

        for (int i = 0; i < X.size(); ++i)
        {
            same = same &&
                approx(at(allDegrees, i, N - 1, N),
                    degreeNMinus1[i], 1.0e-12);
        }

        check(same, "ChebyT and ChebyTN agree");
    }
    // ------------------------------------------------------------
    // dlineseg2D
    // ------------------------------------------------------------
    {
        // Projection lies inside the segment.
        const double d = dlineseg2D(
            1.0, 2.0,
            0.0, 0.0,
            3.0, 0.0);

        check(
            approx(d, 2.0),
            "dlineseg2D: perpendicular projection");
    }

    {
        // Closest point is p1.
        const double d = dlineseg2D(
            -1.0, 1.0,
            0.0, 0.0,
            3.0, 0.0);

        check(
            approx(d, std::sqrt(2.0)),
            "dlineseg2D: closest to first endpoint");
    }

    {
        // Closest point is p2.
        const double d = dlineseg2D(
            4.0, 3.0,
            0.0, 0.0,
            3.0, 0.0);

        check(
            approx(d, std::sqrt(10.0)),
            "dlineseg2D: closest to second endpoint");
    }

    {
        // Degenerate segment.
        const double d = dlineseg2D(
            4.0, 6.0,
            1.0, 2.0,
            1.0, 2.0);

        check(
            approx(d, 5.0),
            "dlineseg2D: degenerate segment");
    }

    {
        // Point lies on the segment.
        const double d = dlineseg2D(
            1.5, 0.0,
            0.0, 0.0,
            3.0, 0.0);

        check(
            approx(d, 0.0),
            "dlineseg2D: point on segment");
    }

    {
        // Diagonal segment from (0,0) to (2,2).
        const double d = dlineseg2D(
            1.0, 0.0,
            0.0, 0.0,
            2.0, 2.0);

        check(
            approx(d, 1.0 / std::sqrt(2.0)),
            "dlineseg2D: diagonal segment");
    }

    // ------------------------------------------------------------
    // GSS
    // ------------------------------------------------------------
    {
        const auto f = [](double x)
        {
            return (x - 2.0) * (x - 2.0) + 3.0;
        };

        const auto [fmin, xmin] = GSS(f, -5.0, 8.0, 1.0e-12);

        check(
            approx(xmin, 2.0, 1.0e-7) &&
                approx(fmin, 3.0, 1.0e-14),
            "GSS: quadratic minimum");
    }

    {
        const auto f = [](double x)
        {
            return x * x;
        };

        const auto [fmin, xmin] = GSS(f, -1.0, 1.0, 1.0e-12);

        check(
            approx(xmin, 0.0, 1.0e-7) &&
                approx(fmin, 0.0, 1.0e-14),
            "GSS: symmetric minimum");
    }

    {
        const auto f = [](double x)
        {
            return std::cos(x);
        };

        const auto [fmin, xmin] = GSS(f, 2.0, 4.0, 1.0e-12);

        check(
            approx(xmin, std::numbers::pi, 1.0e-7) &&
                approx(fmin, -1.0, 1.0e-14),
            "GSS: cosine minimum");
    }

    // ------------------------------------------------------------
    // Bis
    // ------------------------------------------------------------
    {
        const auto f = [](double x)
        {
            return x * x - 2.0;
        };

        const double root = Bis(f, 0.0, 2.0, 100);

        check(
            approx(root, std::sqrt(2.0), 1.0e-12),
            "Bis: square root of two");
    }

    {
        const auto f = [](double x)
        {
            return x * x * x - x - 2.0;
        };

        const double root = Bis(f, 1.0, 2.0, 100);

        check(
            approx(root, 1.5213797068045676, 1.0e-12),
            "Bis: cubic root");
    }

    {
        const auto f = [](double x)
        {
            return std::sin(x);
        };

        const double root = Bis(f, 3.0, 4.0, 100);

        check(
            approx(root, std::numbers::pi, 1.0e-12),
            "Bis: root of sine");
    }

    // ------------------------------------------------------------
    // newtonR1D
    // ------------------------------------------------------------
    {
        const auto f = [](double x)
        {
            return x * x - 2.0;
        };

        const auto df = [](double x)
        {
            return 2.0 * x;
        };

        const double root = newtonR1D(f, df, 1.0, 20);

        check(
            approx(root, std::sqrt(2.0), 1.0e-14),
            "newtonR1D: square root of two");
    }

    {
        const auto f = [](double x)
        {
            return std::cos(x) - x;
        };

        const auto df = [](double x)
        {
            return -std::sin(x) - 1.0;
        };

        const double root = newtonR1D(f, df, 1.0, 20);

        check(
            approx(root, 0.7390851332151607, 1.0e-14),
            "newtonR1D: cos(x) = x");
    }

    {
        const auto f = [](double x)
        {
            return x * x * x - 2.0 * x - 5.0;
        };

        const auto df = [](double x)
        {
            return 3.0 * x * x - 2.0;
        };

        const double root = newtonR1D(f, df, 2.0, 20);

        check(
            approx(root, 2.0945514815423265, 1.0e-13),
            "newtonR1D: cubic root");
    }
    // ------------------------------------------------------------
    // newtonR2D
    // ------------------------------------------------------------
    {
        // Linear system:
        // t + s = 3
        // t - s = 1
        // Solution: (2,1)

        const auto f1 = [](double t, double s)
        {
            return t + s - 3.0;
        };

        const auto f2 = [](double t, double s)
        {
            return t - s - 1.0;
        };

        const auto J = [](double, double)
        {
            return std::tuple<double, double, double, double>{
                1.0, 1.0,
                1.0, -1.0};
        };

        const auto [t, s] =
            newtonR2D(f1, f2, J, 0.0, 0.0, 20);

        check(
            approx(t, 2.0, 1.0e-14) &&
                approx(s, 1.0, 1.0e-14),
            "newtonR2D: linear system");
    }
    {
        // Circle and line:
        // t^2 + s^2 = 2
        // t = s
        // Starting near the positive root gives (1,1).

        const auto f1 = [](double t, double s)
        {
            return t * t + s * s - 2.0;
        };

        const auto f2 = [](double t, double s)
        {
            return t - s;
        };

        const auto J = [](double t, double s)
        {
            return std::tuple<double, double, double, double>{
                2.0 * t, 2.0 * s,
                1.0, -1.0};
        };

        const auto [t, s] =
            newtonR2D(f1, f2, J, 0.8, 1.2, 30);

        check(
            approx(t, 1.0, 1.0e-13) &&
                approx(s, 1.0, 1.0e-13),
            "newtonR2D: nonlinear system");
    }
    {
        // Another nonlinear system:
        // t^2 - s = 0
        // t + s - 2 = 0
        //
        // Positive solution:
        // t = 1, s = 1.

        const auto f1 = [](double t, double s)
        {
            return t * t - s;
        };

        const auto f2 = [](double t, double s)
        {
            return t + s - 2.0;
        };

        const auto J = [](double t, double)
        {
            return std::tuple<double, double, double, double>{
                2.0 * t, -1.0,
                1.0, 1.0};
        };

        const auto [t, s] =
            newtonR2D(f1, f2, J, 0.7, 1.1, 30);

        check(
            approx(t, 1.0, 1.0e-13) &&
                approx(s, 1.0, 1.0e-13),
            "newtonR2D: parabola-line intersection");
    }
    {
        const auto f1 = [](double t, double s)
        {
            return t + s;
        };

        const auto f2 = [](double t, double s)
        {
            return 2.0 * t + 2.0 * s;
        };

        const auto J = [](double, double)
        {
            return std::tuple<double, double, double, double>{
                1.0, 1.0,
                2.0, 2.0};
        };

        const auto [t, s] =
            newtonR2D(f1, f2, J, 1.0, 1.0, 10);

        check(
            std::isnan(t) && std::isnan(s),
            "newtonR2D: singular Jacobian returns NaN");
    }
    {
        const auto f1 = [](double t, double)
        {
            return t * t - 2.0;
        };

        const auto f2 = [](double, double s)
        {
            return s * s - 3.0;
        };

        const auto J = [](double t, double s)
        {
            return std::tuple<double, double, double, double>{
                2.0 * t, 0.0,
                0.0, 2.0 * s};
        };

        // One iteration is not enough to converge from this start.
        const auto [t, s] =
            newtonR2D(f1, f2, J, 10.0, 10.0, 1);

        check(
            std::isnan(t) && std::isnan(s),
            "newtonR2D: maximum iterations returns NaN");
    }

    // ------------------------------------------------------------
    // fejer1w
    // ------------------------------------------------------------
    {
        const std::vector<double> w = fejer1w(1);

        check(w.size() == 1 && approx(w[0], 2.0), "fejer1w: n=1");
    }

    {
        const std::vector<double> w = fejer1w(2);

        check(w.size() == 2 && approx(w[0], 1.0) &&
                  approx(w[1], 1.0),
              "fejer1w: n=2");
    }

    {
        const std::vector<double> w = fejer1w(3);

        check(w.size() == 3 &&
                  approx(w[0], 4.0 / 9.0, 1.0e-14) &&
                  approx(w[1], 10.0 / 9.0, 1.0e-14) &&
                  approx(w[2], 4.0 / 9.0, 1.0e-14),
              "fejer1w: n=3");
    }

    {
        const std::vector<double> w = fejer1w(12);

        bool symmetric = true;

        for (int i = 0; i < w.size(); ++i)
        {
            symmetric =
                symmetric &&
                approx(w[i], w[w.size() - 1U - i], 1.0e-14);
        }

        const double sum =
            std::accumulate(w.begin(), w.end(), 0.0);

        check(
            symmetric && approx(sum, 2.0, 1.0e-14),
            "fejer1w: symmetry and sum");
    }

    // ------------------------------------------------------------
    // makeF1W and getF1W
    // ------------------------------------------------------------
    {
        const std::filesystem::path file = "F1W_test.bin";

        makeF1W(8, file);

        bool correct = true;

        for (int n = 1; n <= 8; ++n)
        {
            const std::vector<double> stored = getF1W(n, file);

            const std::vector<double> expected = fejer1w(n);

            if (stored.size() != expected.size())
            {
                correct = false;
                break;
            }

            for (int i = 0; i < stored.size(); ++i)
            {
                correct = correct && approx(stored[i], expected[i], 1.0e-14);
            }
        }

        check(correct, "makeF1W/getF1W: packed file round trip");

        std::filesystem::remove(file);
    }

    std::cout << "\n";

    if (failures == 0)
    {
        std::cout << "All tests passed.\n";
        return 0;
    }

    std::cout << failures << " test(s) failed.\n";
    return 1;
}