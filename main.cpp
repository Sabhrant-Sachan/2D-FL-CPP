#include <iostream>
#include "subroutines.hpp"

int main()
{

    std::array<double, 8> quad{
    -1.0, -1.0,
     1.0, -1.0,
     1.0,  1.0,
    -1.0,  1.0};

    subroutines::makeF1W();

    std::array<double, 8> extended;
    subroutines::extendqua(extended, quad, 0.2);



    std::cout << "Extended quadrilateral coordinates:\n";
    for (int i = 0; i < extended.size(); i += 2) {
        std::cout << "(" << extended[i] << ", " << extended[i + 1] << ")\n";
    }

    std::cout << sizeof(double) << " bytes per double.\n";
    std::cout << "C++ setup is working.\n";

    return 0;
}