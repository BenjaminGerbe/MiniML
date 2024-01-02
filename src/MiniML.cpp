#include <iostream>
#include "Eigen/Dense"
#include "Headers/Network.hpp"

namespace MiniML{
    void say_hello(){
        Network net(2,1,1);
        std::cout << net.GetValue(1,1) << std::endl;
    }
}   