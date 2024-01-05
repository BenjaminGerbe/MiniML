#pragma once
#include "Headers/Network.hpp"
#include <iostream>
#include "Eigen/Dense"

namespace MiniML{
    Network* setupXor(int i,int y,int z){
        Network* net = new Network(i,y,z);
        return net;
    }
}   