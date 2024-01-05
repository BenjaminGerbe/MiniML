#pragma once
#include "Headers/Network.hpp"
#include <iostream>
#include "Eigen/Dense"

namespace MiniML{
    Network* setupXor(int i,int y,int h,int z){
        Network* net = new Network(i,y,h,z);
        return net;
    }
}   