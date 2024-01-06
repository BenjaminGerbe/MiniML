#pragma once
#include "Headers/Network.hpp"
#include <iostream>
#include "Eigen/Dense"

namespace MiniML{
    Network* setupXor(int i,int y,int h,int z,bool b){
        Network* net = new Network(i,y,h,z,b);
        return net;
    }
}   