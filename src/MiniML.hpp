#pragma once
#include "Headers/Network.hpp"
#include <iostream>
#include "Eigen/Dense"

namespace MiniML{
    Network* setupXor(int i,int y,int z){
        Network* net = new Network(i,y,z);
        net->SetWeight(1,0,0,1);
        net->SetWeight(1,0,1,1);
        net->SetWeight(1,0,2,-0.5);

        net->SetWeight(1,1,0,-1);
        net->SetWeight(1,1,1,-1);
        net->SetWeight(1,1,2,1.5);
        
        net->SetWeight(2,0,0,1);
        net->SetWeight(2,0,1,1);
        net->SetWeight(2,0,2,-1.5);
        return net;
    }
}   