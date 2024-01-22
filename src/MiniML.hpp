#pragma once
#define MINIML_API __declspec(dllexport)
#include "Headers/Network.hpp"
#include <iostream>
#include "Eigen/Dense"


namespace MiniML{
    Network* setupXor(int i,int y,int h,int z,bool b){
        Network* net = new Network(i,y,h,z,b);
        return net;
    }

    extern "C" MINIML_API Network* SetUpNetwork(int i,int y,int h,int z,bool b){
        Network* net = new Network(i,y,h,z,b);
        return net;
    }
}   