#define  MINIML_EXPORT __declspec(dllexport)
#include "MiniML.hpp"
#include "Headers/Network.hpp"
#include <iostream>

MINIML_EXPORT void* SetUpNetwork(int nbInput,int nbHidden,int lengthLayer,int nbOutput,bool regression){
    Network* net = new Network(nbInput,nbHidden,lengthLayer,nbOutput,regression);
    return net;
}

MINIML_EXPORT int GetNetworkSize(void* network){
    Network* net = (Network*)(network);
    return  net->GetNetworkSize();
}

MINIML_EXPORT int GetLayerSize(void* network,int i){
    Network* net = (Network*)(network);
    return  net->GetLayerSize(i);
}

MINIML_EXPORT int GetLayerRealSize(void* network,int i){
    Network* net = (Network*)(network);
    return  net->GetLayerRealSize(i);
}

MINIML_EXPORT float GetWeight(void* network,int i,int k,int j){
    Network* net = (Network*)(network);
    return net->GetWeight(i,k,j);
}

MINIML_EXPORT float* GetError(void* network){
    Network* net = (Network*)network;
    return net->GetError();
}

MINIML_EXPORT float* GetIter(void* network){
    Network* net = (Network*)network;
    return net->GetItr();
}

MINIML_EXPORT int GetSizeError(void* network){
    Network* net = (Network*)network;
    return net->GetSizeError();
}

MINIML_EXPORT float* SimulateNetwork(void* network,float* input,int n){
    Network* net = (Network*)network;
    std::vector<float> value;
    for (int i = 0; i < n; i++)
    {
        value.push_back(input[i]);
    }
    return &net->simulate(value)[0];
}

MINIML_EXPORT void BackPropagation(void* network,float* input,int ninput,float* output,int noutput,float learningRate,float maxIteration){
    Network* net = (Network*)network;
    std::vector<std::vector<float>> inputVec;
    std::vector<std::vector<float>> outputVec;

    for (size_t i = 0; i < ninput; i++)
    {
        std::vector<float> in;
        for (int j = 0; j < net->GetLayerSize(0); j++)
        {
            in.push_back(input[i+j]);
        }
        inputVec.push_back(in);
    }

    for (size_t i = 0; i < noutput; i++)
    {
        std::vector<float> ot;
        for (int j = 0; j < net->GetLayerSize(net->GetNetworkSize()-1); j++)
        {
            ot.push_back(output[i+j]);
        }
        outputVec.push_back(ot);
    }
    
    net->backPropagation(inputVec,outputVec,learningRate,maxIteration);
}