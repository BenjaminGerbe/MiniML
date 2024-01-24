#define  MINIML_EXPORT __declspec(dllexport)
#include "../include/MiniML.hpp"
#include "Headers/Network.hpp"
#include <iostream>

void* MiniML::SetUpNetwork(int nbInput,int nbHidden,int lengthLayer,int nbOutput,bool regression){
    Network* net = new Network(nbInput,nbHidden,lengthLayer,nbOutput,regression);
    return net;
}

int MiniML::GetNetworkSize(void* network){
    Network* net = (Network*)(network);
    return  net->GetNetworkSize();
}

int MiniML::GetLayerSize(void* network,int i){
    Network* net = (Network*)(network);
    return  net->GetLayerSize(i);
}

int MiniML::GetLayerRealSize(void* network,int i){
    Network* net = (Network*)(network);
    return  net->GetLayerRealSize(i);
}

float MiniML::GetWeight(void* network,int i,int k,int j){
    Network* net = (Network*)(network);
    return net->GetWeight(i,k,j);
}

float* MiniML::GetError(void* network){
    Network* net = (Network*)network;
    return net->GetError();
}

float* MiniML::GetIter(void* network){
    Network* net = (Network*)network;
    return net->GetItr();
}

int MiniML::GetSizeError(void* network){
    Network* net = (Network*)network;
    return net->GetSizeError();
}

float* MiniML::SimulateNetwork(void* network,float* input,int n){
    Network* net = (Network*)network;
    std::vector<float> value;
    for (int i = 0; i < n; i++)
    {
        value.push_back(input[i]);
    }
    net->simulate(value);
    float* rslt= &(net->outputVector[0]);
    return rslt;
}

void MiniML::BackPropagation(void* network,float* input,int ninput,float* output,int noutput,float learningRate,float maxIteration){
    Network* net = (Network*)network;
    std::vector<std::vector<float>> inputVec;
    std::vector<std::vector<float>> outputVec;

    for (size_t i = 0; i < ninput; i++)
    {
        std::vector<float> in;
        int size = net->GetLayerSize(0)-1;
        for (int j = 0; j < size; j++)
        {
            in.push_back(input[(i*size)+j]);
        }
        inputVec.push_back(in);
    }

    for (size_t i = 0; i < noutput; i++)
    {
        std::vector<float> ot;
        int size = net->GetLayerSize(net->GetNetworkSize()-1);
        for (int j = 0; j < size; j++)
        {
            ot.push_back(output[(i*size)+j]);
        }
        outputVec.push_back(ot);
    }

    
    net->backPropagation(inputVec,outputVec,learningRate,maxIteration);
}