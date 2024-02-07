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
    net->simulate(input);
    float* rslt= &(net->outputVector[0]);
    return rslt;
}

float* MiniML::RBFSimulate(void* network,float* input,int n,float a){
    Network* net = (Network*)network;
    net->SimulateRBF(input,n,a);
    if(net->outputVector.size() == 0 ) net->outputVector.push_back(0);
    float* rslt= &(net->outputVector[0]);
    return rslt;
}

void MiniML::BackPropagation(void* network,float** input,int ninput,float** output,int noutput,float learningRate,float maxIteration){
    Network* net = (Network*)network;
    net->backPropagation(input,ninput,output,learningRate,maxIteration);
}

void MiniML::LinearPropagation(void* network,float** input,int ninput,float** output,int noutput,float learningRate,float maxIteration){
    Network* net = (Network*)network;
    net->linearPropagation(input,ninput,output,learningRate,maxIteration);
}

void MiniML::RBFPropagation(void* network,float** input,int ninput,int sizeInput,float** output,int noutput,float learningRate,float maxIteration){
    Network* net = (Network*)network;
    net->RBFPropagation(input,ninput,sizeInput,output,learningRate,maxIteration);
}
