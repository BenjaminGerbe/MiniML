#pragma once
#ifdef  MINIML_API_EXPORTS
#define  MINIML_API __declspec(dllexport)
#else
#define  MINIML_API __declspec(dllimport)
#endif

namespace MiniML{

    extern "C" {
        MINIML_API void* SetUpNetwork(int i,int y,int h,int z,bool b);
        MINIML_API int GetNetworkSize(void* network);
        MINIML_API int GetLayerSize(void* network,int i);
        MINIML_API int GetLayerRealSize(void* network,int i);
        MINIML_API float GetWeight(void* network,int i,int k,int j);
        MINIML_API float* SimulateNetwork(void* network,float* input,int n);
        MINIML_API float* GetError(void* network);
        MINIML_API float* GetIter(void* network);
        MINIML_API int GetSizeError(void* network);
        MINIML_API void BackPropagation(void* network,float** input,int ninput,float** output,int noutput,float learningRate,float maxIteration);
        MINIML_API void LinearPropagation(void* network,float** input,int ninput,float** output,int noutput,float learningRate,float maxIteration);
    } 
}   