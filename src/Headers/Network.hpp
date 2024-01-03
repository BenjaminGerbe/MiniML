#pragma once
#include <vector>
#include <time.h> 
#include <stdexcept>
class Network{
    std::vector<std::vector<float>> layer; 
    std::vector<std::vector<std::vector<float>>> wieght; 

    public: 
    Network(int nbInput,int nbCouche,int nboutput){
        std::srand(time(NULL)); // random seed
        std::vector<float> input;
        for (int i = 0; i < nbInput; i++)
        {
            float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
            input.push_back(r);
        }
        input.push_back(1); // biais
        layer.push_back(input);
        std::vector<std::vector<float>> thisLayer;
        for (int i = 0; i < nbCouche; i++)
        {
            std::vector<float> hidden;
            hidden.clear();
            thisLayer.clear();
            for (int j = 0; j < nbInput; j++)
            {
                std::vector<float> wNeurone;
                hidden.push_back(0);
                int size = GetLayerSize(i); // i is here the current layer - 1
                if(i == 0 || i == GetNetworkSize()-1){
                    size++;
                }
                
                for (int k = 0; k < size; k++)
                {
                    float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
                    wNeurone.push_back(r);
                }
                
                thisLayer.push_back(wNeurone);
            }
           
            if(i == nbCouche -1){
                hidden.push_back(1); // biais
            }
            wieght.push_back(thisLayer);
            layer.push_back(hidden);
        }
        

        thisLayer.clear();
        std::vector<float> output;
        for (int i = 0; i < nboutput; i++)
        {
            std::vector<float> wNeurone;
            output.push_back(0);
            int size = GetLayerSize(layer.size()-1)+1;
            for (int j = 0; j < size; j++)
            {
                float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
                wNeurone.push_back(r);
            }
            thisLayer.push_back(wNeurone);
        }
        wieght.push_back(thisLayer);
        layer.push_back(output);
    }
    
    Network(std::vector<float> input,  std::vector<std::vector<std::vector<float>>> w,int heidden,int output):Network(input.size(),heidden,output){
        for (int i = 0; i < input.size(); i++)
        {
            layer[0][i] = input[i];
        }

        for (int i = 0; i < w.size(); i++)
        {
            for (int j = 0; j < w[i].size(); j++)
            {
                for (int k = 0; k < w[i][j].size(); k++)
                {
                    this->wieght[i][j][k] = w[i][j][k];
                }
            }
        }
        
    }

    float GetValue(int l,int i){ return layer[l][i]; };
    float Network::GetWeight(int l,int i,int j){
        int idx = l-1; // beceause the layer of the input is not added
        if(idx < 0 || idx >= GetNetworkSize()){
            return 0.0f;
        }
        return wieght[idx][i][j];
    }

    
    int GetLayerSize(int l){
        if( l < 0 || l >= layer.size()) return 0;
        return l == 0 || l == layer.size()-2 ?  layer[l].size() -1 : layer[l].size(); 
    }
    std::vector<std::vector<float>> GetLayer(){ return this->layer;};

    int GetNetworkSize(){ return layer.size();}
    void Network::SetWeight(int l,int i,int j,float v){
        int idx = l-1; // beceause the layer of the input is not added
        if(idx < 0){
            throw std::invalid_argument( "the value of l can't be 0 beceause the input 0 layer is the input and input don't recieve wieght from precedent layer");
        }
        wieght[idx][i][j] = v;
    }
    
};