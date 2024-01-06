#pragma once
#include <vector>
#include <time.h> 
#include <stdexcept>
#include "Eigen/Dense"
#include <iostream>
#include  <cmath>
class Network{
    std::vector<Eigen::MatrixXd> layer; 
    std::vector<Eigen::MatrixXd> delta; 
    std::vector<std::vector<Eigen::MatrixXd>> wieght; 
    std::vector<std::vector<float>> Error;
    bool Regression;
    public: 
    Network(int nbInput,int nbHidden,int heightHidden,int nboutput,bool _regression):Regression(_regression){
        std::vector<float> e;
        std::vector<float> t;
        t.push_back(0);
        e.push_back(1);
        Error.push_back(e);
        Error.push_back(t);

        std::srand(time(NULL)); // random seed
        Eigen::MatrixXd input(nbInput+1,1);
        for (int i = 0; i < nbInput; i++){
            input(i,0) = (0);
        }

        input(nbInput,0) = 1; // biais
        layer.push_back(input);

        float vs =((float)nbHidden)/(float)heightHidden;
        int nbCouche = std::ceil(vs);
        int p = nbHidden;
        for (int i = 0; i < nbCouche; i++)
        {
            int n = p;
            if (p > heightHidden)
            {
                n = heightHidden;
            }

            Eigen::MatrixXd hidden(n+ (i== nbCouche-1 ? 1 : 0),1);//add one if he need bias
            std::vector<Eigen::MatrixXd> layerweight;
            for (int j = 0; j < n; j++)
            {
                hidden(j,0) = 0;
                Eigen::MatrixXd thisLayer(1,GetLayerSize(i)  );
     
                for (int k = 0; k < GetLayerSize(i); k++)
                {
                    float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
                    thisLayer(0,k) = r;
                }

                layerweight.push_back(thisLayer);
            }
            p -= n;
            if(i == nbCouche -1){
                hidden(n,0) = 1;
            }
            wieght.push_back(layerweight);
            layer.push_back(hidden);
        }
        

        Eigen::MatrixXd thisLayer(nboutput,GetLayerSize(layer.size()-1));
        std::vector<Eigen::MatrixXd> ouputVec;
        Eigen::MatrixXd output(nboutput,1);
        for (int i = 0; i < nboutput; i++)
        {
            output(i,0) = 0;
            for (int j = 0; j < GetLayerSize(layer.size()-1); j++)
            {
                float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
                thisLayer(i,j) = r;
            }
            ouputVec.push_back(thisLayer);
        }

        wieght.push_back(ouputVec);
        layer.push_back(output);

        // initalize delta
        for (int i = 0; i < layer.size(); i++)
        {
            Eigen::MatrixXd m(GetLayerSize(i),1);
            for (int j = 0; j < GetLayerSize(i); j++)
            {
                m(j,0) = 0;
            }
            std::cout << m << std::endl;
            delta.push_back(m);
        }
    }
    
    Network(std::vector<float> input,  std::vector<std::vector<std::vector<float>>> w,int heidden,int heightHidden,int output,bool _regression):Network(input.size(),heidden,heightHidden,output,_regression){
        // for (int i = 0; i < input.size(); i++)
        // {
        //     layer[0][i] = input[i];
        // }

        // for (int i = 0; i < w.size(); i++)
        // {
        //     for (int j = 0; j < w[i].size(); j++)
        //     {
        //         for (int k = 0; k < w[i][j].size(); k++)
        //         {
        //             this->wieght[i][j][k] = w[i][j][k];
        //         }
        //     }
        // }
        
    }

    float GetValue(int l,int i){ return layer[l](i,0); };
    float Network::GetWeight(int l,int i,int j){
        int idx = l-1; // beceause the layer of the input is not added
        if(idx < 0 || idx >= GetNetworkSize()){
            return 0.0f;
        }
        return wieght[idx][i](0,j);
    }

    
    int GetLayerSize(int l){
        if( l < 0 || l >= layer.size()) return 0;
        return layer[l].rows(); 
    }

    int GetLayerRealSize(int l){
        if( l < 0 || l >= layer.size()) return 0;
        return l==0 || l == GetNetworkSize()-2 ? layer[l].rows() -1 : layer[l].rows(); 
    }
    std::vector<Eigen::MatrixXd> GetLayer(){ return this->layer;};

    int GetNetworkSize(){ return layer.size();}
    void Network::SetWeight(int l,int i,int j,float v){
        int idx = l-1; // beceause the layer of the input is not added
        if(idx < 0){
            throw std::invalid_argument( "the value of l can't be 0 beceause the input 0 layer is the input and input don't recieve wieght from precedent layer");
        }
        wieght[idx][i](0,j)= v;
    }
    
    std::vector<float> simulate(std::vector<float> input){

        std::vector<float> v;
        int l = GetNetworkSize();
        for (int i = 0; i < GetLayerRealSize(0); i++)
        {
            layer[0](i,0) = input[i];
        }

        NetWorkProcess(l-1,0);
        for (int i = 0; i < GetLayerSize(l-1); i++)
        {
            if(Regression){
                layer[l-1](i,0) =(this->wieght[l-2][i]*layer[l-2])(0,0);
            }
            else{
                layer[l-1](i,0) = sigmoid((this->wieght[l-2][i]*layer[l-2])(0,0));
            }
            v.push_back(layer[l-1](i,0));
        }

        return v;
    }

    float sigmoid(float a){
        return std::tanh(a);
    }

    float NetWorkProcess(int l,int j){
        float a = 0.0;

        if(l-1 > 0){
            for (int d = 0; d < GetLayerRealSize(l-1); d++)
            {
                layer[l-1](d,0) = NetWorkProcess(l-1,d);
            }
        }
        return sigmoid((this->wieght[l-1][j]*layer[l-1])(0,0));
    }

    void backPropagation(std::vector<std::vector<float>> input,std::vector<std::vector<float>> output,float a,int max_it){
    
        float error = 0.0;
            
        for (int it = 0; it < max_it; it++)
        {
            int idx = rand() % input.size();
            simulate(input[idx]);
            float err=0.0f;
            int w  = GetLayerRealSize(GetNetworkSize()-1);
            for (int i = 0; i <w; i++)
            {
                float value = layer[GetNetworkSize()-1](i,0);
                float b=1.0;
                if(!Regression){
                    b = (1-(value*value));
                }
                float d = b*(value - output[idx][i%output[idx].size()]);
                this->delta[GetNetworkSize()-1](i,0) = d;
                err+=std::abs(value - output[idx][i%output[idx].size()]);
            }
            error+= err/GetLayerRealSize(GetNetworkSize()-1);
            for (int i = GetNetworkSize()-2; i >= 0; i--)
            {
                for (int j = 0; j < GetLayerRealSize(i); j++)
                {
                    float a = 0.0;
                    for (int k = 0; k < GetLayerRealSize(i+1); k++)
                    {
                        a += GetWeight(i+1,k,j)*delta[i+1](k,0);
                    }
                    delta[i](j,0) = (1-(layer[i](j,0)*layer[i](j,0)))*a;
                }
                
            }
            
            for (int i = 0; i < GetNetworkSize()-1; i++)
            {
                for (int j = 0; j < GetLayerSize(i); j++)
                {
                    for (int k = 0; k < GetLayerRealSize(i+1); k++)
                    {
                        float v = GetWeight(i+1,k,j) - a*layer[i](j,0)*delta[i+1](k,0);
                        SetWeight(i+1,k,j,v);
                    }
                    
                }
                
            }
        }

        Error[0].push_back(error/max_it);
        float t = Error[1][Error[1].size()-1]+1;
        Error[1].push_back(t);

    }

    std::vector<std::vector<float>> GetError(){
        return this->Error;
    }
};