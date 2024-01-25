#pragma once
#include <vector>
#include <time.h> 
#include <stdexcept>
#include "../Eigen/Dense"
#include <iostream>
#include  <cmath>
class Network{
    
    std::vector<Eigen::MatrixXd> layer; 
    std::vector<Eigen::MatrixXd> delta; 
    std::vector<std::vector<Eigen::MatrixXd>> wieght; 
    std::vector<float> Error;
    std::vector<float> Iter;
    bool Regression;
    public: 
    std::vector<float> outputVector;
    
    Network(int nbInput,int nbHidden,int heightHidden,int nboutput,bool _regression):Regression(_regression){
        std::vector<float> e;
        std::vector<float> t;
        Error.push_back(1.0f);
        Iter.push_back(0.0f);

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

    float GetValue(int l,int i){ return layer[l](i,0); };
    float Network::GetWeight(int l,int i,int j);
    int GetLayerSize(int l);
    int GetLayerRealSize(int l);
    std::vector<Eigen::MatrixXd> GetLayer(){ return this->layer;};
    int GetNetworkSize(){ return layer.size();}
    void Network::SetWeight(int l,int i,int j,float v);
    float* simulate(float* input);
    float NetWorkProcess(int l,int j);
    void backPropagation(float** input,int sizeInput,float** output,float a,int max_it);

    float sigmoid(float a){ return std::tanh(a);}
    float* GetError(){return &this->Error[0];}
    float* GetItr(){return &this->Iter[0];}
    int GetSizeError(){return this->Error.size();}
};