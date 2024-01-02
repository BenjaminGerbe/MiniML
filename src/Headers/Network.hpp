#pragma once
#include <vector>
#include <time.h> 

class Network{
    std::vector<std::vector<float>> layer; 
    //std::vector<std::vector<float>> wieght; 

    public: 
    Network(int nbInput,int nbCouche,int nboutput){
        std::srand(time(NULL));
        std::vector<float> input;
        for (int i = 0; i < nbInput; i++)
        {
            float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
            input.push_back(r);
        }
        layer.push_back(input);
        for (int i = 0; i < nbCouche; i++)
        {
            std::vector<float> hidden;
            hidden.clear();
            for (int i = 0; i < nbInput; i++)
            {
                float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
                hidden.push_back(r);
            }
            layer.push_back(hidden);
        }

        std::vector<float> output;
        for (int i = 0; i < nboutput; i++)
        {
            float r = (((double) rand() / (RAND_MAX))*2.0f)-1.0f;
            output.push_back(r);
        }

        layer.push_back(output);

    }

    float GetValue(int l,int i){
        return layer[l][i];
    }

};