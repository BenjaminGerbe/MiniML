#include "../Headers/Network.hpp"
#include <algorithm>
float Network::GetWeight(int l,int i,int j){
    int idx = l-1; // beceause the layer of the input is not added
    if(idx < 0 || idx >= GetNetworkSize()){
        return 0.0f;
    }
    return wieght[idx][i](0,j);
}

int Network::GetLayerSize(int l){
    if( l < 0 || l >= layer.size()) return 0;
    return layer[l].rows(); 
}

int Network::GetLayerRealSize(int l){
    if( l < 0 || l >= layer.size()) return 0;
    return l < GetNetworkSize()-1 ? layer[l].rows() -1 : layer[l].rows(); 
}

void Network::SetWeight(int l,int i,int j,float v){
    int idx = l-1; // beceause the layer of the input is not added
    if(idx < 0){
        throw std::invalid_argument( "the value of l can't be 0 beceause the input 0 layer is the input and input don't recieve wieght from precedent layer");
    }
    wieght[idx][i](0,j)= v;
}

float* Network::simulate(float* input){

    outputVector.clear();
    int l = GetNetworkSize();
    for (int i = 0; i < GetLayerRealSize(0); i++)
    {   
        layer[0](i,0) = input[i];
    }

    NetWorkProcess(l-1,0);
    for (int i = 0; i < GetLayerSize(l-1); i++)
    {
        if(Regression){
                
            float min = std::numeric_limits<float>::min();
            float max = std::numeric_limits<float>::max();
            float v = (this->wieght[l-2][i]*layer[l-2])(0,0);
            layer[l-1](i,0) = std::clamp(v,min,max);
        }
        else{
            layer[l-1](i,0) = sigmoid((this->wieght[l-2][i]*layer[l-2])(0,0));
        }
        outputVector.push_back(layer[l-1](i,0));
    }
    return &outputVector[0];
}

float Network::NetWorkProcess(int l,int j){
    float a = 0.0;

    if(l-1 > 0){
        NetWorkProcess(l-1,0);
        for (int d = 0; d < GetLayerRealSize(l-1); d++)
        {
            layer[l-1](d,0) = sigmoid((this->wieght[l-2][d]*layer[l-2])(0,0));
        }
    }
    return sigmoid((this->wieght[l-1][j]*layer[l-1])(0,0));
}

void Network::backPropagation(float** input,int sizeInput,float** output,float a,int max_it){
    
    float error = 0.0;
    for (int it = 0; it < max_it; it++)
    {
        int idx = rand() % sizeInput;
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
            float d = b*(value - output[idx][i]);
            this->delta[GetNetworkSize()-1](i,0) = d;
            err+=std::abs(value - output[idx][i]);
        }
        error+= err/(float)w;
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

    Error.push_back(error/max_it);
    float t = Iter[Iter.size()-1]+1;
    Iter.push_back(t);
}

void Network::linearPropagation(float** input,int sizeInput,float** output,float a,float max_it){
    if(this->GetNetworkSize() > 2) return;

    float error = 0.0f;
    if(!this->Regression){
        for (int i = 0; i < max_it; i++)
        {
            int idx = rand() % sizeInput;
            simulate(input[idx]);
            int nbInput = this->GetLayerSize(0);
            Eigen::MatrixXd X(1,nbInput);
            for (int j = 0; j < nbInput-1; j++)
            {
                X(0,j) = input[idx][j];
            }
            X(0,nbInput-1) = 1;
            int w  = GetLayerRealSize(GetNetworkSize()-1);  
            float err;
            for (int k = 0; k < w; k++)
            {
                float value = sigmoid( layer[GetNetworkSize()-1](k,0));
                err= output[idx][k]-value;
                wieght[0][k] = wieght[0][k]+(a*(output[idx][k]-value)*X); 
            }

            error += err/(float)w;
            
        }
    }
    else{

        int nbInput = this->GetLayerSize(0);
        int w  = GetLayerRealSize(GetNetworkSize()-1);       
        Eigen::MatrixXd X(sizeInput,nbInput);
        Eigen::MatrixXd Y(sizeInput,w);
        for (int i = 0; i < sizeInput; i++)
        {
            X(i,0) = 1;
            for (int j = 1; j < nbInput; j++)
            {
                X(i,j) = input[i][j-1];
            }
            
            for (int j = 0; j < w; j++)
            {
                Y(i,j) = output[i][j];
            }

        }

        int idx = rand() % sizeInput;
        Eigen::MatrixXd tX = X.transpose();
        Eigen::MatrixXd tX_X = tX * X;
        if(!tX_X.completeOrthogonalDecomposition().isInvertible()){
            std::cout << "the matrix tX is not invertible" << std::endl;
            return;
        }
        Eigen::MatrixXd W = ((tX * X).inverse() * tX) * Y;
        for (int i = 0; i < nbInput; i++)
        {
            wieght[0][0](0,i) = W((nbInput-1)-i,0);
        }

        simulate(input[idx]);
        
    }

}

void Network::SimulateRBF(float* input,int size,float a){
    int nbInput = size;
    Eigen::MatrixXd X(size,1);
    for (int i = 0; i < size; i++)
    {
        X(i,0) = input[i];
    }

    outputVector.clear();
    for (int i = 0; i < exempleParameter.size(); i++)
    {
        float v = std::exp(-a*(std::pow((X-exempleParameter[i]).norm(),2)));
        outputVector.push_back(v);
    }
    
    simulate(&outputVector[0]);
}

void Network::RBFPropagation(float** input,int sizeInput,int fLayerLength,float** output,float a,float max_it){
    if(this->GetNetworkSize() > 2) return;

    float error = 0.0f;
    if(!this->Regression){

        // int nbInput = this->GetLayerSize(0);
        // int w  = GetLayerRealSize(GetNetworkSize()-1);       
        // std::vector<Eigen::MatrixXd> X;
        // Eigen::MatrixXd Y(sizeInput,w);
        // for (int i = 0; i < sizeInput; i++)
        // {
        //     Eigen::MatrixXd Xtmp(sizeInput,nbInput);
        //     Xtmp(0,0) = 1;
        //     for (int j = 1; j < nbInput; j++)
        //     {
        //         Xtmp(0,j) = input[i][j-1];
        //     }
        //     X.push_back(Xtmp);

        //     for (int j = 0; j < w; j++)
        //     {
        //         Y(i,j) = output[i][j];
        //     }

        // }
        

//        float output;


    }
    else{

        int nbInput = fLayerLength;
        int w  = GetLayerRealSize(GetNetworkSize()-1);       
        std::vector<Eigen::MatrixXd> X;
        Eigen::MatrixXd Y(sizeInput,w);
        for (int i = 0; i < sizeInput; i++)
        {
            Eigen::MatrixXd Xtmp(sizeInput,nbInput);
            Xtmp(0,0) = 1;
            for (int j = 1; j < nbInput; j++)
            {
                Xtmp(0,j) = input[i][j-1];
            }
            X.push_back(Xtmp);

            for (int j = 0; j < w; j++)
            {
                Y(i,j) = output[i][j];
            }

        }

        Eigen::MatrixXd phi(X.size(),X.size());
        for (int i = 0; i < X.size(); i++)
        {
            for (int j = 0; j < X.size(); j++)
            {
                double delta = std::powl((X[i] - X[j]).norm(),2);
                phi(i,j) = std::exp(-a*delta);
            }   
        }

        int idx = rand() % sizeInput;

        Eigen::MatrixXd W = phi.inverse()*Y;
        for (int i = 0; i < nbInput; i++)
        {
            wieght[0][0](0,i) = W(i,0);
        }

        SimulateRBF(input[idx],sizeInput,a);
    }

}