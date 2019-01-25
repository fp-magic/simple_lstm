#ifndef __LSTM_H__
#define __LSTM_H__

#include "../source/tensor/XGlobal.h"
#include "../source/tensor/XTensor.h"
#include "../source/tensor/core/CHeader.h"
#include "../source/tensor/function/FHeader.h"
#include <string>
using namespace nts;

namespace lstm
{
#define _EXIT_(x)
#define CheckErrors(x, msg) { if(!(x)) { fprintf(stderr, "Error! calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__, msg);  _EXIT_(1); } }
#define ShowErrors(msg) { { fprintf(stderr, "Error! (%s line %d): %s\n", __FILENAME__, __LINE__, msg); _EXIT_(1); } } 

/*one simple cell*/
class lstmcell
{

    /*four states:cell, hidden, output, input*/ 
    protected:XTensor C,H,Y,X;
    protected:XTensor W,Wi,Wf,Wo,Wy,Z,Zi,Zf,Zo,tmp0,tmp1,tmp2;
    
    /*all inputs, all outputs*/
    protected:XTensor output;
    protected:XTensor* input;
    
    /**/
    public:int unitNum,embSize;

    /*initializer, may add other parameter like regularizer*/
    lstmcell(int myunitNum=32, int myembSize=32, std::string weightInitializer="rand")
    {
        embSize=myembSize;
        unitNum=myunitNum;
        InitTensor2D(&output,0,embSize);
        InitTensor2D(&W,embSize<<1,embSize);
        InitTensor2D(&Wi,embSize<<1,embSize);
        InitTensor2D(&Wf,embSize<<1,embSize);
        InitTensor2D(&Wo,embSize<<1,embSize);
        InitTensor2D(&Wy,embSize,embSize);
        InitTensor2D(&C,1,embSize);
        InitTensor2D(&H,1,embSize);
        InitTensor2D(&Y,1,embSize);
        InitTensor2D(&X,1,embSize);
        InitTensor2D(&tmp0,1,embSize);
        InitTensor2D(&tmp1,1,embSize);
        InitTensor2D(&tmp2,1,embSize<<1);
        if(weightInitializer=="zero")
        {
            _SetDataFixedFloat(&W,0.0);
            _SetDataFixedFloat(&Wi,0.0);
            _SetDataFixedFloat(&Wf,0.0);
            _SetDataFixedFloat(&Wo,0.0);
            _SetDataFixedFloat(&Wy,0.0);
        }else 
        if(weightInitializer=="rand")
        {
            _SetDataRand(&W,-1.0,1.0);
            _SetDataRand(&Wi,-1.0,1.0);
            _SetDataRand(&Wf,-1.0,1.0);
            _SetDataRand(&Wo,-1.0,1.0);
            _SetDataRand(&Wy,-1.0,1.0);
        }else
        {
            ShowErrors("Unable to find indicated weightInitializer.");
        }
        
    }

    /**/
    public:void setInput(XTensor& inputTensor)
    {
        input=&inputTensor;
    }

    /**/
    public:XTensor Recur()
    {
        XTensor tmp2;
        if(input==NULL)ShowErrors("lstm cell not set input!\n");
        for(int i=0;i<unitNum;++i)
        {
            if(i<input->dimSize[0])
                X=SelectRange(input,0,i,i+1);
            else
                _SetDataFixedFloat(&X,0.0);
            tmp1=MatrixMul(&tmp2,X_NOTRANS,&W,X_NOTRANS);
            Z=HardTanH(&tmp1);
            tmp1=MatrixMul(&tmp2,X_NOTRANS,&Wi,X_NOTRANS);
            Zi=Sigmoid(&tmp1);
            tmp1=MatrixMul(&tmp2,X_NOTRANS,&Wf,X_NOTRANS);
            Zf=Sigmoid(&tmp1);
            tmp1=Sigmoid(MatrixMul(&tmp2,X_NOTRANS,&Wo,X_NOTRANS));
            Zo=Sigmoid(&tmp1);
            tmp0=Multiply(&C,&Zf);
            tmp1=Multiply(&Zi,&Z);
            C=Sum(&tmp0,&tmp1);
            tmp1=HardTanH(&C);
            H=Multiply(&tmp1,&Zo);
            tmp1=MatrixMul(&H,&Wy);
            Y=Sigmoid(&tmp1);
            output=Merge(&Y,0);
        } 
        return  output;
    }

    public:void partClear()
    {
        InitTensor2D(&output,0,embSize);
    }
    
};

class lstmnet
{
    /**/
    protected:XTensor* input;
    protected:XTensor output;
    protected:int layerNum,batchSize,epochs;
    protected:lstmcell* layer0,*layer1,*layer2,*layer3;
    public:bool isBidirection,isShuffle;
    public:std::string biMode;

    /**/
    lstmnet(XTensor* inputTensor, lstmcell* l0, int eph=1, int batchSz=128, bool bidir=false, std::string bimd="concat", bool isSufl=false)
    {   
        input=inputTensor;
        layer0=l0;
        layerNum=1;
        epochs=eph;
        isBidirection=bidir;
        biMode=bimd;
        batchSize=batchSz;
        isShuffle=isSufl;
    }

    /**/
    lstmnet(XTensor* inputTensor, lstmcell* l0, lstmcell* l1,int eph=1, int batchSz=128, bool bidir=false, std::string bimd="concat",bool isSufl=false)
    {   
        input=inputTensor;
        layer0=l0;
        layer1=l1;
        layerNum=2;
        epochs=eph;
        isBidirection=bidir;
        biMode=bimd;
        batchSize=batchSz;
        isShuffle=isSufl;
    }

    /**/
    public:void train()
    {
        int dataSize=input->dimSize[0];
        XTensor middleInput;
        XTensor middleOutput;
        for(int epochNum=0;epochNum<epochs;epochNum++)
        for(int batchNum=0;batchNum<dataSize/batchSize;batchNum++)
        {
            if(layerNum>0)
            {
                middleInput=SelectRange(input,0,batchNum*batchSize,(batchNum+1)*batchSize-1);
                layer0->setInput(middleInput);middleOutput=layer0->Recur();
            }
            if(layerNum>1)
            {
                middleInput=middleOutput;
                layer1->setInput(middleInput);
                middleOutput=layer1->Recur();
            }
        }
    }

    /**/
    public:void test()
    {
        
    }
};

}; // namespace lstm
#endif