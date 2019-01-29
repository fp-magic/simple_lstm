#ifndef __LSTM_H__
#define __LSTM_H__
/*
**determine if use gpu
**0:cpu 1:gpu
*/
//#define USE_CUDA
#include "../source/tensor/XGlobal.h"
#include "../source/tensor/XTensor.h"
#include "../source/tensor/XUtility.h"
#include "../source/tensor/XDevice.h"
#include "../source/tensor/core/CHeader.h"
#include "../source/tensor/function/FHeader.h"
#include "../source/network/XNet.h"
#include <cmath>
#include <string>
using namespace nts;

namespace lstm
{
#define _EXIT_(x)
#define CheckErrors(x, msg) { if(!(x)) { fprintf(stderr, "Error! calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__, msg);  _EXIT_(1); } }
#define ShowErrors(msg) { { fprintf(stderr, "Error! (%s line %d): %s\n", __FILENAME__, __LINE__, msg); _EXIT_(1); } } 

class rnncell
{
    protected:XTensor H,Y,X;
    protected:XTensor W,Wy,Wi,mid0,mid1,mid2;

    /*all inputs, all outputs*/
    protected:XTensor output;

    /**/
    public:int embSize;

    rnncell( int myembSize=32, std::string weightInitializer="rand")
    {
        embSize=myembSize;
        InitTensor2D(&output,0,embSize);
        InitTensor2D(&W,embSize,embSize);
        InitTensor2D(&Wy,embSize,embSize);
        InitTensor2D(&Wi,embSize,embSize);
        InitTensor2D(&H,1,embSize);
        InitTensor2D(&Y,1,embSize);
        InitTensor2D(&X,1,embSize);
        InitTensor2D(&mid0,0,embSize);
        InitTensor2D(&mid1,1,embSize);
        InitTensor2D(&mid2,1,embSize);
        if(weightInitializer=="zero")
        {
            _SetDataFixedFloat(&W,0.0);
            _SetDataFixedFloat(&Wy,0.0);
            _SetDataFixedFloat(&Wi,0.0);
        }else 
        if(weightInitializer=="rand")
        {
            _SetDataRand(&W,-1.0,1.0);
            _SetDataRand(&Wy,-1.0,1.0);
            _SetDataRand(&Wi,-1.0,1.0);
        }else
        {
            ShowErrors("Unable to find indicated weightInitializer.");
        }  
    }

    /**/
    public:XTensor Recur()
    {
        char ch='-';
        //XPRINT1(0, stderr,"1%c",ch);
        mid0=MatrixMul(&H,X_NOTRANS,&W,X_NOTRANS);
        mid1=MatrixMul(&X,X_NOTRANS,&Wi,X_NOTRANS);
        mid2=Sum(&mid0,&mid1);
        H=HardTanH(&mid2);
        Y=Sigmoid(&H);
    }

    /**/
    public:void update(float learningRate)
    {
        char ch='-';
        //XPRINT1(0, stderr,"80%c",ch);
        if(W.grad!=NULL)_Sum(&W,W.grad,&W,-learningRate);
        //else XPRINT1(0, stderr,"81%c",ch);
        if(Wi.grad!=NULL)_Sum(&Wi,Wi.grad,&Wi,-learningRate);
        if(Wy.grad!=NULL)_Sum(&Wy,Wy.grad,&Wy,-learningRate);
    }
    
    /**/
    public:void partClear()
    {
        InitTensor2D(&output,0,embSize);
        if(W.grad!=NULL)_SetDataFixedFloat(W.grad,0.0);
        if(Wi.grad!=NULL)_SetDataFixedFloat(Wi.grad,0.0);
        if(Wy.grad!=NULL)_SetDataFixedFloat(Wy.grad,0.0);
    }

    /**/
    public:void back(XTensor y,bool isLast=false)
    {
        if(isLast)
        {
            H.grad=new XTensor;
            Y.grad=new XTensor;
            X.grad=new XTensor;
            W.grad=new XTensor;
            Wi.grad=new XTensor;
            Wy.grad=new XTensor;
            _SetDataFixedFloat(&H,0.0);
        }
        /*TO DO*/
    }
};

/*one simple cell*/
class lstmcell
{

    /*four states:cell, hidden, output, input*/ 
    public:XTensor preH,H,Y,X,C;
    protected:XTensor W,Wi,Wf,Wo,Wy,Z,Zi,Zf,Zo,mid0,mid1,mid2,mid3,mid4,mid00,mid01,mid02,mid03;
    public:int embSize,devId;
    
    /*initializer, may add other parameter like regularizer*/
    lstmcell(int mydevId, int myembSize=32, std::string weightInitializer="rand")
    {
        devId=mydevId;
        embSize=myembSize;
        InitTensor2D(&W,embSize<<1,embSize,X_FLOAT,devId);
        InitTensor2D(&Wi,embSize<<1,embSize,X_FLOAT,devId);
        InitTensor2D(&Wf,embSize<<1,embSize,X_FLOAT,devId);
        InitTensor2D(&Wo,embSize<<1,embSize,X_FLOAT,devId);
        InitTensor2D(&Wy,embSize,embSize,X_FLOAT,devId);
        InitTensor2D(&preH,1,embSize,X_FLOAT,devId);
        InitTensor2D(&C,1,embSize,X_FLOAT,devId);
        InitTensor2D(&H,1,embSize,X_FLOAT,devId);
        InitTensor2D(&Y,1,embSize,X_FLOAT,devId);
        InitTensor2D(&X,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid0,1,embSize<<1,X_FLOAT,devId);
        InitTensor2D(&mid00,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid01,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid02,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid03,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid1,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid2,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid3,1,embSize,X_FLOAT,devId);
        InitTensor2D(&mid4,1,embSize,X_FLOAT,devId);
        _SetDataFixedFloat(&H,0.0);
        _SetDataFixedFloat(&Y,0.0);
        _SetDataFixedFloat(&X,0.0);
        _SetDataFixedFloat(&C,0.0);
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
    public:void Recur()
    {
        char ch='-';
        XList * smallList = new XList();
        smallList->Add(&preH);
        smallList->Add(&X);
        XPRINT1(0, stderr,"1%c",ch);
        mid0=Merge(*smallList,1);
        mid00=MatrixMul(mid0,X_NOTRANS,W,X_NOTRANS);
        Z=HardTanH(mid00);
        //XPRINT1(0, stderr,"2%c",ch);
        mid01=MatrixMul(mid0,X_NOTRANS,Wi,X_NOTRANS);
        Zi=Sigmoid(mid01);
        //XPRINT1(0, stderr,"3%c",ch);
        mid02=MatrixMul(mid0,X_NOTRANS,Wf,X_NOTRANS);
        Zf=Sigmoid(mid02);
        //XPRINT1(0, stderr,"4%c",ch);
        mid03=MatrixMul(mid0,X_NOTRANS,Wo,X_NOTRANS);
        Zo=Sigmoid(mid03);
        //XPRINT1(0, stderr,"5%c",ch);
        mid1=Multiply(C,Zf);
        mid2=Multiply(Zi,Z);
        C=Sum(mid1,mid2);
        mid3=HardTanH(C);
        //XPRINT1(0, stderr,"6%c",ch);
        H=Multiply(mid3,Zo);
        //XPRINT1(0, stderr,"61%c",ch);
        mid4=MatrixMul(H,Wy);
        //XPRINT1(0, stderr,"62%c",ch);
        Y=Sigmoid(mid4);
        //XPRINT1(0, stderr,"7%c",ch);
    }

    /**/
    public:void update(float learningRate)
    {
        char ch='-';
        //XPRINT1(0, stderr,"80%c",ch);
        if(W.grad!=NULL)_Sum(&W,W.grad,&W,-learningRate);
        //else XPRINT1(0, stderr,"81%c",ch);
        if(Wi.grad!=NULL)_Sum(&Wi,Wi.grad,&Wi,-learningRate);
        if(Wf.grad!=NULL)_Sum(&Wf,Wf.grad,&Wf,-learningRate);
        if(Wo.grad!=NULL)_Sum(&Wo,Wo.grad,&Wo,-learningRate);
        //XPRINT1(0, stderr,"82%c",ch);
        if(Wy.grad!=NULL)_Sum(&Wy,Wy.grad,&Wy,-learningRate);
    }
    
    /**/
    public:void partClear()
    {
        if(preH.grad!=NULL)preH.grad->SetZeroAll();
        if(H.grad!=NULL)H.grad->SetZeroAll();
        if(Y.grad!=NULL)Y.grad->SetZeroAll();
        if(X.grad!=NULL)X.grad->SetZeroAll();
        if(C.grad!=NULL)C.grad->SetZeroAll();
        if(W.grad!=NULL)W.grad->SetZeroAll();
        if(Wi.grad!=NULL)Wi.grad->SetZeroAll();
        if(Wf.grad!=NULL)Wf.grad->SetZeroAll();
        if(Wo.grad!=NULL)Wo.grad->SetZeroAll();
        if(Wy.grad!=NULL)Wy.grad->SetZeroAll();
        if(Z.grad!=NULL)Z.grad->SetZeroAll();
        if(Zi.grad!=NULL)Zi.grad->SetZeroAll();
        if(Zf.grad!=NULL)Zf.grad->SetZeroAll();
        if(Zo.grad!=NULL)Zo.grad->SetZeroAll();
        if(mid0.grad!=NULL)mid0.grad->SetZeroAll();
        if(mid1.grad!=NULL)mid1.grad->SetZeroAll();
        if(mid2.grad!=NULL)mid2.grad->SetZeroAll();
        if(mid3.grad!=NULL)mid3.grad->SetZeroAll();
        if(mid4.grad!=NULL)mid4.grad->SetZeroAll();
        if(mid00.grad!=NULL)mid00.grad->SetZeroAll();
        if(mid01.grad!=NULL)mid01.grad->SetZeroAll();
        if(mid02.grad!=NULL)mid02.grad->SetZeroAll();
        if(mid03.grad!=NULL)mid03.grad->SetZeroAll();
    }

    /**/
    public:void back()
    {

    }
    
};

class lstmnet
{
    /*
    *input:sentence num * sentece length * embedding size
    *output:sentence num * embedding size
    */
    protected:XTensor* input,*gold;
    protected:XTensor output;
    protected:int layerNum,batchSize,epochs;
    protected:float learningRate;
    protected:lstmcell** layer0,*layer1,*layer2,*layer3;
    protected:int unitNum,embSize,devId;
    protected:XList outputList,goldList;
    public:bool isBidirection,isShuffle;
    public:std::string biMode;

    /*one-layer, more layers to do*/
    lstmnet(bool use_gpu, XTensor* inputTensor, int um=32,int emb=32,int eph=1, int batchSz=128, float lrRate=0.01,bool bidir=false, std::string bimd="concat", bool isSufl=false)
    {  
        char ch='-';
        XPRINT1(0, stderr,"000%c",ch);
        devId=use_gpu-1;
        input=inputTensor;
        //gold=goldTensor;
        layer0=new lstmcell*[um];
        for(int i=0;i<um;i++)layer0[i]=new lstmcell(devId, emb, "rand");
        unitNum=um;
        embSize=emb;
        layerNum=1;
        epochs=eph;
        isBidirection=bidir;
        biMode=bimd;
        batchSize=batchSz;
        learningRate=lrRate;
        isShuffle=isSufl;
        XPRINT1(0, stderr,"001%c",ch);
    }

    public:XTensor Selectfrom3D(XTensor& fromTensor,int index)
    {
        char ch='-';
        int dim[3]={fromTensor.dimSize[0],fromTensor.dimSize[1],fromTensor.dimSize[2]};
        int mdim[2]={dim[0],dim[1]*dim[2]};
        int ndim[3]={1,dim[1],dim[2]};
        XTensor toTensor,idxTensor;
        toTensor.SetTMPFlag();
        idxTensor.SetTMPFlag();
        InitTensor2D(&toTensor,1,dim[1]*dim[2],X_FLOAT,devId);
        InitTensor1D(&idxTensor,1,X_INT,devId);
        idxTensor.Set1DInt(index,0);
        fromTensor=Reshape(fromTensor,2,mdim);
        toTensor=Gather(fromTensor,idxTensor);
        fromTensor=Reshape(fromTensor,3,dim);
        toTensor=Reshape(toTensor,3,ndim);
        XPRINT1(0, stderr,"00%c",ch);
        return toTensor;
    }

    public:XTensor Selectfrom2D(XTensor& fromTensor,int index)
    {
        char ch='-';
        int dim[2]={fromTensor.dimSize[0],fromTensor.dimSize[1]};
        XTensor toTensor,idxTensor;
        //InitTensor2D(&toTensor,1,fromTensor.dimSize[1]); 
        InitTensor1D(&idxTensor,1,X_INT,devId);
        idxTensor.Set1DInt(index,0);
        toTensor=Gather(fromTensor,idxTensor);      
        //XPRINT1(0, stderr,"01%c",ch);
        return toTensor;
    }

    /**/
    public:void train()
    {
        char ch='-';
        int dataSize=input->dimSize[0];
        XTensor middleInput,middleOutput,middleGold;
        XTensor tmp;
        XPRINT1(0, stderr,"002%c",ch);
        InitTensor2D(&tmp,1,embSize,X_FLOAT,devId);
        _SetDataFixedFloat(&tmp,0.0);
        XNet autoDiffer;
        XPRINT1(0, stderr,"003%c",ch);
        float loss;
        double startT = GetClockSec();
        for(int epochNum=0;epochNum<epochs;epochNum++)
        {
            loss=0;
            for(int batchNum=0;batchNum<dataSize/batchSize;batchNum++)
            {
                if(layerNum==1)
                {
                    for(int i=0;i<batchSize;++i)
                    {
                        middleInput=ReduceSum(Selectfrom3D(*input,batchNum*batchSize+i),0);
                        for(int j=0;j<unitNum;j++)
                        {
                            layer0[j]->X=Selectfrom2D(middleInput,j);
                            //XPRINT1(0, stderr,"02%c",ch);
                            if(j!=0)
                            {
                                layer0[j]->preH=Sum(layer0[j-1]->H,tmp);
                                loss+=_CrossEntropyFast(&layer0[j-1]->Y,&layer0[j]->X);
                                goldList.Add(&layer0[j]->X);
                            }else
                                _SetDataFixedFloat(&layer0[j]->preH,0.0);
                            //XPRINT1(0, stderr,"03%c",ch);
                            layer0[j]->Recur();
                        }
                        //XPRINT1(0, stderr,"8%c",ch);
                        //autoDiffer.ShowNetwork(stderr, &layer0[0]->Y);
                        //exit(1);
                        autoDiffer.Backward(outputList,goldList,CROSSENTROPY);
                        //XPRINT1(0, stderr,"9%c",ch);
                        for(int j=unitNum-1;j>=0;j--)
                        {
                            layer0[j]->update(learningRate);
                            layer0[j]->partClear();
                        }
                    }
                }else
                {
                    /*TO DO*/
                }
                XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, epoch=%d, batch=%d/%d, ppl=%.3f\n",GetClockSec() - startT, epochNum + 1, batchNum+1, dataSize/batchSize,exp(loss / (batchSize*batchNum*unitNum)));
            }
        }
    }

    /**/
    public:void test()
    {
        
    }

    /**/
    public:void dump()
    {

    }

    /**/
    public:void load()
    {

    }
};

}; // namespace lstm
#endif