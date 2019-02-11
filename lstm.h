#ifndef __LSTM_H__
#define __LSTM_H__
/*
**determine if use gpu
**0:cpu 1:gpu
*/
#include "../tensor/XGlobal.h"
#include "../tensor/XTensor.h"
#include "../tensor/XUtility.h"
#include "../tensor/XDevice.h"
#include "../tensor/core/CHeader.h"
#include "../tensor/function/FHeader.h"
#include "../network/XNet.h"
#include <cmath>
#include <string>
using namespace nts;

namespace lstm
{
#define _EXIT_(x)
#define CheckErrors(x, msg)                                                                             \
    {                                                                                                   \
        if (!(x))                                                                                       \
        {                                                                                               \
            fprintf(stderr, "Error! calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__, msg); \
            _EXIT_(1);                                                                                  \
        }                                                                                               \
    }
#define ShowErrors(msg)                                                                \
    {                                                                                  \
        {                                                                              \
            fprintf(stderr, "Error! (%s line %d): %s\n", __FILENAME__, __LINE__, msg); \
            _EXIT_(1);                                                                 \
        }                                                                              \
    }

class rnncell
{
  protected:
    XTensor H, Y, X;

  protected:
    XTensor W, Wy, Wi, mid0, mid1, mid2;

    /*all inputs, all outputs*/
  protected:
    XTensor output;

    /**/
  public:
    int embSize;

    rnncell(int myembSize = 32, std::string weightInitializer = "rand")
    {
        embSize = myembSize;
        InitTensor2D(&output, 0, embSize);
        InitTensor2D(&W, embSize, embSize);
        InitTensor2D(&Wy, embSize, embSize);
        InitTensor2D(&Wi, embSize, embSize);
        InitTensor2D(&H, 1, embSize);
        InitTensor2D(&Y, 1, embSize);
        InitTensor2D(&X, 1, embSize);
        InitTensor2D(&mid0, 0, embSize);
        InitTensor2D(&mid1, 1, embSize);
        InitTensor2D(&mid2, 1, embSize);
        if (weightInitializer == "zero")
        {
            _SetDataFixedFloat(&W, 0.0);
            _SetDataFixedFloat(&Wy, 0.0);
            _SetDataFixedFloat(&Wi, 0.0);
        }
        else if (weightInitializer == "rand")
        {
            _SetDataRand(&W, 0.0, 1.0);
            _SetDataRand(&Wy, 0.0, 1.0);
            _SetDataRand(&Wi, 0.0, 1.0);
        }
        else
        {
            ShowErrors("Unable to find indicated weightInitializer.");
        }
    }

    /**/
  public:
    XTensor Recur()
    {
        char ch = '-';
        //XPRINT1(0, stderr,"1%c",ch);
        mid0 = MatrixMul(&H, X_NOTRANS, &W, X_NOTRANS);
        mid1 = MatrixMul(&X, X_NOTRANS, &Wi, X_NOTRANS);
        mid2 = Sum(&mid0, &mid1);
        H = HardTanH(&mid2);
        Y = Sigmoid(&H);
    }

    /**/
  public:
    void update(float learningRate)
    {
        char ch = '-';
        //XPRINT1(0, stderr,"80%c",ch);
        if (W.grad != NULL)
            _Sum(&W, W.grad, &W, -learningRate);
        //else XPRINT1(0, stderr,"81%c",ch);
        if (Wi.grad != NULL)
            _Sum(&Wi, Wi.grad, &Wi, -learningRate);
        if (Wy.grad != NULL)
            _Sum(&Wy, Wy.grad, &Wy, -learningRate);
    }

    /**/
  public:
    void partClear()
    {
        InitTensor2D(&output, 0, embSize);
        if (W.grad != NULL)
            _SetDataFixedFloat(W.grad, 0.0);
        if (Wi.grad != NULL)
            _SetDataFixedFloat(Wi.grad, 0.0);
        if (Wy.grad != NULL)
            _SetDataFixedFloat(Wy.grad, 0.0);
    }

    /**/
  public:
    void back(XTensor y, bool isLast = false)
    {
        if (isLast)
        {
            H.grad = new XTensor;
            Y.grad = new XTensor;
            X.grad = new XTensor;
            W.grad = new XTensor;
            Wi.grad = new XTensor;
            Wy.grad = new XTensor;
            _SetDataFixedFloat(&H, 0.0);
        }
        /*TO DO*/
    }
};

/*one simple cell*/
class lstmcell
{

    /*four states:cell, hidden, output, input*/
  protected:
    XTensor H, Y, X, C, Z, Zi, Zf, Zo, mid0, mid1, mid2;
    XTensor W, Wi, Wf, Wo, Wy;
	XList smallList;
    int embSize, devId, batchSize;
	float learningRate;

  public:
    /*initializer, may add other parameter like regularizer*/
    lstmcell(int mydevId, int myembSize, std::string weightInitializer = "rand", int batchSz=128,float lr = 0.01)
    {
        devId = mydevId;
        embSize = myembSize;
		batchSize = batchSz;
        learningRate = lr;
        InitTensor2D(&C, batchSize, embSize, X_FLOAT, devId);
        InitTensor2D(&H, batchSize, embSize, X_FLOAT, devId);
        InitTensor2D(&Y, batchSize, embSize, X_FLOAT, devId);
        InitTensor2D(&X, batchSize, embSize, X_FLOAT, devId);
        InitTensor2D(&mid0, batchSize, embSize << 1, X_FLOAT, devId);
        InitTensor2D(&mid1, batchSize, embSize << 1, X_FLOAT, devId);
        InitTensor2D(&mid2, batchSize, embSize, X_FLOAT, devId);
        _SetDataFixedFloat(&H, 0.0);
        _SetDataFixedFloat(&Y, 0.0);
        _SetDataFixedFloat(&X, 0.0);
        _SetDataFixedFloat(&C, 0.0);
        _SetDataFixedFloat(&mid2, 0.0);
        InitTensor2D(&W, embSize << 1, embSize, X_FLOAT, devId);
        InitTensor2D(&Wi, embSize << 1, embSize, X_FLOAT, devId);
        InitTensor2D(&Wf, embSize << 1, embSize, X_FLOAT, devId);
        InitTensor2D(&Wo, embSize << 1, embSize, X_FLOAT, devId);
        InitTensor2D(&Wy, embSize, embSize, X_FLOAT, devId);
        if (weightInitializer == "zero")
        {
            _SetDataFixedFloat(&W, 0.0);
            _SetDataFixedFloat(&Wi, 0.0);
            _SetDataFixedFloat(&Wf, 0.0);
            _SetDataFixedFloat(&Wo, 0.0);
            _SetDataFixedFloat(&Wy, 0.0);
        }
        else if (weightInitializer == "rand")
        {
            _SetDataRand(&W, -1.0, 1.0);
            _SetDataRand(&Wi, -1.0, 1.0);
            _SetDataRand(&Wf, -1.0, 1.0);
            _SetDataRand(&Wo, -1.0, 1.0);
            _SetDataRand(&Wy, -1.0, 1.0);
        }
        else
        {
            ShowErrors("Unable to find indicated weightInitializer.");
        }
    }

    void setX(XTensor &toX)
    {
        X = Sum(toX,mid2);
    }

    void getYcopy(XTensor &Yto)
    {
        Yto = Sum(Y, mid2);
    }

    XTensor *getYpointer()
    {
        return &Y;
    }
    /**/
    void Recur()
    {
		smallList.Clear();
        smallList.Add(&H);
        smallList.Add(&X);
        mid0 = Merge(smallList, 1);
        Z = MatrixMul(mid0, X_NOTRANS, W, X_NOTRANS);
        Z = HardTanH(Z);
        Zi = MatrixMul(mid0, X_NOTRANS, Wi, X_NOTRANS);
        Zi = Sigmoid(Zi);
        Zf = MatrixMul(mid0, X_NOTRANS, Wf, X_NOTRANS);
        Zf = Sigmoid(Zf);
        Zo = MatrixMul(mid0, X_NOTRANS, Wo, X_NOTRANS);
        Zo = Sigmoid(Zo);
        mid0 = Multiply(C, Zf);
        mid1 = Multiply(Zi, Z);
        C = Sum(mid0, mid1);
        mid0 = HardTanH(C);
        H = Multiply(mid0, Zo);
        mid0 = MatrixMul(H, X_NOTRANS, Wy, X_NOTRANS);
        Y = Sigmoid(mid0);
    }

    void update()
    {
		if (W.grad != NULL)
		{
			_Sum(&W, W.grad, &W, -learningRate);
			float wwww[64][32];
			for (int i = 0; i < 64; i++)
			{
				for (int j = 0; j < 32; j++)
					wwww[i][j]=W.grad->Get2D(i, j);
			}
		}
					
        if (Wi.grad != NULL)
            _Sum(&Wi, Wi.grad, &Wi, -learningRate);
        if (Wf.grad != NULL)
            _Sum(&Wf, Wf.grad, &Wf, -learningRate);
        if (Wo.grad != NULL)
            _Sum(&Wo, Wo.grad, &Wo, -learningRate);
        if (Wy.grad != NULL)
            _Sum(&Wy, Wy.grad, &Wy, -learningRate);
    }

    /**/
    void partClear()
    {
        if (H.grad != NULL)
            H.grad->SetZeroAll();
        if (Y.grad != NULL)
            Y.grad->SetZeroAll();
        if (X.grad != NULL)
            X.grad->SetZeroAll();
        if (C.grad != NULL)
            C.grad->SetZeroAll();
        if (Z.grad != NULL)
            Z.grad->SetZeroAll();
        if (Zi.grad != NULL)
            Zi.grad->SetZeroAll();
        if (Zf.grad != NULL)
            Zf.grad->SetZeroAll();
        if (Zo.grad != NULL)
            Zo.grad->SetZeroAll();
        if (mid0.grad != NULL)
            mid0.grad->SetZeroAll();
        if (mid1.grad != NULL)
            mid1.grad->SetZeroAll();
		if (mid2.grad != NULL)
			mid2.grad->SetZeroAll();
        if (W.grad != NULL)
            W.grad->SetZeroAll();
        if (Wi.grad != NULL)
            Wi.grad->SetZeroAll();
        if (Wf.grad != NULL)
            Wf.grad->SetZeroAll();
        if (Wo.grad != NULL)
            Wo.grad->SetZeroAll();
        if (Wy.grad != NULL)
            Wy.grad->SetZeroAll();
		H=Sum(mid2,mid2);//totally clear to zero
		Y = Sum(mid2, mid2);
		X = Sum(mid2, mid2);
		C = Sum(mid2, mid2);
    }

};

class lstmnet
{
  protected:
    XTensor **input;
    XTensor word2vec, zero;
    int layerNum, batchSize, epochs;
    lstmcell *layer0, *layer1, *layer2, *layer3;
    int dataSize,wordNum, unitNum, embSize, devId;
    XList outputList, goldList;
    XTensor *finalOutputs,*finalInputs;
    bool isBidirection, isShuffle;
	std::string biMode;

  public:
    /*one-layer, more layers to do*/
    lstmnet(bool use_gpu, int wm = 1024, int um = 32, int emb = 32, int eph = 1, int batchSz = 128, float lrRate = 0.01, std::string weightInitializer = "rand", bool bidir = false, std::string bimd = "concat", bool isSufl = false)
    {
        char ch = '-';
        devId = use_gpu - 1;
        wordNum = wm;
        embSize = emb;
        layer0 = new lstmcell(devId, emb, "rand", batchSz, lrRate);
        unitNum = um;
        layerNum = 1;
        epochs = eph;
        isBidirection = bidir;
        biMode = bimd;
        batchSize = batchSz;
        isShuffle = isSufl;
        InitTensor2D(&word2vec, wordNum, embSize, X_FLOAT,devId);
        _SetDataRand(&word2vec, -1.0, 1.0);
		finalOutputs = new XTensor[unitNum];
		for (int i = 0; i < unitNum; ++i)
			InitTensor2D(&finalOutputs[i], batchSize, embSize, X_FLOAT, devId);
		finalInputs = new XTensor[unitNum];
		for (int i = 0; i < unitNum; ++i)
			InitTensor2D(&finalInputs[i], batchSize, embSize, X_FLOAT, devId);
		InitTensor2D(&zero, batchSize, embSize, X_FLOAT, devId);
		_SetDataFixedFloat(&zero, 0.0);
    }

    XTensor Selectfrom2D(XTensor &fromTensor, int index)
    {
        char ch = '-';
        int dim[2] = {fromTensor.dimSize[0], fromTensor.dimSize[1]};
        XTensor toTensor, idxTensor;
        InitTensor1D(&idxTensor, 1, X_INT, devId);
        idxTensor.Set1DInt(index, 0);
        toTensor = Gather(fromTensor, idxTensor);
        //XPRINT1(0, stderr,"01%c",ch);
        return toTensor;
    }

	void getInput(std::string inPh, int dtSz,int minLength,int maxLength)
	{
		dataSize = dtSz;
		input = new XTensor*[dataSize / batchSize];
		for (int i = 0; i < dataSize / batchSize; ++i)
		{
			input[i] = new XTensor[unitNum];
			for (int j = 0; j < unitNum; ++j)
			{
				InitTensor2D(&input[i][j], batchSize, wordNum, X_FLOAT, devId);
				_SetDataFixedFloat(&input[i][j], 0.0);
			}
		}
		freopen(inPh.c_str(), "r", stdin);
		int batchNum=0,num=0,snum=0,token;
		while (scanf("%d", &token) != EOF)
		{
			if (num >= batchSize)
			{
				num = 0;
				batchNum++;
				if (batchNum>=dataSize/batchSize)break;
			}
			if (token == 2)
			{
				if (snum < maxLength)
				{
					for (int i = snum; i < maxLength; i++)
					{
						input[batchNum][i].Set2D(1.0,num,2);
					}
				}
				snum = 0;
				num++;
			}
			else if (snum < maxLength)
			{
				if (token >= wordNum)token = 2;
				input[batchNum][snum].Set2D(1.0, num, token);
				snum++;
			}
		}
	}
    /**/
    void train()
    {
        char ch = '-';
        XTensor middleInput,finalInput;
        XNet autoDiffer;
        float loss,avgloss;
        double startT = GetClockSec();
        for (int epochNum = 0; epochNum < epochs; epochNum++)
        {
			avgloss = 0;
            for (int batchNum = 0; batchNum < dataSize / batchSize; batchNum++)
            {
                loss = 0;
                goldList.Clear();
                outputList.Clear();
                if (layerNum == 1)
                {
                    for (int i = 0; i < unitNum; ++i)
					{
						if(finalInputs[i].grad!=NULL)finalInputs[i].grad->SetZeroAll();
                        finalInputs[i] = Sigmoid(MatrixMul(input[batchNum][i], X_NOTRANS, word2vec, X_NOTRANS));
                        layer0->setX(finalInputs[i]);
                        if (i != 0)
                        {
							//XPRINT1(0, stderr,"%.3f\n", _CrossEntropyFast(layer0->getYpointer(), &finalInputs[i]));
                            loss += _CrossEntropyFast(layer0->getYpointer(), &finalInputs[i]);
							goldList.Add(&finalInputs[i]);
                        }
						if (i < unitNum - 1)
						{
							layer0->Recur();
							layer0->getYcopy(finalOutputs[i]);
							outputList.Add(&finalOutputs[i]);
						}   
                        //XPRINT1(0, stderr,"8%c",ch);
                    }
					autoDiffer.Backward(outputList, goldList, CROSSENTROPY);
					//autoDiffer.ShowNetwork(stderr, &finalOutputs[0]);
					//XPRINT1(0, stderr,"9%c",ch);
					layer0->update();
					layer0->partClear();
                }
                else
                {
                    /*TO DO*/
                }
				avgloss += loss / ((float)(unitNum - 1));
                //XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, epoch=%d, batch=%d/%d, ppl=%.5f\n", GetClockSec() - startT, epochNum + 1, batchNum + 1, dataSize / batchSize, loss / ((float)(unitNum-1)));
            }
			XPRINT2(0,stderr,"[INFO] epoch=%d, avgloss=%.5f\n", epochNum + 1, avgloss / (dataSize / batchSize));
        }
    }

    /**/
    void test()
    {
    }

    /**/
    void dump()
    {
    }

    /**/
    void load()
    {
    }
};

}; // namespace lstm
#endif