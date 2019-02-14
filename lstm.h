#ifndef __LSTM_H__
#define __LSTM_H__
#include "../tensor/XGlobal.h"
#include "../tensor/XTensor.h"
#include "../tensor/XUtility.h"
#include "../tensor/XDevice.h"
#include "../tensor/core/CHeader.h"
#include "../tensor/function/FHeader.h"
#include "../network/XNet.h"
#include <cmath>
#include <string>
#include<algorithm>
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

/*one simple lstm cell*/
class lstmcell
{

    /*four states:cell, hidden, output, input*/
  protected:
    XTensor H, Y, X, C, Z, Zi, Zf, Zo, mid0, mid10,mid11, mid2,mid3;
    XTensor W, Wi, Wf, Wo, Wy;
	XList smallList;
    int embSize, unitNum, devId, batchSize;
	float learningRate;

  public:
    /*initializer with parameters*/
    lstmcell(int mydevId, int myembSize, int myUnitnum,std::string weightInitializer = "rand", int batchSz=128,float lr = 0.01)
    {
        devId = mydevId;
        embSize = myembSize;
		unitNum = myUnitnum;
		batchSize = batchSz;
        learningRate = lr;
        InitTensor2D(&C, batchSize, unitNum, X_FLOAT, devId);
        InitTensor2D(&H, batchSize, unitNum, X_FLOAT, devId);
        InitTensor2D(&Y, batchSize, unitNum, X_FLOAT, devId);
        InitTensor2D(&X, batchSize, embSize, X_FLOAT, devId);
		InitTensor2D(&Z, batchSize, unitNum, X_FLOAT, devId);
		InitTensor2D(&Zi, batchSize, unitNum, X_FLOAT, devId);
		InitTensor2D(&Zf, batchSize, unitNum, X_FLOAT, devId);
		InitTensor2D(&Zo, batchSize, unitNum, X_FLOAT, devId);
        InitTensor2D(&mid0, batchSize, embSize+unitNum, X_FLOAT, devId);
        InitTensor2D(&mid10, batchSize, unitNum, X_FLOAT, devId);
		InitTensor2D(&mid11, batchSize, unitNum, X_FLOAT, devId);
        InitTensor2D(&mid2, batchSize, embSize, X_FLOAT, devId);
		InitTensor2D(&mid3, batchSize, unitNum, X_FLOAT, devId);
        _SetDataFixedFloat(&H, 0.0);
        _SetDataFixedFloat(&Y, 0.0);
        _SetDataFixedFloat(&X, 0.0);
        _SetDataFixedFloat(&C, 0.0);
        _SetDataFixedFloat(&mid2, 0.0);
		_SetDataFixedFloat(&mid3, 0.0);
        InitTensor2D(&W, embSize+unitNum, unitNum, X_FLOAT, devId);
        InitTensor2D(&Wi, embSize + unitNum, unitNum, X_FLOAT, devId);
        InitTensor2D(&Wf, embSize + unitNum, unitNum, X_FLOAT, devId);
        InitTensor2D(&Wo, embSize + unitNum, unitNum, X_FLOAT, devId);
        InitTensor2D(&Wy, unitNum, embSize, X_FLOAT, devId);
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
        Yto = Sum(Y, mid3);
    }

    XTensor *getYpointer()
    {
        return &Y;
    }
    /**/
    void Recur()
    {
		//char ch = '-';
		smallList.Clear();
        smallList.Add(&H);
        smallList.Add(&X);
        mid0 = Merge(smallList, 1);//error when unitnum!=embsize
		//XPRINT1(0, stderr, "3%c", ch);
        Z = MatrixMul(mid0, X_NOTRANS, W, X_NOTRANS);
        Z = HardTanH(Z);
        Zi = MatrixMul(mid0, X_NOTRANS, Wi, X_NOTRANS);
        Zi = Sigmoid(Zi);
        Zf = MatrixMul(mid0, X_NOTRANS, Wf, X_NOTRANS);
		//XPRINT1(0, stderr, "4%c", ch);
        Zf = Sigmoid(Zf);
		//XPRINT1(0, stderr, "41%c", ch);
        Zo = MatrixMul(mid0, X_NOTRANS, Wo, X_NOTRANS);
		//XPRINT1(0, stderr, "42%c", ch);
        Zo = Sigmoid(Zo);
		//XPRINT1(0, stderr, "5%c", ch);
        mid10 = Multiply(C, Zf);
        mid11 = Multiply(Zi, Z);
        C = Sum(mid10, mid11);
        mid10 = HardTanH(C);
		//XPRINT1(0, stderr, "60%c", ch);
		H = Multiply(Zo,mid10);
		//XPRINT1(0, stderr, "61%c", ch);
        Y = Sigmoid(MatrixMul(H, X_NOTRANS, Wy, X_NOTRANS));
		//XPRINT1(0, stderr, "7%c", ch);
    }

    void update()
    {
		if (W.grad != NULL)
		{
			_Sum(&W, W.grad, &W, -learningRate);
			/*float wwww[64][32];
			for (int i = 0; i < 64; i++)
			{
				for (int j = 0; j < 32; j++)
					wwww[i][j]=W.grad->Get2D(i, j);
			}*/
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
        if (mid10.grad != NULL)
            mid10.grad->SetZeroAll();
		if (mid11.grad != NULL)
			mid11.grad->SetZeroAll();
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
		H = HardTanH(mid2);//totally clear to zero
		Y = HardTanH(mid2);
		X = HardTanH(mid2);
		C = HardTanH(mid2);
    }

};

class lstmnet
{
  protected:
    XTensor **input,*batchInput;
    XTensor word2vec, vec2word;
    int layerNum, batchSize, epochs;
    lstmcell *layer0, *layer1, *layer2, *layer3;
    int dataSize,wordNum, unitNum,embSize, maxLen, devId;
    XList outputList, goldList;
    XTensor *finalOutputs,*finalInputs;
	/*if the lstm is bidirection, if the data is shuffled randomly, if the data is loaded in batchSize*/
    bool isBidirection, isShuffle,loadDataInBatch;
	float learningRate,limitForEarlyStop;
	/*amount of batch left for validation, the number for early stopping*/
	int useValidation,earlyStop;
	std::string biMode,infoPath;

  public:
    /*one-layer, more layers to do
	use_gpu:	i>1 for gpu(device i),0 for cpu;
	wm:			wordNum,number of word(in another word,size of vocabulary);
	emb:		size of embedding;
	eph:		number of traning epoch;
	batchSz:	size of a batch;
	lrRate:		learning rate;
	weightInitializer:"zero" for zero,"rand" for random(-1.0,1.0);
	ldDtInBatch:if load data in batch size while training;
	useVld:		amount of batches left for validation;
	isSufl:		if shuffle the data
	*/
    lstmnet(int use_gpu, int wm = 1024, int um = 32, int emb = 32, int eph = 1, int batchSz = 128, float lrRate = 0.001, std::string weightInitializer = "rand", bool LdDtInBatch=true,int usVld=4, bool isSufl = false,std::string ifPh="" )
    {
        char ch = '-';
        devId = use_gpu - 1;
        wordNum = wm;
        embSize = emb;
        layer0 = new lstmcell(devId, emb, um,weightInitializer, batchSz, lrRate);
        unitNum = um;
        layerNum = 1;
        epochs = eph;
		loadDataInBatch = LdDtInBatch;
		useValidation = usVld;
		infoPath = ifPh;
		earlyStop = 0;
		limitForEarlyStop = 0.0;
        batchSize = batchSz;
		learningRate=lrRate;
        isShuffle = isSufl;
        InitTensor2D(&word2vec, wordNum, embSize, X_FLOAT,devId);
        _SetDataRand(&word2vec, -1.0, 1.0);
		InitTensor2D(&vec2word, embSize, wordNum, X_FLOAT, devId);
		_SetDataRand(&vec2word, -1.0, 1.0);
    }
	/*set parameters for stop training. If you need this, please set before training*/
	void setStop(int elyStp = 0, float limit = 0.0)
	{
		earlyStop = elyStp;
		limitForEarlyStop = limit;
	}
	
	/*read all data at one time, quicker but need much more memory*/
	void setInput(std::string inPh, int dtSz,int minLength,int maxLength)
	{
		dataSize = dtSz;
		maxLen = maxLength;
		if (loadDataInBatch)batchInput = new XTensor[maxLen];
		finalOutputs = new XTensor[maxLen];
		for (int i = 0; i < maxLen; ++i)
			InitTensor2D(&finalOutputs[i], batchSize, wordNum, X_FLOAT, devId);
		finalInputs = new XTensor[maxLen];
		for (int i = 0; i < maxLen; ++i)
			InitTensor2D(&finalInputs[i], batchSize, embSize, X_FLOAT, devId);
		input = new XTensor*[dataSize / batchSize];
		for (int i = 0; i < dataSize / batchSize; ++i)
		{
			input[i] = new XTensor[maxLen];
			for (int j = 0; j < maxLen; ++j)
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
    
	/*transform input data into batch size, one file for one batch, which prepare for batch input, memory needed depends on batchSizewwwww*/
	void setBatchInput(std::string inPh, int dtSz, int minLength, int maxLength)
	{
		dataSize = dtSz;
		maxLen = maxLength;
		if (loadDataInBatch)batchInput = new XTensor[maxLen];
		finalOutputs = new XTensor[maxLen];
		finalInputs = new XTensor[maxLen];
		for (int i = 0; i < maxLen; ++i)
			InitTensor2D(&finalInputs[i], batchSize, embSize, X_FLOAT, devId);
		char outPh[20];
		int outIdx[8192];//if there are more than 8192 batches, please edit here
		int sentence[8192];//if the maxlength of sentence is more than 8192, please edit here
		int wordCount[10005];
		int batchNum = 0, num = 0, snum = 0, token;
		memset(wordCount, 0, sizeof(wordCount));
		freopen(inPh.c_str(), "r", stdin);
		while (scanf("%d", &token) != EOF)wordCount[token]++;
		freopen(inPh.c_str(), "r", stdin);
		for (int i = 0; i < dataSize / batchSize; ++i)outIdx[i] = i;
		if (isShuffle)std::random_shuffle(outIdx, outIdx + dataSize / batchSize);//shuffle the sequence of batches
		sprintf(outPh, "data_batch_%d.txt", outIdx[batchNum]);//batch file path
		freopen(outPh, "w", stdout);
		while (scanf("%d", &token) != EOF)
		{
			if (num >= batchSize)
			{
				num = 0;
				batchNum++;
				if (batchNum >= dataSize / batchSize)break;
				sprintf(outPh, "data_batch_%d.txt", outIdx[batchNum]);
				freopen(outPh, "w", stdout);
			}
			if (token == 2)
			{
				if (snum >= minLength)
				{
					if (snum < maxLength)
					{
						for (int i = snum; i < maxLength; i++)
						{
							sentence[i] = 2;
							//printf("2 ");
						}
					}
					for (int i = 0; i < maxLength; i++)
					{
						if (wordCount[sentence[i]] > 20)//delete words that seldom appears 
							printf("%d ", sentence[i]);
						else
							printf("2 ");
						wordCount[sentence[i]]++;
					}
					printf("\n");
					num++;
				}
				snum = 0;

			}
			else if (snum < maxLength)
			{
				if (token >= wordNum)token = 2;
				sentence[snum] = token;
				//printf("%d ", token);
				snum++;
			}
		}
		#ifdef WIN32
		if (infoPath == "")
			freopen("CON", "w", stdout);//redirect to console
		else
			freopen(infoPath.c_str(), "w", stdout);
		#else
		if (infoPath == "")
			freopen("/dev/tty", "w", stdout);
		else
			freopen(infoPath.c_str(), "w", stdout);
		#endif
	}

	/*read the data of designated batchnum, Xtensor:array maxLength of batchSize*wordNum (must call setBatchInput() first)*/
	void getBatchInput(int batchNum,XTensor* batchData)
	{
		char inPh[20];
		int i, j, k,t;
		sprintf(inPh, "data_batch_%d.txt", batchNum);
		freopen(inPh, "r", stdin);
		for (i = 0; i < maxLen; ++i)
		{
			InitTensor2D(&batchData[i], batchSize, wordNum,X_FLOAT,devId);
			_SetDataFixedFloat(&batchData[i], 0.0);
		}
		for(j=0;j<batchSize;++j)
			for(i=0;i<maxLen;++i)
			{
				scanf("%d", &k);
				batchData[i].Set2D(1.0, j, k);
			}
	}

	/*another way or calculating loss*/
	float countLoss(XTensor& gold, XTensor& output)
	{
		XTensor mid1,mid2,mid3;
		int dims[2] = { 1,gold.dimSize[0] };
		mid1 = Multiply(gold, output);
		mid2 = ReduceSum(mid1, 1);
		mid2.Reshape(2, dims);
		mid3 = ReduceSum(mid2, 1);
		return mid3.Get1D(0);
		
	}
	void update()
	{
		if (layer0 != NULL)layer0->update();
		if (word2vec.grad != NULL)
			_Sum(&word2vec, word2vec.grad, &word2vec, -learningRate);
		if (vec2word.grad != NULL)
			_Sum(&vec2word, vec2word.grad, &vec2word, -learningRate);

	}
	void partClear()
	{
		if (layer0 != NULL)layer0->partClear();
		if (word2vec.grad != NULL)
			word2vec.grad->SetZeroAll();
		if (vec2word.grad != NULL)
			vec2word.grad->SetZeroAll();
	}
	/*train model*/
    void train()
    {
        char ch = '-';
        XTensor middle,finalInput;
        XNet autoDiffer;
        float loss,avgloss,pastloss[8192];//if the number of batches is more than 8192, please edit here;
        double startT = GetClockSec();
        for (int epochNum = 0; epochNum < epochs; epochNum++)
        {
			avgloss = 0;
            for (int batchNum = 0; batchNum < dataSize / batchSize-useValidation; batchNum++)
            {
                loss = 0;
                goldList.Clear();
                outputList.Clear();
				//XPRINT1(0, stderr, "0%c", ch);
				if (loadDataInBatch)getBatchInput(batchNum, batchInput);
                if (layerNum == 1)
                {
                    for (int i = 0; i < maxLen; ++i)
					{
						//XPRINT1(0, stderr, "1%c", ch);
						if(finalInputs[i].grad!=NULL)finalInputs[i].grad->SetZeroAll();//finalInputs[i]:batchSize*embSize
                        if(loadDataInBatch)
							finalInputs[i] = Sigmoid(MatrixMul(batchInput[i], X_NOTRANS, word2vec, X_NOTRANS));
						else
							finalInputs[i] = Sigmoid(MatrixMul(input[batchNum][i], X_NOTRANS, word2vec, X_NOTRANS));
                        layer0->setX(finalInputs[i]);
                        if (i > 3)
                        {
							//XPRINT1(0, stderr,"%.3f\n", _CrossEntropyFast(layer0->getYpointer(), &finalInputs[i]));
                            //loss += _CrossEntropyFast(layer0->getYpointer(), &finalInputs[i], REDUCE_SUM);
							//loss += _CrossEntropyFast(&finalOutputs[i - 1], &batchInput[i], REDUCE_MEAN);
							loss -= countLoss(middle, batchInput[i]);
							//goldList.Add(&finalInputs[i]);
							goldList.Add(&batchInput[i]);
							/*float w[10][10000];
							for (int ii = 0; ii < 10; ++ii)
								for (int jj = 0; jj < wordNum; ++jj)
									w[ii][jj] = finalOutputs[i-1].Get2D(ii, jj);
							float ww[128][32];
							for (int ii = 0; ii < 10; ++ii)
								for (int jj = 0; jj < wordNum; ++jj)
									ww[ii][jj] = batchInput[i].Get2D(ii, jj);
							int www;
							www = 1;*/
                        }
						//XPRINT1(0, stderr, "2%c", ch);
						if (i < maxLen - 1)
						{
							layer0->Recur();
							if (i > 2)
							{
								layer0->getYcopy(finalOutputs[i]);//finalOutpus[i]:batchSize*embSize->batchSize*wordNum
								finalOutputs[i] = MatrixMul(finalOutputs[i], X_NOTRANS, vec2word, X_NOTRANS);
								middle = LogSoftmax(finalOutputs[i], 1);
								finalOutputs[i] = LogSoftmax(finalOutputs[i], 1);
								outputList.Add(&finalOutputs[i]);
							}
						}   
                        //XPRINT1(0, stderr,"8%c",ch);
                    }
					autoDiffer.Backward(outputList, goldList, CROSSENTROPY);
					//autoDiffer.ShowNetwork(stderr, &finalOutputs[0]);
					//XPRINT1(0, stderr,"9%c",ch);
					this->update();
					this->partClear();
                }
                else
                {
                    /*TO DO*/
                }
				loss /= (float)((maxLen - 4)*(batchSize));
				avgloss += loss;
                XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, epoch=%d, batch=%d/%d, prob=%.5f\n", GetClockSec() - startT, epochNum + 1, batchNum + 1, dataSize / batchSize-useValidation, exp(loss) );
            }
			pastloss[epochNum] = exp(avgloss / (dataSize / batchSize - useValidation));
			printf("[INFO] epoch = %d, prob = %.5f\n", epochNum + 1, exp(avgloss / (dataSize / batchSize - useValidation)));
			XPRINT2(0,stderr,"[INFO] epoch=%d, prob=%.5f\n", epochNum + 1, exp(avgloss / (dataSize / batchSize-useValidation)));
			validate();
			if (epochNum > earlyStop)
			{
				bool nearSame = true;
				for(int ii=epochNum-earlyStop;ii<=epochNum;++ii)
					for (int jj = ii + 1; jj < epochNum; ++jj)
					{
						if (fabs(preloss[ii] - preloss[jj]) > limitForEarlyStop)
							nearSame = false;
					}
				if (nearSame)break;
			}
        }
    }
	
	/**/
	void validate()
	{
		XTensor middleInput, finalInput,middle;
		float loss = 0.0;
		for (int batchNum = dataSize / batchSize - useValidation; batchNum < dataSize / batchSize; ++batchNum)
		{
            goldList.Clear();
            outputList.Clear();
			if (loadDataInBatch)getBatchInput(batchNum, batchInput);
			if (layerNum == 1)
			{
				for (int i = 0; i < maxLen; ++i)
				{
					//XPRINT1(0, stderr, "1%c", ch);
					if(finalInputs[i].grad!=NULL)finalInputs[i].grad->SetZeroAll();//finalInputs[i]:batchSize*embSize
					if(loadDataInBatch)
						finalInputs[i] = Sigmoid(MatrixMul(batchInput[i], X_NOTRANS, word2vec, X_NOTRANS));
					else
						finalInputs[i] = Sigmoid(MatrixMul(input[batchNum][i], X_NOTRANS, word2vec, X_NOTRANS));
					layer0->setX(finalInputs[i]);
					if (i > 3)
					{
						loss -= countLoss(middle, batchInput[i]);
						//goldList.Add(&finalInputs[i]);
						goldList.Add(&batchInput[i]);
					}
					//XPRINT1(0, stderr, "2%c", ch);
					if (i < maxLen - 1)
					{
						layer0->Recur();
						if (i > 2)
						{
							layer0->getYcopy(finalOutputs[i]);//finalOutpus[i]:batchSize*embSize->batchSize*wordNum
							finalOutputs[i] = MatrixMul(finalOutputs[i], X_NOTRANS, vec2word, X_NOTRANS);
							middle = LogSoftmax(finalOutputs[i], 1);
							finalOutputs[i] = LogSoftmax(finalOutputs[i], 1);
							outputList.Add(&finalOutputs[i]);
						}
					}   
                        //XPRINT1(0, stderr,"8%c",ch);
				}
				this->partClear();
			}
			else
			{
				/*TO DO*/
			}
		}
		printf("[INFO] validation ppl=%.5f\n", exp(loss / ((float)((maxLen - 4)*useValidation*batchSize))));
		XPRINT1(0, stderr, "[INFO] validation ppl=%.5f\n",exp(loss / ((float)((maxLen - 4)*useValidation*batchSize))));
	}

    /**/
    void test()
    {
		XTensor middleInput, finalInput, middle;
		float loss = 0.0;
		for (int batchNum = 0; batchNum < dataSize / batchSize; ++batchNum)
		{
			goldList.Clear();
			outputList.Clear();
			if (loadDataInBatch)getBatchInput(batchNum, batchInput);
			if (layerNum == 1)
			{
				for (int i = 0; i < maxLen; ++i)
				{
					if (finalInputs[i].grad != NULL)finalInputs[i].grad->SetZeroAll();//finalInputs[i]:batchSize*embSize
					if (loadDataInBatch)
						finalInputs[i] = Sigmoid(MatrixMul(batchInput[i], X_NOTRANS, word2vec, X_NOTRANS));
					else
						finalInputs[i] = Sigmoid(MatrixMul(input[batchNum][i], X_NOTRANS, word2vec, X_NOTRANS));
					layer0->setX(finalInputs[i]);
					if (i > 3)
					{
						loss -= countLoss(middle, batchInput[i]);
						goldList.Add(&batchInput[i]);
					}
					if (i < maxLen - 1)
					{
						layer0->Recur();
						if (i > 2)
						{
							layer0->getYcopy(finalOutputs[i]);//finalOutpus[i]:batchSize*embSize->batchSize*wordNum
							finalOutputs[i] = MatrixMul(finalOutputs[i], X_NOTRANS, vec2word, X_NOTRANS);
							middle = LogSoftmax(finalOutputs[i], 1);
							finalOutputs[i] = LogSoftmax(finalOutputs[i], 1);
							outputList.Add(&finalOutputs[i]);
						}
					}
				}
				this->partClear();
			}
			else
			{
				/*TO DO*/
			}
		}
		printf("[INFO] test ppl=%.5f\n", exp(loss / ((float)((maxLen - 4)*dataSize / batchSize*batchSize))));
		XPRINT1(0, stderr, "[INFO] test ppl=%.5f\n", exp(loss / ((float)((maxLen - 4)*dataSize / batchSize*batchSize))));
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