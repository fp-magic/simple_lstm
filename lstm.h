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

	~lstmcell()
	{
		XPRINT(0, stderr, "cell destructed");
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
    
	/*skr the cell(recurrence)!*/
    void Recur()
    {
		smallList.Clear();
        smallList.Add(&H);
        smallList.Add(&X);
        mid0 = Merge(smallList, 1);//error when H's size is not euqual to X's, when unitnum!=embsize
        Z = MatrixMul(mid0, X_NOTRANS, W, X_NOTRANS);
        Z = HardTanH(Z);
        Zi = MatrixMul(mid0, X_NOTRANS, Wi, X_NOTRANS);
        Zi = Sigmoid(Zi);
        Zf = MatrixMul(mid0, X_NOTRANS, Wf, X_NOTRANS);
        Zf = Sigmoid(Zf);
        Zo = MatrixMul(mid0, X_NOTRANS, Wo, X_NOTRANS);
        Zo = Sigmoid(Zo);
        mid10 = Multiply(C, Zf);
        mid11 = Multiply(Zi, Z);
        C = Sum(mid10, mid11);
        mid10 = HardTanH(C);
		H = Multiply(Zo,mid10);
        Y = Sigmoid(MatrixMul(H, X_NOTRANS, Wy, X_NOTRANS));
    }

	/*update*/
    void update(std::string optMethod="default",float optParam1=0)
    {
		XList list;
		list.Add(&W);
		list.Add(&Wi);
		list.Add(&Wf);
		list.Add(&Wo);
		list.Add(&Wy);
		if (optMethod == "default")
		{
			for (int i = 0; i < list.count; i++)
			{
				XTensor* para = (XTensor*)list.GetItem(i);
				_Sum(para, para->grad, para, -learningRate);
			}
		}
		if (optMethod == "momentum")
		{
			for (int i = 0; i < list.count; i++)
			{
				XTensor* para = (XTensor*)list.GetItem(i);
				_Sum(para->grad, para->grad, para->grad, optParam1 - 1.0);
				_Sum(para, para->grad, para, -learningRate);
			}
		}
		/*if (W.grad != NULL)
			_Sum(&W, W.grad, &W, -learningRate);					
        if (Wi.grad != NULL)
            _Sum(&Wi, Wi.grad, &Wi, -learningRate);
        if (Wf.grad != NULL)
            _Sum(&Wf, Wf.grad, &Wf, -learningRate);
        if (Wo.grad != NULL)
            _Sum(&Wo, Wo.grad, &Wo, -learningRate);
        if (Wy.grad != NULL)
            _Sum(&Wy, Wy.grad, &Wy, -learningRate);*/
    }

    /*clear needed part*/
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

	/*save parameters to file*/
	void dump(FILE* file)
	{
		W.Dump(file, "W");
		Wi.Dump(file, "Wi");
		Wf.Dump(file, "Wf");
		Wo.Dump(file, "Wo");
		Wy.Dump(file, "Wy");
	}

	/*load parameters from file*/
	void read(FILE* file)
	{
		W.Read(file, "W");
		Wi.Read(file, "Wi");
		Wf.Read(file, "Wf");
		Wo.Read(file, "Wo");
		Wy.Read(file, "Wy");
	}
};

class lstmnet
{
  protected:
    XTensor **input,*batchInput;
    XTensor word2vec, vec2word;
    int layerNum, batchSize, epochs;
    lstmcell *layer0, *layer1,*layer2,*layer3;
    int dataSize,wordNum, unitNum,embSize, maxLen, devId;
    XList outputList, goldList;
    XTensor *finalOutputs,*finalInputs;
	/*if the lstm is bidirection, if the data is shuffled randomly, if the data is loaded in batchSize*/
    bool isBidirection, isShuffle,loadDataInBatch;
	float learningRate,limitForEarlyStop;
	/*amount of batch left for validation, the number for early stopping*/
	int useValidation,earlyStop;
	std::string infoPath,lrMethod,optMethod;
	float lrParam1, lrParam2, lrParam3, lrParam4;
	float optParam1, optParam2;
	XNet autoDiffer;

  public:
    /*one-layer, more layers to do
	**use_gpu:	i>1 for gpu(device i),0 for cpu;
	**wm:			wordNum,number of word(in another word,size of vocabulary);
	**emb:		size of embedding;
	**eph:		number of traning epoch;
	**batchSz:	size of a batch;
	**lrRate:		learning rate;
	**weightInitializer:"zero" for zero,"rand" for random(-1.0,1.0);
	**ldDtInBatch:if load data in batch size while training;
	**useVld:		amount of batches left for validation;
	**isSufl:		if shuffle the data
	**layerNum:		lstm layer number
	*/
    lstmnet(int use_gpu=0, int wm = 1024, int um = 32, int emb = 32, int eph = 1, int batchSz = 128, float lrRate = 0.001, std::string weightInitializer = "rand", bool LdDtInBatch=true,int usVld=2, bool isSufl = false,std::string ifPh="" ,int lyNm=1)
    {
        devId = use_gpu - 1;
        wordNum = wm;
        embSize = emb;
        unitNum = um;
        epochs = eph;
		batchSize = batchSz;
		learningRate=lrRate;
		loadDataInBatch = LdDtInBatch;
		useValidation = usVld;
		isShuffle = isSufl;
		infoPath = ifPh;
        layerNum = lyNm;
        InitTensor2D(&word2vec, wordNum, embSize, X_FLOAT,devId);
        _SetDataRand(&word2vec, -1.0, 1.0);
		InitTensor2D(&vec2word, embSize, wordNum, X_FLOAT, devId);
		_SetDataRand(&vec2word, -1.0, 1.0);
		if (layerNum>0)layer0 = new lstmcell(devId, embSize, unitNum, weightInitializer, batchSize, learningRate);
		if (layerNum>1)layer1 = new lstmcell(devId, embSize, unitNum, weightInitializer, batchSize, learningRate);
		if (layerNum>2)layer2 = new lstmcell(devId, embSize, unitNum, weightInitializer, batchSize, learningRate);
		if (layerNum>3)layer3 = new lstmcell(devId, embSize, unitNum, weightInitializer, batchSize, learningRate);
		earlyStop = 0;
		limitForEarlyStop = 0.0;
		lrMethod = "default";
		optMethod = "default";
    }
	
	~lstmnet()
	{
		delete[] batchInput;
		delete[] finalOutputs;
		delete[] finalInputs;
		if (layerNum>0)delete layer0;
		if (layerNum>1)delete layer1;
		if (layerNum>2)delete layer2;
		if (layerNum>3)delete layer3;
		XPRINT(0, stderr, "net destructed");
	}
	
	/*set method of updating learning rate online*/
	/*details follow https://blog.csdn.net/langb2014/article/details/51274376*/
	void setLearningRate(std::string method ="default",float param1=0, float param2=0,float param3=0)
	{
		lrMethod = method;
		if (method == "default")return;
		if (method == "exp")//start quick, end slow, consective##lr=lr*pow(param1,1.0/param2) or lr=learningRate*pow(param1,step/param2)
		{
			if (param2 == 0 && param1 == 0)
				lrParam1 = 0.9999;
			else
				lrParam1 = pow(param1, 1.0 / param2);
		}
		if (method == "step")//start quick, end slow, consecutive##lr=learningRate*pow(param1,floor(step/param2)
		{
			if (param1 == 0)
				lrParam1 = 0.99;
			else	
				lrParam1 = param1;
			if (param2 == 0)
				lrParam2 = 1000;
			else
				lrParam2 = param2;
			lrParam3 = 0;
		}
		if (method == "inv")//start quick, end slow,consecutive##lr=learningRate*(1+param1*step)^(-param2)
		{
			if (param1 == 0)
				lrParam1 = 0.0001;
			else
				lrParam1 = param1;
			if (param2 == 0)
				lrParam2 = 0.75;
			else
				lrParam2 = param2;
			lrParam3 = 0;
			lrParam4 = learningRate;

		}
	}
	
	/*set method of optimizer for updating*/
	void setOptimizer(std::string method="default",float param1=0,float param2=0)
	{
		optMethod = method;
		if (optMethod == "default")return;
		if (optMethod == "momentum")
		{
			optParam1 = param1;
		}

	}

	/*set parameters for stop training. If you need this, please set before training
	**auto stop when loss change is smaller than "limitForEarlyStop" for more than "earlyStop" consecutive epochs
	*/
	void setStop(int elyStp = 0, float limit = 0.0)
	{
		earlyStop = elyStp;
		limitForEarlyStop = limit;
	}
	
	/*read all data at one time, quicker but need much more memory
	**have not been updated for a while, strongly not recommend
	*/
	void setInput(std::string inPh, int dtSz,int minLength,int maxLength)
	{
		dataSize = dtSz;
		maxLen = maxLength;
		/*initialize xtensor about data for training period*/
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
		/*open input file*/
		freopen(inPh.c_str(), "r", stdin);
		int batchNum=0,num=0,snum=0,token;
		/*load data*/
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
    
	/*transform input data into batch size, one file for one batch, which prepare for batch input, memory needed depends on batchSize
	**inPh:path of input data
	**dtSz:number of sentence to be loaded
	**maxLength:maximum length of a sentence(the extra length will be cut)
	**minLength:minimum length of a sentence(the sentence shorter than this limit will not be used)
	**forTest:if the data is set for test. if true, the output stream will not be redirected
	*/
	void setBatchInput(std::string inPh, int dtSz, int minLength, int maxLength,bool forTest=false)
	{
		dataSize = dtSz;
		maxLen = maxLength;
		/*initialize xtensor about data for training period*/
		batchInput = new XTensor[maxLen];
		finalOutputs = new XTensor[maxLen];
		finalInputs = new XTensor[maxLen];
		for (int i = 0; i < maxLen; ++i)
			InitTensor2D(&finalInputs[i], batchSize, embSize, X_FLOAT, devId);
		char outPh[20];
		int outIdx[8192];//if there are more than 8192 batches, please edit here
		int sentence[8192];//if the maxlength of sentence is more than 8192, please edit here
		int wordCount[10005];//record word frequency
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
						if (wordCount[sentence[i]] > 20)//replace words with extremely low frequency 
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
				snum++;
			}
		}
		#ifdef WIN32
		if (infoPath == "")
			freopen("CON", "w", stdout);//redirect to console
		else
			if (forTest)infoPath = "test_" + infoPath;
			freopen(infoPath.c_str(), "w", stdout);
		#else
		if (infoPath == "")
			freopen("/dev/tty", "w", stdout);
		else
			if (forTest)infoPath = "test_" + infoPath;
			freopen(infoPath.c_str(), "w", stdout);
		#endif
	}

	/*read the data of designated batchnum, Xtensor:array maxLength of batchSize*wordNum (must call setBatchInput() first)
	**load the "batchNum"th batch into "batchData"
	*/
	void getBatchInput(int batchNum,XTensor* batchData)
	{
		char inPh[20];
		int k;
		sprintf(inPh, "data_batch_%d.txt", batchNum);
		freopen(inPh, "r", stdin);
		for (int i = 0; i < maxLen; ++i)
		{
			InitTensor2D(&batchData[i], batchSize, wordNum,X_FLOAT,devId);
			_SetDataFixedFloat(&batchData[i], 0.0);
		}
		for(int j=0;j<batchSize;++j)
			for(int i=0;i<maxLen;++i)
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

	/*update all(learning rate as well)*/
	void update()
	{
		if (lrMethod == "exp")
		{
			learningRate *= lrParam1;
		}
		if (lrMethod == "step")
		{
			lrParam3++;
			if (lrParam3 >= lrParam2)
			{
				lrParam3 = 0;
				learningRate *= lrParam1;
			}
		}
		if (lrMethod == "inv")
		{
			lrParam3++;
			learningRate = lrParam4*pow(1.0 + lrParam1*lrParam3, -lrParam2);
		}
		if (layer0 != NULL)layer0->update(optMethod,optParam1);
		if (optMethod == "default")
		{
			if (word2vec.grad != NULL)
				_Sum(&word2vec, word2vec.grad, &word2vec, -learningRate);
			if (vec2word.grad != NULL)
				_Sum(&vec2word, vec2word.grad, &vec2word, -learningRate);
		}
		if (optMethod == "momentum")
		{
			if (word2vec.grad != NULL)
			{
				_Sum(word2vec.grad, word2vec.grad, word2vec.grad, optParam1 - 1.0);
				_Sum(&word2vec, word2vec.grad, &word2vec, -learningRate);
			}
			if (vec2word.grad != NULL)
			{
				_Sum(vec2word.grad, vec2word.grad, vec2word.grad, optParam1 - 1.0);
				_Sum(&vec2word, vec2word.grad, &vec2word, -learningRate);
			}
		}
		

	}

	/*clear needed part*/
	void partClear()
	{
		if (layer0 != NULL)layer0->partClear();
		if (word2vec.grad != NULL)
			word2vec.grad->SetZeroAll();
		if (vec2word.grad != NULL)
			vec2word.grad->SetZeroAll();
	}

	/*clear and retrain model
	**MAY NOT WORK ON LINUX!
	*/
	void retrain()
	{
		XPRINT(0, stderr, "loss is too high, retrain start\n");
		InitTensor2D(&word2vec, wordNum, embSize, X_FLOAT, devId);
		_SetDataRand(&word2vec, -1.0, 1.0);
		InitTensor2D(&vec2word, embSize, wordNum, X_FLOAT, devId);
		_SetDataRand(&vec2word, -1.0, 1.0);
		delete layer0;
		XPRINT(0, stderr, "e");
		layer0 = new lstmcell(devId, embSize, unitNum, "rand", batchSize, learningRate);
		XPRINT(0, stderr, "f");
		train();
	}

	/*train model*/
    void train()
    {
        XTensor middle,finalInput;
        float loss,avgloss,pastloss[8192];//if the number of batches is more than 8192, please edit here;
        double startT = GetClockSec(),dT=0,nowT=0;
		/*iterate epoch*/
        for (int epochNum = 0; epochNum < epochs; epochNum++)
        {
			avgloss = 0;
			/*iterate batch*/
            for (int batchNum = 0; batchNum < dataSize / batchSize-useValidation; batchNum++)
            {
                loss = 0;
                goldList.Clear();
                outputList.Clear();
				if (loadDataInBatch)getBatchInput(batchNum, batchInput);
				/*iterate hidden unit*/
				for (int i = 0; i < maxLen; ++i)
				{
					if (finalInputs[i].grad != NULL)finalInputs[i].grad->SetZeroAll();//finalInputs[i]:batchSize*embSize
					/*get input*/
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
						if (layerNum > 0)
						{
							layer0->Recur();
							layer0->getYcopy(finalOutputs[i]);
						}
						if (layerNum > 1)
						{
							layer1->setX(finalOutputs[i]);
							layer1->Recur();
							layer1->getYcopy(finalOutputs[i]);
						}
						if (layerNum > 2)
						{
							layer2->setX(finalOutputs[i]);
							layer2->Recur();
							layer2->getYcopy(finalOutputs[i]);
						}
						if (layerNum > 3)
						{
							layer3->setX(finalOutputs[i]);
							layer3->Recur();
							layer3->getYcopy(finalOutputs[i]);
						}
						if (i > 2)
						{
							//finalOutpus[i]:batchSize*embSize->batchSize*wordNum
							finalOutputs[i] = MatrixMul(finalOutputs[i], X_NOTRANS, vec2word, X_NOTRANS);
							middle = LogSoftmax(finalOutputs[i], 1);
							finalOutputs[i] = LogSoftmax(finalOutputs[i], 1);
							outputList.Add(&finalOutputs[i]);
						}
					}
				}
				autoDiffer.Backward(outputList, goldList, CROSSENTROPY);
				this->update();
				this->partClear();
				loss /= (float)((maxLen - 4)*(batchSize));
				avgloss += loss;
				dT = dT*0.9 +(GetClockSec() - startT - nowT)*0.1;
				nowT = GetClockSec() - startT;
                XPRINT6(0, stderr, "[INFO]elapsed=%.1fs, epoch=%d, batch=%d/%d, prob=%8.1f, rest=%.1fs\n", nowT, epochNum + 1, batchNum + 1, dataSize / batchSize-useValidation, exp(loss), dT*(dataSize / batchSize - useValidation-batchNum-1));
				XPRINT5(0, stdout, "[INFO]elapsed=%.1fs, epoch=%d, batch=%d/%d, prob=%.1f\n", nowT, epochNum + 1, batchNum + 1, dataSize / batchSize - useValidation, exp(loss));
            }
			XPRINT3(0,stderr,"[INFO] epoch=%d/%d, prob=%.5f\n", epochNum + 1,epochs, exp(avgloss / (dataSize / batchSize-useValidation)));
			XPRINT3(0, stdout, "[INFO] epoch=%d/%d, prob=%.5f\n", epochNum + 1, epochs, exp(avgloss / (dataSize / batchSize - useValidation)));
			this->validate();
			/*save model after every epoch*/
			this->dump(epochNum + 1);
			/*check if stop early*/
			pastloss[epochNum] = exp(avgloss / (dataSize / batchSize - useValidation));
			if (epochNum > earlyStop)
			{
				bool nearSame = true;
				for(int ii=epochNum-earlyStop;ii<=epochNum;++ii)
					for (int jj = ii + 1; jj < epochNum; ++jj)
					{
						if (fabs(pastloss[ii] - pastloss[jj]) > limitForEarlyStop)
							nearSame = false;
					}
				if (nearSame)break;
			}
			/*check if the loss is so high that we need retrain*/
			if (epochNum == 0 && exp(avgloss / (dataSize / batchSize - useValidation)) >= 10000.0)
			{
				this->retrain();
				return;
			}
        }
    }
	
	/*validation procedure*/
	void validate()
	{
		XTensor middleInput, finalInput,middle;
		float loss = 0.0;
		for (int batchNum = dataSize / batchSize - useValidation; batchNum < dataSize / batchSize; ++batchNum)
		{
            goldList.Clear();
            outputList.Clear();
			if (loadDataInBatch)getBatchInput(batchNum, batchInput);
			for (int i = 0; i < maxLen; ++i)
			{
				if(finalInputs[i].grad!=NULL)finalInputs[i].grad->SetZeroAll();//finalInputs[i]:batchSize*embSize
				if(loadDataInBatch)
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
					if (layerNum > 0)
					{
						layer0->Recur();
						layer0->getYcopy(finalOutputs[i]);
					}
					if (layerNum > 1)
					{
						layer1->setX(finalOutputs[i]);
						layer1->Recur();
						layer1->getYcopy(finalOutputs[i]);
					}
					if (layerNum > 2)
					{
						layer2->setX(finalOutputs[i]);
						layer2->Recur();
						layer2->getYcopy(finalOutputs[i]);
					}
					if (layerNum > 3)
					{
						layer3->setX(finalOutputs[i]);
						layer3->Recur();
						layer3->getYcopy(finalOutputs[i]);
					}
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
		XPRINT1(0, stderr, "[INFO] validation ppl=%.5f\n",exp(loss / ((float)((maxLen - 4)*useValidation*batchSize))));
		XPRINT1(0, stdout, "[INFO] validation ppl=%.5f\n", exp(loss / ((float)((maxLen - 4)*useValidation*batchSize))));
	}

    /*test procedure*/
    void test()
    {
		XTensor middleInput, finalInput, middle;
		float loss = 0.0;
		for (int batchNum = 0; batchNum < dataSize / batchSize; ++batchNum)
		{
			if (loadDataInBatch)getBatchInput(batchNum, batchInput);
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
				}
				if (i < maxLen - 1)
				{
					if (layerNum > 0)
					{
						layer0->Recur();
						layer0->getYcopy(finalOutputs[i]);
					}
					if (layerNum > 1)
					{
						layer1->setX(finalOutputs[i]);
						layer1->Recur();
						layer1->getYcopy(finalOutputs[i]);
					}
					if (layerNum > 2)
					{
						layer2->setX(finalOutputs[i]);
						layer2->Recur();
						layer2->getYcopy(finalOutputs[i]);
					}
					if (layerNum > 3)
					{
						layer3->setX(finalOutputs[i]);
						layer3->Recur();
						layer3->getYcopy(finalOutputs[i]);
					}
					if (i > 2)
					{
						layer0->getYcopy(finalOutputs[i]);//finalOutpus[i]:batchSize*embSize->batchSize*wordNum
						finalOutputs[i] = MatrixMul(finalOutputs[i], X_NOTRANS, vec2word, X_NOTRANS);
						middle = LogSoftmax(finalOutputs[i], 1);
						finalOutputs[i] = LogSoftmax(finalOutputs[i], 1);
					}
				}
			}
			this->partClear();
		}
		printf("[INFO] test ppl=%.5f\n", exp(loss / ((float)((maxLen - 4)*dataSize / batchSize*batchSize))));
		XPRINT1(0, stderr, "[INFO] test ppl=%.5f\n", exp(loss / ((float)((maxLen - 4)*dataSize / batchSize*batchSize))));
	}

    /*save model*/
    void dump(int someParam=0,char* modelPath=NULL)
    {
		if (modelPath == NULL)
		{
			modelPath = new char[50];
			sprintf(modelPath, "model_%d_%d_%d.ckpt", dataSize, batchSize, someParam);
		}
		FILE * file = fopen(modelPath, "w+b");
		layer0->dump(file);
		word2vec.Dump(file, "word2vec");
		vec2word.Dump(file, "vec2word");
		fclose(file);
		XPRINT(0, stderr, "[INFO] model saved\n");
    }

    /*load model*/
	void read(int someParam = 0,char* modelPath = NULL )
	{
		if (modelPath == NULL)
		{
			modelPath = new char[50];
			sprintf(modelPath, "model_%d_%d_%d.ckpt", dataSize, batchSize, someParam);
		}
		FILE * file = fopen(modelPath, "r+b");
		layer0->read(file);
		word2vec.Read(file, "word2vec");
		vec2word.Read(file, "vec2word");
		fclose(file);
		XPRINT(0, stderr, "[INFO] model loaded\n");
	}
};

}; // namespace lstm
#endif