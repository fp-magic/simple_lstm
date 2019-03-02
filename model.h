#ifndef __MODEL_H__
#define __MODEL_H__
#include "../tensor/XGlobal.h"
#include "../tensor/XTensor.h"
#include "../tensor/XUtility.h"
#include "../tensor/XDevice.h"
#include "../tensor/core/CHeader.h"
#include "../tensor/function/FHeader.h"
#include "../network/XNet.h"
#include <cmath>
#include <string>
#include <vector>
#include<algorithm>
using namespace nts;
namespace onemodel
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
class Basiclayer
{
protected:
	XTensor X, Y;
	XTensor Zero;
	std::string Name;
	XList Para, All;
public:
	/* Size*From->Size*To */
	Basiclayer(int Size,int From, int To)
	{
		InitTensor2D(&X, Size, From);
		InitTensor2D(&Y, Size, To);
		Para.Add(&X);
		Para.Add(&Y);
		All.Add(&X);
		All.Add(&Y);
		All.Add(&Zero);
	}

	void SetX(XTensor* Px)
	{
		InitTensor2D(&Zero, Px->dimSize[0], Px->dimSize[1]);
		_SetDataFixedFloat(&Zero, 0.0);
		X = Sum(Zero, *Px);
	}

	XTensor* GetY()
	{
		return &Y;
	}

	void Show()
	{
		XPRINT5(0, stderr, "%s: [%d ,%d ] ---> [%d, %d ]\n", Name.c_str(), X.dimSize[0], X.dimSize[1], Y.dimSize[0], Y.dimSize[1]);
	}

	virtual void Train() = 0;

	std::string GetName()
	{
		return Name;
	}
	void Activate(XTensor& T, std::string Act)
	{
		if (Act == "Sigmoid")T = Sigmoid(T);
		if (Act == "HardTanh")T = HardTanH(T);
		if (Act == "Softmax")T = Softmax(T, 1);
		if (Act == "LogSoftmax")T = LogSoftmax(T, 1);
	}

	void Update(float learningrate)
	{
		int i;
		for (i = 0; i < Para.count; ++i)
		{
			XTensor* T = (XTensor*)Para.GetItem(i);
			if (T->grad != NULL)
			{
				_Sum(T, T->grad, T, -learningrate);
			}
		}
	}
	void Clear()
	{
		int i;
		for (i = 0; i < All.count; ++i)
		{
			XTensor* T = (XTensor*)All.GetItem(i);
			if (T->grad != NULL)
			{
				T->grad->SetZeroAll();
			}
		}
	}
};

class Inputlayer :public Basiclayer
{
protected:
	int BatchNum,BatchNow,BatchSize,DataLength;
	std::string Path;
public:
	Inputlayer(std::string InPath, int Batch, int Size, int Length) :Basiclayer(Size, Length, Length)
	{
		int i, j, k;
		float token;
		char OutPath[128];
		BatchNum = Batch;
		BatchNow = 0;
		Path = InPath;
		BatchSize = Size;
		DataLength = Length;
		Name = "Inputlayer";
		freopen(Path.c_str(), "r", stdin);
		for (i = 0; i < BatchNum; ++i)
		{
			sprintf(OutPath, "%s_batch%d.txt", InPath.c_str(), i);
			freopen(OutPath, "w", stdout);
			for (j = 0; j < BatchSize; ++j)
			{
				for (k = 0; k < DataLength; ++k)
				{
					scanf("%f", &token);
					printf("%f ", token);
				}
				printf("\n");
			}
		}	
#ifdef WIN32			
		freopen("CON", "w", stdout);
#else
		freopen("/dev/tty", "w", stdout);
#endif
	}
	void Train()
	{
		char InPath[128];
		int i, j;
		float token;
		sprintf(InPath, "%s_batch%d.txt", Path.c_str(), BatchNow);
		freopen(InPath, "r", stdin);
		for (i = 0; i < BatchSize; ++i)
		{
			for (j = 0; j < DataLength; ++j)
			{
				scanf("%f", &token);
				Y.Set2D(token, i, j);
			}
		}
		BatchNow = (BatchNow + 1) % BatchNum;
	}
};


class Denselayer :public Basiclayer
{
protected:
	XTensor W;
	std::string Activation;
public:
	Denselayer(int Size,int From,int To,std::string activation):Basiclayer(Size,From,To)
	{
		InitTensor2D(&W, From, To);
		_SetDataRand(&W, -1.0, 1.0);
		Activation = activation;
		Name = "Denselayer";
		Para.Add(&W);
		All.Add(&W);
	}
	void Train()
	{
		Y = MatrixMul(X,X_NOTRANS,W,X_NOTRANS);
		Activate(Y, Activation);
	}
};

class Rnnlayer :public Basiclayer
{
protected:
	XTensor X, Y, mid2, mid3;
	std::string Type;
	int BatchNow,BatchSize,UnitNum,Length;
	XList input, output;
    Rnnlayer(int Size,int From,int To,int unitnum,int length,std::string type,bool outputh)
    :Basiclayer(Size,From,outputh?UnitNum:UnitNum*length){
		input.Clear();
		CheckErrors(From%length == 0, "input size of rnn is not right");
		if (!outputh)
		{
			for (int i = 0; i < input.count; ++i)
			{

			}
		}
        
    }
};

class Model
{
protected:
	XNet Differ;
	int Epoch, BatchNum, BatchSize;
	std::vector<Basiclayer*>Layer;
	float LearningRate;
	std::string LrMethod;
	float LrParam1, LrParam2, LrParam3;
public:
	Model(int devid = 0, int epoch= 1,float learningrate = 0.001)
	{
		Layer.clear();
		Epoch = epoch;
		LearningRate = learningrate;
		LrMethod = "default";
	}

	~Model()
	{
		for (int i = 0; i < Layer.size(); i++)
			delete Layer[i];
	}

	void AddInput(std::string path,int batchnum,int batchsize,int length)
	{
		BatchNum = batchnum;
		BatchSize = batchsize;
		Basiclayer* onelayer = new Inputlayer(path, BatchNum,BatchSize,length);
		Layer.push_back(onelayer);
	}

	void AddGold(std::string path)
	{
		Basiclayer* onelayer = new Inputlayer(path, BatchNum, BatchSize, Layer[Layer.size()-1]->GetY()->dimSize[1]);
		Layer.push_back(onelayer);
	}

	void AddDense(int To,std::string activation="Sigmoid")
	{
		Basiclayer* onelayer = new Denselayer(BatchSize,Layer[Layer.size()-1]->GetY()->dimSize[1],To,activation);
		Layer.push_back(onelayer);
	}

    void AddRnn()
    {
        /*todo*/
    }

	void Train(LOSS_FUNCTION_NAME lossmode=CROSSENTROPY,std::string mode="Train")
	{
		this->Show();
		this->check();
		int i, j, k;
		float loss=0.0;
		XList Output, Gold;
		for (i = 1; i <= Epoch; ++i)
		{
			for (j = 0; j < BatchNum; ++j)
			{
				for (k = 0; k < Layer.size(); ++k)
				{
					if(k!=0)Layer[k]->SetX(Layer[k-1]->GetY());
					Layer[k]->Train();
					//this->Output(Layer[k]->GetY());
				}
				Output.Add(Layer[Layer.size() - 2]->GetY());
				Gold.Add(Layer[Layer.size() - 1]->GetY());
				loss += _LossCompute(Layer[Layer.size() - 1]->GetY(), Layer[Layer.size() - 2]->GetY(), lossmode, false, 0, 0, Layer[Layer.size() - 1]->GetY()->dimSize[0], 0);
				if(mode=="Train")Differ.Backward(Output, Gold, lossmode);
				this->Update();
				XPRINT3(0,stderr,"[INFO] Epoch=%d Batch=%d Loss=%.5f\n", i, j+1, loss);
			}
		}
	}

	void Test(LOSS_FUNCTION_NAME lossmode = CROSSENTROPY)/*todo*/
	{
		this->Train(lossmode, "Test");
	}

	void Update()
	{
		int i;
		for (i = 1; i < Layer.size() - 1; ++i)
			Layer[i]->Update(LearningRate);
		for (i = 0; i < Layer.size(); ++i)
			Layer[i]->Clear();
	}

	void check()
	{
		int i;
		XPRINT(0,stderr,"Checking model...\n");
		for (i = 1; i < Layer.size() - 1; i++)
		{
			CheckNTErrors(Layer[i]->GetName() != "Inputlayer", "Input only in the first layer and gold only in the last layer!");
		}
		CheckNTErrors(Layer[0]->GetName()=="Inputlayer", "The first layer should be input!");
		CheckNTErrors(Layer[Layer.size()-1]->GetName()=="Inputlayer","The last layer should be gold!");
		CheckNTErrors(Layer.size() >= 3, "Model should have be at least 3 layers!");
		XPRINT(0,stderr,"Model checked!\n")
	}

	void Output(XTensor* T)
	{
		int i, j;
		for (i = 0; i < T->dimSize[0]; ++i)
		{
			for (j = 0; j < T->dimSize[1]; ++j)
				XPRINT1(0,stderr,"%.2f ", T->Get2D(i, j));
			//XPRINT(0,stderr,"\n");
		}
		XPRINT(0, stderr, "\n\n");
		
	}

	void Show()
	{
		int i;
		for (i = 0; i < Layer.size(); ++i)
		{
			Layer[i]->Show();
		}
	}

	void Dump()/*todo*/
	{

	}

	void Read()/*todo*/
	{

	}
};
}//namespace onemodel
#endif