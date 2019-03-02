#include"lstm.h"
#include"model.h"
#include<string>
using namespace nts;
/*parameters:
**sentenceNum:	number of sentence to be loaded
**maxLength:	maximum length of a sentence(the extra length will be cut)
**minLength:	minimum length of a sentence(the sentence shorter than this limit will not be used)
**embSize:		size of embedding, should equal to unitnum because of xtensor's function limitation
**wordNum:		size of word table, or vocabulary
**epochNum:		number of traning epoch
**unitNum:		lstm cell unit size
**batchSize:	size of a batch(set as 1 may lead to error on LINUX)
**kearningRate:	learning rate for training
**weightInitiallizer:"rand" for random initializer between -1.0 and 1.0, "zero" for all zero
**loadDataInBatch:if load data in batch size, need much more memory if choose false
**amountForValidation:the number of batch left for validation
**shuffle:		if shuffle the input data
**testsentenceNum:number of sentence to be loaded during testing
**infoPath:		the file path of recording information
**earlyStop:	auto stop when loss change is smaller than "limitEarlyStop" for more than "earlyStop" consecutive epochs
**limitEarlyStop:see above one
**devId:		compute id, 0 for CPU, 1 or other for GPU
**testPath:		the model loaded when test
**lrRateUpdateMethod:method of updating learning rate, provide "default","exp","step","inv",call setLearningRate to set, see the description of setLearningRate for details
**optimizer		method of optimizer, provide "default","momentum","adagrad", call setOptimizer to set, see the description of setOptimizer for details
**cell			rnn cell type, provide "lstm","gru"
*/
namespace rnn
{
	
	const int sentenceNum = 30000;
	const int maxLength = 50;
	const int minLength = 10;
	const int embSize = 100;
	const int wordNum = 10000;
	const int epochNum = 20;
	const int unitNum = 100;
	const int batchSize = 256;
	const float learningRate = 0.00025;
	const std::string weightInitializer = "rand";
	const bool loadDataInBatch = true;
	const int amountForValidation = 16;
	const bool shuffle = true;
	const int testsentenceNum = 2560;
	const std::string infoPath = "info.txt";
	const int layerNum = 2;
	const int earlyStop = 3;
	const float limitEarlyStop = 0.5;
	const int devId = 2;
	char testPath[50] = "model_30000_256_18.ckpt";
	const std::string lrRateUpdateMethod = "default";
	const std::string optimizer = "default";
	const std::string cell = "lstm";
	/*
	const int sentenceNum = 512;
	const int maxLength = 20;
	const int minLength = 1;
	const int embSize = 20;
	const int wordNum = 10000;
	const int epochNum = 10;
	const int unitNum = 25;
	const int batchSize = 32;
	const float learningRate = 0.001;
	const std::string weightInitializer = "rand";
	const bool loadDataInBatch = true;
	const int amountForValidation = 2;
	const bool shuffle = true;
	const int testsentenceNum = 256;
	const std::string infoPath = "info.txt";
	const int layerNum = 1;
	const int earlyStop = 3;
	const float limitEarlyStop = 0.5;
	const int devId = 1;
	char testPath[50] = "model_1024_128_2.ckpt";
	const std::string lrRateUpdateMethod = "default";
	const std::string optimizer = "default";
	const std::string cell = "lstm";
	*/
	/*train*/
	void trueMain()
	{
		printf("train begin\n");
		/*initialize a lstmnet*/

		lstmnet testLstmnet(devId, wordNum, unitNum, embSize, epochNum, batchSize, learningRate, weightInitializer, loadDataInBatch, amountForValidation, shuffle, infoPath,layerNum);
		/*use setBatchInput to set data when load data into batch size, else use setInput*/
		//testLstmnet.setInput("wsj-00-20.id.vocab10k", sentenceNum,minLength,maxLength);
		testLstmnet.setBatchInput("wsj-00-20.id.vocab10k", sentenceNum, minLength, maxLength, false);
		/*use setLearningRate to change learning rate online*/
		testLstmnet.setLearningRate("lrRateUpdateMethod", 0.9, 50.0);
		/*use setOptimizer to set optimizer*/
		testLstmnet.setOptimizer(optimizer);
		/*use setStop to set early stop parameters*/
		testLstmnet.setStop(earlyStop, limitEarlyStop);
		printf("data and parameter set, train begin\n");
		/*train it*/
		testLstmnet.train();
		printf("train end\n");
		printf("test begin\n");
		testLstmnet.setBatchInput("wsj-21-22.id.vocab10k", testsentenceNum, minLength, maxLength, true);
		testLstmnet.test();
		printf("test end\n");
	}

	/*load and test*/
	void trueTest()
	{
		printf("test begin\n");
		lstmnet testLstmnet(devId, wordNum, unitNum, embSize, epochNum, batchSize, learningRate, weightInitializer, loadDataInBatch, amountForValidation, shuffle, infoPath);
		/*use read to load pretrained model*/
		testLstmnet.read(0, testPath);
		testLstmnet.setBatchInput("wsj-21-22.id.vocab10k", testsentenceNum, minLength, maxLength, true);
		testLstmnet.test();
	};
}
namespace onemodel{
	void fakeMain()
	{
		Model model(1, 5, 0.001);
		model.AddInput("1data.txt", 25, 32, 2);
		model.AddDense(2);
		model.AddDense(2);
		model.AddDense(1);
		model.AddGold("1gold.txt");
		model.Train(CROSSENTROPY);
	}

}
int main(int argc, const char ** argv)
{
    rnn::trueMain();
	//rnn::trueTest();
	//onemodel::fakeMain();
    return 0;
}

