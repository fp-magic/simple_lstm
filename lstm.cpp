#include"lstm.h"
#include<string>
using namespace nts;

namespace lstm
{
    
const int sentenceNum=32768;
const int maxLength=50;
const int minLength=1;
const int embSize=100;
const int wordNum=10000;
const int epochNum = 10;
const int unitNum = 100;
const int batchSize = 256;
const float learningRate = 0.001;
const std::string weightInitializer = "rand";
const bool loadDataInBatch = true;
const int amountForValidation = 16;
const bool shuffle=true;
const int testsentenceNum = 2560;
const std::string infoPath = "info.txt";
const int earlyStop = 3;
const float limitEarlyStop = 0.5;

void trueMain(int use_gpu)
{
    printf("w\n");
	//checkInput(trueInput);
	printf("e");
    lstmnet testLstmnet(use_gpu,wordNum,unitNum,embSize,epochNum,batchSize,learningRate,weightInitializer,loadDataInBatch,amountForValidation,shuffle,infoPath,earlyStop);
	//testLstmnet.getInput("wsj-00-20.id.vocab10k", sentenceNum,minLength,maxLength);
	testLstmnet.setBatchInput("wsj-00-20.id.vocab10k", sentenceNum, minLength, maxLength);
	testLstmnet.setStop(earlyStop, limitEarlyStop);
	printf("r\n");
    testLstmnet.train();
	testLstmnet.setBatchInput("wsj-00-20.id.vocab10k", testsentenceNum, minLength, maxLength);
	testLstmnet.test();
}
};
int main(int argc, const char ** argv)
{
	//printf("using cuda");
    //lstm::test(true);//only test code,not predict
    lstm::trueMain(1);
    //lstm::test(false);
	//printf("not using cuda");
    //lstm::trueMain(false);
    return 0;
}

