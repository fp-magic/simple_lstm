#include"lstm.h"
using namespace nts;

namespace lstm
{
    
const int sentenceNum=2048;
const int maxLength=32;
const int minLength=16;
const int embSize=32;
const int wordNum=10000;
const int epochNum = 10;
const int unitNum = 32;
const int batchSize = 128;
float firstEmb[wordNum][wordNum];
int fileInputsTokens[sentenceNum][maxLength];
void trueMain(bool use_gpu)
{
    printf("w\n");
	//checkInput(trueInput);
	printf("e");
    lstmnet testLstmnet(use_gpu,wordNum,unitNum,embSize,epochNum,batchSize,0.01,"rand");
	testLstmnet.getInput("wsj-00-20.id.vocab10k", sentenceNum,minLength,maxLength);
    printf("r\n");
    testLstmnet.train();
}
};
int main(int argc, const char ** argv)
{
	//printf("using cuda");
    //lstm::test(true);//only test code,not predict
    lstm::trueMain(true);
    //lstm::test(false);
	//printf("not using cuda");
    //lstm::trueMain(false);
    return 0;
}

