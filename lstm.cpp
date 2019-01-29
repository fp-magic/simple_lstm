#include"lstm.h"
using namespace nts;

namespace lstm
{
    
const int sentenceNum=1024;
const int maxLength=32;
const int minLength=16;
const int embSize=32;
float firstEmb[10010][embSize];
float fileInputs[sentenceNum][maxLength][embSize];
int fileInputsTokens[sentenceNum][maxLength];

XTensor trueInput;

void test(bool use_gpu)
{
    XTensor testInput;
    InitTensor3D(&testInput,sentenceNum,maxLength,embSize,X_FLOAT,use_gpu-1);
    _SetDataRand(&testInput,-1.0,1.0);
    lstmnet testLstmnet(use_gpu,&testInput,32,32,5,64);
    testLstmnet.train();
}
void getInput(XTensor& trueInput,bool use_gpu)
{
    int num = 0,snum=0, token;
    memset(firstEmb,0,sizeof(firstEmb));
    freopen("wsj-00-20.id.vocab10k", "r", stdin);
    while (scanf("%d", &token) != EOF)
    {
        if (token == 2)
        {
            if (snum < minLength)
            {
                for (int i = minLength; i < maxLength; i++)
                {
                    fileInputsTokens[num][i] = 2;
                    for (int j = 0; j < embSize; j++)
                        fileInputs[num][i][j] = firstEmb[2][j];
                }
            }printf("%d\n",num);
            snum = 0;
            num++;
            if (num > sentenceNum)
                break;
        }
        else if (snum < maxLength)
        {
            fileInputsTokens[num][snum] = token;
            for (int j = 0; j < embSize; j++)
                fileInputs[num][snum][j] = firstEmb[token][j];
            snum++;
        }
    }printf("wq\n");
    InitTensor3D(&trueInput,sentenceNum,maxLength,embSize,X_FLOAT,-1);
    printf("ww\n");
    trueInput.SetData(&fileInputs,sentenceNum*maxLength*embSize);
}
void trueMain(bool use_gpu)
{
    int minLength,maxLength,sentenceNum;
    printf("w\n");
    getInput(trueInput,use_gpu);
    printf("e\n");
    lstmnet testLstmnet(use_gpu,&trueInput,32,32,5,64);
    printf("r\n");
    testLstmnet.train();
}
};
int main(int argc, const char ** argv)
{
    printf("q");
    #ifdef USE_CUDA
    //lstm::test(true);//only test code,not predict
    lstm::trueMain(true);
    #else
    //lstm::test(false);
    lstm::trueMain(false);
    #endif
    //lstm::trueMain();
    return 0;
}

