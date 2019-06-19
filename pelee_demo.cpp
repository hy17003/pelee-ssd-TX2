#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
#include <vector>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace std;
using namespace cv;

#define SHOW_IMAGE 1

// Network details
const char* gNetworkName = "pelee-ssd";       // Network name
static const int kINPUT_C = 3;          // Input image channels
static const int kINPUT_H = 304;        // Input image height
static const int kINPUT_W = 304;        // Input image width
static const int kOUTPUT_CLS_SIZE = 21; // Number of classes
static const int kKEEP_TOPK = 200;      // Number of total bboxes to be kept per image after NMS step. It is same as detection_output_param.keep_top_k in prototxt file


const std::string gCLASSES[kOUTPUT_CLS_SIZE] = {"background", "aeroplane", "bicycle", "bird", "boat",
                                                "bottle", "bus", "car", "cat", "chair", 
                                                "cow", "diningtable", "dog", "horse", "motorbike",
                                                "person", "pottedplant", "sheep", "sofa", "train",
                                                "tvmonitor"}; // List of class labels

static const char* kINPUT_BLOB_NAME = "data";            // Input blob name
static const char* kOUTPUT_BLOB_NAME0 = "detection_out"; // Output blob name
static const char* kOUTPUT_BLOB_NAME1 = "keep_count";    // Output blob name

Logger logger;
// Visualization
const float kVISUAL_THRESHOLD = 0.6f;


void caffeToTRTModel(const std::string& deployFile,           // Name for caffe prototxt
                     const std::string& modelFile,            // Name for model
                     const std::vector<std::string>& outputs, // Network outputs
                     unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
                     IHostMemory** trtModelStream)            // Output stream for the TensorRT model
{
    // Create the builder
    IBuilder* builder = createInferBuilder(logger);
    assert(builder != nullptr);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              nvinfer1::DataType::kHALF);
    // Specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(36 << 20);
    builder->setFp16Mode(true);
    builder->setInt8Mode(false);
    builder->allowGPUFallback(true);
    ICudaEngine* engine;
    engine = builder->buildCudaEngine(*network);
    assert(engine);
    network->destroy();
    parser->destroy();
    (*trtModelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(kINPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(kOUTPUT_BLOB_NAME0),
        outputIndex1 = engine.getBindingIndex(kOUTPUT_BLOB_NAME1);

    // Create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float))); // Data
    CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * kKEEP_TOPK * 7 * sizeof(float)));               // Detection_out
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * sizeof(int)));                                  // KeepCount (BBoxs left for each batch)

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * kKEEP_TOPK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

int main(int argc, char** argv)
{
    printf("start...\n");
    initLibNvInferPlugins(&logger, "");
    IHostMemory* trtModelStream{nullptr};
    // Create a TensorRT model from the caffe model and serialize it to a stream

    const int N = 1; // Batch size
    printf("load and transfer model...\n");
    caffeToTRTModel("./model/pelee.prototxt",
                    "./model/pelee_SSD_304x304_iter_120000.caffemodel",
                    std::vector<std::string>{kOUTPUT_BLOB_NAME0, kOUTPUT_BLOB_NAME1},
                    N, &trtModelStream);
    printf("read images...\n");
    std::vector<cv::String> imageList;
    cv::glob("./images/*.jpg", imageList);
    printf("get %d images\n", imageList.size());
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    trtModelStream->destroy();
    float* detectionOut = new float[N * kKEEP_TOPK * 7];
    int* keepCount = new int[N];
    
    for(int idx = 0;idx<imageList.size();idx++)
    {
        cv::Mat image = cv::imread(imageList[idx]);
        cv::Mat image_f;
        image.convertTo(image_f, CV_32FC3);
        cv::resize(image_f, image_f, cv::Size(kINPUT_W, kINPUT_H));
        float *data = new float[kINPUT_C * kINPUT_H * kINPUT_W];
        // float mean[3] = {104, 117, 123};
        float mean[3] = {127.5, 127.5, 127.5};
        for(int r = 0; r < kINPUT_H;r++)
        {
            for(int c = 0; c < kINPUT_W;c++)
            {
                for(int t = 0;t < kINPUT_C;t++)
                {
                    data[t * kINPUT_H * kINPUT_W + r * kINPUT_W + c] = (image_f.ptr<Vec3f>(r)[c][0] - mean[t]) * 0.007843;
                }
            }
        }
        double t1 = cv::getTickCount();
        doInference(*context, data, detectionOut, keepCount, N);
        double t2 = cv::getTickCount();
        std::cout << "time consume: " << (t2 - t1) * 1000 / cv::getTickFrequency() << " ms" << std::endl;
        for (int p = 0; p < N; ++p)
        {
            for (int i = 0; i < keepCount[p]; ++i)
            {
                float* det = detectionOut + (p * kKEEP_TOPK + i) * 7;
                if (det[2] < kVISUAL_THRESHOLD)
                    continue;
                float xmin = det[3] * image.cols;
                float ymin = det[4] * image.rows;
                float xmax = det[5] * image.cols;
                float ymax = det[6] * image.rows;
                std::string className = gCLASSES[(int) det[1]];                
                cv::rectangle(image, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 255, 0), 1);
                cv::putText(image, className, cv::Point(xmin + 5, ymin + 20), 1, 1.5, cv::Scalar(0, 0, 255));
            }
        }
        delete [] data;
        data = nullptr;
#if SHOW_IMAGE
        cv::imshow("image", image);
        cv::waitKey(0);
#endif
    }
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete[] detectionOut;
    delete[] keepCount;
    shutdownProtobufLibrary();
}
