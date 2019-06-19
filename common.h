#ifndef FBC_TENSORRT_TEST_COMMON_HPP_
#define FBC_TENSORRT_TEST_COMMON_HPP_


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <NvInfer.h>
#include <iostream>


#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)


class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) override
	{
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
};
#endif
