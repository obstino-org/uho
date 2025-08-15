// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef VAD_H
#define VAD_H

#include <iostream>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"

using namespace std;

class SileroVAD {
	// built with help of https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
public:
	int audioChunkSize = 512;	// assumes 16kHz sample rate
	Ort::Env env;
	Ort::SessionOptions options;
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	
	unique_ptr<Ort::Session> session;
	
	vector<float> context = vector<float>(64, 0.0f);
	vector<float> state = vector<float>(2 * 128, 0.0f);

	void resetStates() {	
		for (int i = 0; i < context.size(); i++)
			context[i] = 0.0f;

		for (int i = 0; i < state.size(); i++)
			state[i] = 0.0f;
	}

	SileroVAD(AAssetManager *manager) {
		try {
			options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "vad_inference");

            vector<uint8_t> modelData = readAsset(manager, (char*)"silero_vad_opset15.onnx");
            session = make_unique<Ort::Session>(env, modelData.data(), modelData.size(), options);
            //session = make_unique<Ort::Session>(env, L"silero_vad_opset15.onnx", options);
		}
		catch (const Ort::Exception& e) {
			std::cerr << "Error creating session: " << e.what() << std::endl;
		}

		resetStates();
	}

    vector<uint8_t> readAsset(AAssetManager *manager, char *assetFileName) {
        AAsset *asset = AAssetManager_open(manager, assetFileName, AASSET_MODE_BUFFER);
        uint8_t *buffer = (uint8_t*)AAsset_getBuffer(asset);
        long len = AAsset_getLength(asset);
        vector<uint8_t> v(buffer, buffer + len);
        return v;
    }

	float forward(vector <float>& input512) {
		std::vector<int64_t> inputSignalShape{ 1, 512 + 64 };	// 512 for signal, plus 64 for context
		std::vector<int64_t> inputStateShape{ 2, 1, 128 };
		std::vector<int64_t> inputSrShape{ 1 };
		std::vector<int64_t> outputProbabilityShape{ 1, 1 };
		std::vector<int64_t> outputStateShape{ 2, 1, 128 };

		Ort::RunOptions runOptions;

        const char* inputNames[] = { "input", "state", "sr" };
        const char* outputNames[] = { "output", "stateN" };

		vector<float> input;

		vector<float> outputProbability(1);
		vector<float> outputState(2 * 128, 0.0f);
		vector<int64_t> sr = { 16000 };

		// Concatenate input with context
		input = context;
		input.insert(input.end(), input512.begin(), input512.end());

		vector<Ort::Value> inputTensors;
		vector<Ort::Value> outputTensors;
		outputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, outputProbability.data(), outputProbability.size(), outputProbabilityShape.data(), outputProbabilityShape.size()));
		outputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, outputState.data(), outputState.size(), outputStateShape.data(), outputStateShape.size()));

		inputTensors.emplace_back(Ort::Value::CreateTensor(memoryInfo, input.data(), input.size(), inputSignalShape.data(), inputSignalShape.size()));
		inputTensors.emplace_back(Ort::Value::CreateTensor(memoryInfo, state.data(), state.size(), inputStateShape.data(), inputStateShape.size()));
		inputTensors.emplace_back(Ort::Value::CreateTensor(memoryInfo, sr.data(), sr.size(), inputSrShape.data(), inputSrShape.size()));

        //session->Run(runOptions, inputNames.data(), inputTensors.data(), 3, outputNames.data(), outputTensors.data(), 2);
        session->Run(runOptions, inputNames, inputTensors.data(), 3, outputNames, outputTensors.data(), 2);

        state = outputState;
		context = vector<float>(input512.end() - context.size(), input512.end());

		return outputProbability[0];
	}
};

#endif