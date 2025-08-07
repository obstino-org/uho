// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef VAD_H
#define VAD_H

#include "onnxruntime_cxx_api.h"

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

	SileroVAD() {
		try {
			options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "vad_inference");
			session = make_unique<Ort::Session>(env, L"silero_vad_opset15.onnx", options);
		}
		catch (const Ort::Exception& e) {
			std::cerr << "Error creating session: " << e.what() << std::endl;
		}

		resetStates();
	}

	float forward(vector <float>& input512) {
		// return 1.0;	// <-- for testing only

		std::vector<int64_t> inputSignalShape{ 1, 512 + 64 };	// 512 for signal, plus 64 for context
		std::vector<int64_t> inputStateShape{ 2, 1, 128 };
		std::vector<int64_t> inputSrShape{ 1 };
		std::vector<int64_t> outputProbabilityShape{ 1, 1 };
		std::vector<int64_t> outputStateShape{ 2, 1, 128 };

		Ort::RunOptions runOptions;
		vector<const char*> inputNames = { "input", "state", "sr" };
		vector<const char*> outputNames = { "output", "stateN" };

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

		session->Run(runOptions, inputNames.data(), inputTensors.data(), 3, outputNames.data(), outputTensors.data(), 2);

		state = outputState;
		context = vector<float>(input512.end() - context.size(), input512.end());

		return outputProbability[0];
	}
};

#endif