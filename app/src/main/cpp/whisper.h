// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.
// Credit goes to "https://github.com/nyadla-sys/whisper.cpp" for log_mel_spectrogram function along with dft, fft, filters_vocab_gen.bin usage and code

#ifndef UHO_WHISPER_H
#define UHO_WHISPER_H

//#define CTX6_BASE_MODEL
#define CTX10_BASE_MODEL
//#define TINY_MODEL

#include <jni.h>
#include <string>

#include <map>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <locale>
#include <codecvt>
#include <regex>
#include <deque>
#include <set>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

//#include <corecrt_math_defines.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include "gzip/compress.hpp"

using namespace std;

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_N_MEL       80
#define WHISPER_HOP_LENGTH  160

#ifdef CTX10_BASE_MODEL
#define WHISPER_CHUNK_SIZE  10
#elifdef CTX6_BASE_MODEL
#define WHISPER_CHUNK_SIZE  6
#else
#define WHISPER_CHUNK_SIZE  30
#endif

#define WHISPER_MEL_LEN     3000
#define WHISPER_STEP_MS 2700
#define WHISPER_LENGTH_MS (2*WHISPER_STEP_MS)

int special_tokens_start = 50257;
double maxComprRatio = 2.0;

vector<wstring> whisper_special_tokens = {
        L"<|endoftext|>",
        L"<|startoftranscript|>",
        L"<|en|>",
        L"<|zh|>",
        L"<|de|>",
        L"<|es|>",
        L"<|ru|>",
        L"<|ko|>",
        L"<|fr|>",
        L"<|ja|>",
        L"<|pt|>",
        L"<|tr|>",
        L"<|pl|>",
        L"<|ca|>",
        L"<|nl|>",
        L"<|ar|>",
        L"<|sv|>",
        L"<|it|>",
        L"<|id|>",
        L"<|hi|>",
        L"<|fi|>",
        L"<|vi|>",
        L"<|iw|>",
        L"<|uk|>",
        L"<|el|>",
        L"<|ms|>",
        L"<|cs|>",
        L"<|ro|>",
        L"<|da|>",
        L"<|hu|>",
        L"<|ta|>",
        L"<|no|>",
        L"<|th|>",
        L"<|ur|>",
        L"<|hr|>",
        L"<|bg|>",
        L"<|lt|>",
        L"<|la|>",
        L"<|mi|>",
        L"<|ml|>",
        L"<|cy|>",
        L"<|sk|>",
        L"<|te|>",
        L"<|fa|>",
        L"<|lv|>",
        L"<|bn|>",
        L"<|sr|>",
        L"<|az|>",
        L"<|sl|>",
        L"<|kn|>",
        L"<|et|>",
        L"<|mk|>",
        L"<|br|>",
        L"<|eu|>",
        L"<|is|>",
        L"<|hy|>",
        L"<|ne|>",
        L"<|mn|>",
        L"<|bs|>",
        L"<|kk|>",
        L"<|sq|>",
        L"<|sw|>",
        L"<|gl|>",
        L"<|mr|>",
        L"<|pa|>",
        L"<|si|>",
        L"<|km|>",
        L"<|sn|>",
        L"<|yo|>",
        L"<|so|>",
        L"<|af|>",
        L"<|oc|>",
        L"<|ka|>",
        L"<|be|>",
        L"<|tg|>",
        L"<|sd|>",
        L"<|gu|>",
        L"<|am|>",
        L"<|yi|>",
        L"<|lo|>",
        L"<|uz|>",
        L"<|fo|>",
        L"<|ht|>",
        L"<|ps|>",
        L"<|tk|>",
        L"<|nn|>",
        L"<|mt|>",
        L"<|sa|>",
        L"<|lb|>",
        L"<|my|>",
        L"<|bo|>",
        L"<|tl|>",
        L"<|mg|>",
        L"<|as|>",
        L"<|tt|>",
        L"<|haw|>",
        L"<|ln|>",
        L"<|ha|>",
        L"<|ba|>",
        L"<|jw|>",
        L"<|su|>",
        L"<|translate|>",
        L"<|transcribe|>",
        L"<|startoflm|>",
        L"<|startofprev|>",
        L"<|nocaptions|>",
        L"<|notimestamps|>",
};

int timestamp_token_start = special_tokens_start + whisper_special_tokens.size();
int timestamp_token_last = timestamp_token_start + 1500;

bool isTimestampToken(int tokenId) {
    return (tokenId >= timestamp_token_start && tokenId <= timestamp_token_last);
}

double getTokenTimestamp(int tokenId) {
    return (double)(tokenId - timestamp_token_start) * 0.02;
}

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

whisper_filters g_filters;

struct whisper_mel {
    int n_len;
    int n_mel;

    std::vector<float> data;
};

// fastExp by Johan Rade https://gist.github.com/jrade/293a73f89dfef51da6522428c857802d
inline float fastExp(float x)
{
    constexpr float a = (1 << 23) / 0.69314718f;
    constexpr float b = (1 << 23) * (127 - 0.043677448f);
    x = a * x + b;

    // Remove these lines if bounds checking is not needed
    constexpr float c = (1 << 23);
    constexpr float d = (1 << 23) * 255;
    if (x < c || x > d)
        x = (x < c) ? 0.0f : d;

    // With C++20 one can use std::bit_cast instead
    uint32_t n = static_cast<uint32_t>(x);
    memcpy(&x, &n, 4);
    return x;
}

int gzipRawSize(int gzipSize) {
    int headerSize = 10;
    int footerSize = 8;
    return gzipSize - headerSize - footerSize;
}

double gzipCompressionRatio(std::string orig, std::string compr)
{
    return (double)orig.length() / (double)gzipRawSize(compr.length());
}

// return peak gzip compression ratio
// -- returns index of string, where ratio is highest
pair<double, int> peakGzipCompressionRatio(string s) {
    double peakCompressionRatio = 0.0;
    int chunkSize = 1; //3; // (changed from 3 to 1 on 28.7.2025)
    int maxIdx = 0;
    std::reverse(s.begin(), s.end());
    for (int i = 0; i < s.length(); i += chunkSize) {
        string subStr = s.substr(0, i);
        string subStrZip = gzip::compress(subStr.c_str(), subStr.length());
        double compressionRatio = gzipCompressionRatio(subStr, subStrZip);
        //printf("Compression ratio: %.2f\n", compressionRatio);
        if (compressionRatio > peakCompressionRatio) {
            peakCompressionRatio = compressionRatio;
            maxIdx = i;
        }
    }

    maxIdx = s.length() - 1 - maxIdx;
    return make_pair(peakCompressionRatio, maxIdx);
}

class Whisper {
public:
    Ort::Env env;
    unique_ptr<Ort::Session> encoder;
    unique_ptr <Ort::Session> decoder;
    Ort::SessionOptions options;

    int32_t numVocab = 51865;
    //std::map <int, wstring> id_to_vocab;
    std::map <int, string> mapIdToUTF8;

    Whisper(AAssetManager *manager)
    {
        /*OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);*/

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "Creating env");
        env = Ort::Env(
                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,//OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                "whisper_inference");
        //options.SetLogSeverityLevel(0);

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "Adding nnapi");
        uint32_t nnapi_flags = 0;
        nnapi_flags |= NNAPI_FLAG_USE_FP16; //| NNAPI_FLAG_CPU_DISABLED; (dont use this CPU DISABLED flag because it might cause crash)
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(options, nnapi_flags));

        options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "Reading assets");
#ifdef TINY_MODEL
        vector<uint8_t> encoderData = readAsset(manager, (char*)"tiny_encoder.onnx");
        vector<uint8_t> decoderData = readAsset(manager, (char*)"tiny_decoder.onnx");
#elifdef CTX10_BASE_MODEL
        vector<uint8_t> encoderData = readAsset(manager, (char*)"ctx10encoder.onnx");
        vector<uint8_t> decoderData = readAsset(manager, (char*)"ctx10decoder.onnx");
#elifdef CTX6_BASE_MODEL
        vector<uint8_t> encoderData = readAsset(manager, (char*)"ctx6encoder.onnx");
        vector<uint8_t> decoderData = readAsset(manager, (char*)"ctx6decoder.onnx");
#else
        vector<uint8_t> encoderData = readAsset(manager, (char*)"kv_encoder.onnx");
        vector<uint8_t> decoderData = readAsset(manager, (char*)"kv_decoder.onnx");
//        vector<uint8_t> encoderData = readAsset(manager, (char*)"kv_encoder.quant.onnx");
//        vector<uint8_t> decoderData = readAsset(manager, (char*)"kv_decoder.quant.onnx");
#endif

        encoder = make_unique<Ort::Session>(Ort::Session(env, encoderData.data(), encoderData.size(), options));
        decoder = make_unique<Ort::Session>(Ort::Session(env, decoderData.data(), decoderData.size(), options));

        // Read log-Mel filters
        vector<uint8_t> filtersVect = readAsset(manager, (char*)"filters_vocab_gen.bin");
        uint8_t *pFilters = &filtersVect[0];
        int offset = sizeof(uint32_t);  // skip magic
        g_filters.n_mel = *(int32_t*)((uint64_t)pFilters + offset);
        offset += sizeof(g_filters.n_mel);
        g_filters.n_fft = *(int32_t*)((uint64_t)pFilters + offset);
        offset += sizeof(g_filters.n_fft);
        float *filtersBegin = (float*)((uint64_t)pFilters + offset);
        g_filters.data = vector<float>(filtersBegin, filtersBegin + g_filters.n_mel * g_filters.n_fft);

        // Read vocab json file
        vector<uint8_t> vocabVect = readAsset(manager, (char*)"vocab.json");
        string strVocab = string((char*)vocabVect.data(), vocabVect.size());
        stringstream fVocab(strVocab);
        string line;
        fVocab >> line;
        while (true) {
            string utf8Chars;
            string comma;
            int id;
            fVocab >> utf8Chars;
            fVocab >> id;
            fVocab >> comma;

            if (comma == "}" || fVocab.eof())
                break;

            string utf8CharsFinal;
            for (int i = 1; i < utf8Chars.length() - 2; i++) {
                if (utf8Chars[i] == '\\')	// skip backslash
                    continue;
                utf8CharsFinal += utf8Chars[i];
            }

            mapIdToUTF8[id] = utf8CharsFinal;
        }
    }

    static wstring utf8TextToWstring(string utf8TextASR) {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        wstring textASR = converter.from_bytes(utf8TextASR);

        // Slovenian noise sounds (šumniki)
        map<wstring, wstring> decodeMappings = {
                {L"Äį", L"č"},
                {L"Äĩ", L"ć"},
                {L"ÄĮ", L"Č"},
                {L"ÄĨ", L"Ć"},
                {L"Å¡", L"š"},
                {L"Åł", L"Š"},
                {L"Å¾", L"ž"},
                {L"Å½", L"Ž"},
                {L"Äĳ", L"đ"},
                {L"ÄĲ", L"Đ"},
                {L"Ġ", L" "}
        };

        for (auto m : decodeMappings) {
            textASR = std::regex_replace(textASR, std::wregex(m.first), m.second);
        }

        return textASR;
    }

    vector<uint8_t> readAsset(AAssetManager *manager, char *assetFileName) {
        __android_log_print(ANDROID_LOG_INFO, "UHO1", "Opening asset with name %s", assetFileName);
        AAsset *asset = AAssetManager_open(manager, assetFileName, AASSET_MODE_BUFFER);
        if(asset == nullptr)
            __android_log_print(ANDROID_LOG_INFO, "UHO1", "asset is NULL");
        else
            __android_log_print(ANDROID_LOG_INFO, "UHO1", "asset is OK");

        uint8_t *buffer = (uint8_t*)AAsset_getBuffer(asset);
        long len = AAsset_getLength(asset);
        vector<uint8_t> v(buffer, buffer + len);
        AAsset_close(asset);
        return v;
    }

    std::string testInferenceKV(whisper_mel& mel, int maxSeqLen=200) {
        std::string strOutputUTF8;

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // RUN ENCODER
        std::vector<int64_t> encoderInputDims = { 1, mel.n_mel, mel.n_len };
        size_t encoderInputTensorSize = mel.n_len * mel.n_mel;
        std::vector<Ort::Value> encoderInputTensor;
        encoderInputTensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, mel.data.data(), encoderInputTensorSize, encoderInputDims.data(), encoderInputDims.size()));

#ifdef TINY_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 1500, 384 };
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 500, 512 };
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 300, 512 };
#else
        std::vector<int64_t> encoderOutputShape = { 1, 1500, 512 };
#endif
        std::vector<Ort::Value> encoderOutputTensor;
        std::vector<float> encoderOutputTensorValues(vectorProduct(encoderOutputShape));
        encoderOutputTensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, encoderOutputTensorValues.data(), encoderOutputTensorValues.size(), encoderOutputShape.data(), encoderOutputShape.size()));

        const char* encInputName[] = { "mel" };   // log-Mel spectrogram
        const char* encOutputName[] = { "audio" };  // latent (feature) representation of audio

        encoder->Run(Ort::RunOptions{ nullptr }, encInputName, encoderInputTensor.data(), 1, encOutputName, encoderOutputTensor.data(), 1);
        // At this point, "encoderOutputTensor" and "encoderOutputTensorValues" contain the latent representation of audio (x_audio)

        // RUN DECODER
#ifdef TINY_MODEL
        std::vector<int64_t> prevSelfAttnShape = {8, 1, 0, 384};    // initial dimensions (later its (8, 1, numInTokens, 384))
        std::vector<int64_t> prevCrossAttnShape = {8, 1, 0, 384};   // initial dimensions (later its (8, 1, 1500, 384))
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> prevSelfAttnShape = {12, 1, 0, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 0, 512};   // initial dimensions (later its (12, 1, 1500, 512))
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> prevSelfAttnShape = {12, 1, 0, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 0, 512};   // initial dimensions (later its (12, 1, 1500, 512))
#else
        std::vector<int64_t> prevSelfAttnShape = {12, 1, 0, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 0, 512};   // initial dimensions (later its (12, 1, 1500, 512))
#endif
        std::vector<Ort::Value> decoderInputTensors;
        const char* decInputNames[] = { "tokens", "audio", "cache_self_attn", "cache_cross_attn" };

        std::vector<int64_t> inputIds = { 50258, 50305, 50359 };
        std::vector<int64_t> inputIdsShape = { 1, 3 };

        int step = 3;
        int vocabSize = 51865;
        std::vector<Ort::Value> decoderOutputTensors;
        const char* decOutputNames[] = { "logits", "new_cache_self_attn", "new_cache_cross_attn" };

        //std::vector<float> selfAttnInputTensorValues(vectorProduct(selfAttnShape));
        //std::vector<float> crossAttnInputTensorValues(vectorProduct(crossAttnShape));
        size_t prevSelfAttnInputTensorSize = vectorProduct(prevSelfAttnShape);
        float *prevSelfAttnInputTensorValues = new float [prevSelfAttnInputTensorSize];

        size_t prevCrossAttnInputTensorSize = vectorProduct(prevCrossAttnShape);
        float *prevCrossAttnInputTensorValues = new float [prevCrossAttnInputTensorSize];

        while(true) {
            auto t0 = chrono::high_resolution_clock::now();
            // prepare input tensors
            decoderInputTensors.clear();
            decoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputIds.data(), inputIds.size(), inputIdsShape.data(), inputIdsShape.size()));
            decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, encoderOutputTensorValues.data(), encoderOutputTensorValues.size(), encoderOutputShape.data(), encoderOutputShape.size()));
            decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, prevSelfAttnInputTensorValues, prevSelfAttnInputTensorSize, prevSelfAttnShape.data(), prevSelfAttnShape.size()));
            decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, prevCrossAttnInputTensorValues, prevCrossAttnInputTensorSize, prevCrossAttnShape.data(), prevCrossAttnShape.size()));

            // now prepare output tensors
            decoderOutputTensors.clear();
            std::vector<int64_t> decoderOutputLogitsShape = { 1, (step==3)?3:1, vocabSize };
#ifdef TINY_MODEL
            std::vector<int64_t> outSelfAttnShape = {8, 1, step, 384};
            std::vector<int64_t> outCrossAttnShape = {8, 1, 1500, 384};
#elifdef CTX10_BASE_MODEL
            std::vector<int64_t> outSelfAttnShape = {12, 1, step, 512};
            std::vector<int64_t> outCrossAttnShape = {12, 1, 500, 512};
#elifdef CTX6_BASE_MODEL
            std::vector<int64_t> outSelfAttnShape = {12, 1, step, 512};
            std::vector<int64_t> outCrossAttnShape = {12, 1, 300, 512};
#else
            std::vector<int64_t> outSelfAttnShape = {12, 1, step, 512};
            std::vector<int64_t> outCrossAttnShape = {12, 1, 1500, 512};
#endif
            std::vector<float> decoderOutputLogits(vectorProduct(decoderOutputLogitsShape));
            size_t decoderOutputSelfAttnSize = vectorProduct(outSelfAttnShape);
            float *decoderOutputSelfAttn = new float [decoderOutputSelfAttnSize];
            size_t decoderOutputCrossAttnSize = vectorProduct(outCrossAttnShape);
            float *decoderOutputCrossAttn = new float [decoderOutputCrossAttnSize];

            decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputLogits.data(), decoderOutputLogits.size(), decoderOutputLogitsShape.data(), decoderOutputLogitsShape.size()));
            decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputSelfAttn, decoderOutputSelfAttnSize, outSelfAttnShape.data(), outSelfAttnShape.size()));
            decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputCrossAttn, decoderOutputCrossAttnSize, outCrossAttnShape.data(), outCrossAttnShape.size()));
            // inference
            decoder->Run(Ort::RunOptions{ nullptr }, decInputNames, decoderInputTensors.data(), 4, decOutputNames, decoderOutputTensors.data(), 3);

            // prepare values for next iteration
            int lastToken = greedyDecodeToken(decoderOutputLogits.data(), (step==3)?2:0, vocabSize);
            inputIds = { lastToken };
            inputIdsShape = {1, 1};

            delete [] prevSelfAttnInputTensorValues;
            prevSelfAttnInputTensorValues = decoderOutputSelfAttn;
            prevSelfAttnInputTensorSize = decoderOutputSelfAttnSize;
            prevSelfAttnShape = outSelfAttnShape;

            delete [] prevCrossAttnInputTensorValues;
            prevCrossAttnInputTensorValues = decoderOutputCrossAttn;
            prevCrossAttnInputTensorSize = decoderOutputCrossAttnSize;
            prevCrossAttnShape = outCrossAttnShape;

            // Create an empty encoder values (IMPORTANT! without this inference is very slow)
            encoderOutputTensorValues = std::vector<float>();
#ifdef TINY_MODEL
            encoderOutputShape = { 1, 0, 384 };
#else
            encoderOutputShape = { 1, 0, 512 };
#endif

            int special_tokens_start = 50257;
            if(lastToken >= special_tokens_start) {
                break;
            }
            else {
                strOutputUTF8 += mapIdToUTF8[lastToken];
            }

            step++;

            auto t1 = chrono::high_resolution_clock::now();
            //strOutputUTF8 += to_string(chrono::duration_cast<chrono::milliseconds>(t1-t0).count()) + "...";
        }

        return strOutputUTF8;
    }

    // ##########################################################################################################
    float computeTokenProbability(int tokenIndex, int tokenId, float* decoderOutput, float temperature)
    {
        float softmaxDenominator = 0.0;
        float softmaxNumerator = 0.0;
        int numTokens = 51865;
        // tokenIndex is index in sequence of output tokens
        // tokenId is the id of that token on the specified tokenIndex

        softmaxNumerator = exp(decoderOutput[tokenIndex * numTokens + tokenId] / temperature);
        for (int i = 0; i < numTokens; i++) {
            float logit = decoderOutput[tokenIndex * numTokens + i];
            softmaxDenominator += fastExp(logit / temperature);
        }

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "decoderOutput[...]=%.2f, softmaxNumerator=%.2f, denominator=%.2f", (double)decoderOutput[tokenIndex * numTokens + tokenId], (double)softmaxNumerator, (double)softmaxDenominator);
        return softmaxNumerator / softmaxDenominator;
    }

    void runEncoder(whisper_mel& mel, std::vector<float> & encoderOutputTensorValues) {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // RUN ENCODER
        std::vector<int64_t> encoderInputDims = { 1, mel.n_mel, mel.n_len };
        size_t encoderInputTensorSize = mel.n_len * mel.n_mel;
        std::vector<Ort::Value> encoderInputTensor;
        encoderInputTensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, mel.data.data(), encoderInputTensorSize, encoderInputDims.data(), encoderInputDims.size()));

#ifdef TINY_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 1500, 384 };
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 500, 512 };
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 300, 512 };
#else
        std::vector<int64_t> encoderOutputShape = { 1, 1500, 512 };
#endif
        std::vector<Ort::Value> encoderOutputTensor;
        encoderOutputTensorValues = std::vector<float>(vectorProduct(encoderOutputShape)); //std::vector<float> encoderOutputTensorValues(vectorProduct(encoderOutputShape));
        encoderOutputTensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, encoderOutputTensorValues.data(), encoderOutputTensorValues.size(), encoderOutputShape.data(), encoderOutputShape.size()));

        const char* encInputName[] = { "mel" };   // log-Mel spectrogram
        const char* encOutputName[] = { "audio" };  // latent (feature) representation of audio

        encoder->Run(Ort::RunOptions{ nullptr }, encInputName, encoderInputTensor.data(), 1, encOutputName, encoderOutputTensor.data(), 1);
    }

    class DecoderState {
    public:
        float *decoderOutputSelfAttn;
        size_t decoderOutputSelfAttnSize;   // size means number of elements

        float *decoderOutputCrossAttn;
        size_t decoderOutputCrossAttnSize;  // size means number of elements

        DecoderState(float *_decoderOutputSelfAttn, size_t _decoderOutputSelfAttnSize,
                     float *_decoderOutputCrossAttn, size_t _decoderOutputCrossAttnSize):
                decoderOutputSelfAttn(_decoderOutputSelfAttn), decoderOutputSelfAttnSize(_decoderOutputSelfAttnSize),
                decoderOutputCrossAttn(_decoderOutputCrossAttn), decoderOutputCrossAttnSize(_decoderOutputCrossAttnSize)
        {}

        DecoderState(const DecoderState& other) {
            // Make a deep copy

            this->decoderOutputSelfAttn = new float[other.decoderOutputSelfAttnSize];
            memcpy((void*)(this->decoderOutputSelfAttn),
                   (void*)(other.decoderOutputSelfAttn),
                   other.decoderOutputSelfAttnSize * sizeof(float));
            this->decoderOutputSelfAttnSize = other.decoderOutputSelfAttnSize;

            this->decoderOutputCrossAttn = new float[other.decoderOutputCrossAttnSize];
            memcpy((void*)(this->decoderOutputCrossAttn),
                   (void*)(other.decoderOutputCrossAttn),
                   other.decoderOutputCrossAttnSize * sizeof(float));
            this->decoderOutputCrossAttnSize = other.decoderOutputCrossAttnSize;
        }

        ~DecoderState() {
            delete [] decoderOutputSelfAttn;
            delete [] decoderOutputCrossAttn;
        }
    };

    class DecoderOutput {
    public:
        float *decoderOutputLogits;
        size_t decoderOutputLogitsSize; // size means number of elements

        DecoderOutput(float *_decoderOutputLogits, size_t _decoderOutputLogitsSize):
                decoderOutputLogits(_decoderOutputLogits),
                decoderOutputLogitsSize(_decoderOutputLogitsSize) {}

        DecoderOutput(const DecoderOutput & other) {
            decoderOutputLogits = new float[other.decoderOutputLogitsSize];
            decoderOutputLogitsSize = other.decoderOutputLogitsSize;
            memcpy((void*)decoderOutputLogits, (void*)other.decoderOutputLogits, other.decoderOutputLogitsSize * sizeof(float));
        }

        ~DecoderOutput() {
            delete [] decoderOutputLogits;
        }
    };

    pair<shared_ptr<DecoderState>, shared_ptr<DecoderOutput>> decoderForwardPass(vector<float> &encoderOutputTensorValues, std::vector<size_t> basicPromptIds)
    {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

#ifdef TINY_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 1500, 384 };
        std::vector<int64_t> prevSelfAttnShape = {8, 1, 0, 384};    // initial dimensions (later its (8, 1, numInTokens, 384))
        std::vector<int64_t> prevCrossAttnShape = {8, 1, 0, 384};   // initial dimensions (later its (8, 1, 1500, 384))
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 500, 512 };
        std::vector<int64_t> prevSelfAttnShape = {12, 1, 0, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 0, 512};   // initial dimensions (later its (12, 1, 500, 512))
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 300, 512 };
        std::vector<int64_t> prevSelfAttnShape = {12, 1, 0, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 0, 512};   // initial dimensions (later its (12, 1, 300, 512))
#else
        std::vector<int64_t> encoderOutputShape = { 1, 1500, 512 };
        std::vector<int64_t> prevSelfAttnShape = {12, 1, 0, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 0, 512};   // initial dimensions (later its (12, 1, 1500, 512))
#endif
        std::vector<Ort::Value> decoderInputTensors;
        const char* decInputNames[] = { "tokens", "audio", "cache_self_attn", "cache_cross_attn" };

        std::vector<int64_t> inputIds = { 50258, 50305, 50359 }; // <|startoftranscript|><|sl|><|transcribe|>
        if(basicPromptIds.size() > 0)
            inputIds.insert(inputIds.end(), basicPromptIds.begin(), basicPromptIds.end());

        std::vector<int64_t> inputIdsShape = { 1, static_cast<long>(inputIds.size()) };

        int step = inputIds.size();
        int vocabSize = 51865;
        std::vector<Ort::Value> decoderOutputTensors;
        const char* decOutputNames[] = { "logits", "new_cache_self_attn", "new_cache_cross_attn" };

        //std::vector<float> selfAttnInputTensorValues(vectorProduct(selfAttnShape));
        //std::vector<float> crossAttnInputTensorValues(vectorProduct(crossAttnShape));
        size_t prevSelfAttnInputTensorSize = vectorProduct(prevSelfAttnShape);
        float *prevSelfAttnInputTensorValues = new float [prevSelfAttnInputTensorSize];

        size_t prevCrossAttnInputTensorSize = vectorProduct(prevCrossAttnShape);
        float *prevCrossAttnInputTensorValues = new float [prevCrossAttnInputTensorSize];

        auto t0 = chrono::high_resolution_clock::now();
        // prepare input tensors
        decoderInputTensors.clear();
        decoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputIds.data(), inputIds.size(), inputIdsShape.data(), inputIdsShape.size()));
        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, encoderOutputTensorValues.data(), encoderOutputTensorValues.size(), encoderOutputShape.data(), encoderOutputShape.size()));
        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, prevSelfAttnInputTensorValues, prevSelfAttnInputTensorSize, prevSelfAttnShape.data(), prevSelfAttnShape.size()));
        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, prevCrossAttnInputTensorValues, prevCrossAttnInputTensorSize, prevCrossAttnShape.data(), prevCrossAttnShape.size()));

        // now prepare output tensors
        decoderOutputTensors.clear();
        std::vector<int64_t> decoderOutputLogitsShape = { 1, static_cast<long>(inputIds.size()), vocabSize };
#ifdef TINY_MODEL
        std::vector<int64_t> outSelfAttnShape = {8, 1, static_cast<long>(inputIds.size()), 384};
        std::vector<int64_t> outCrossAttnShape = {8, 1, 1500, 384};
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> outSelfAttnShape = {12, 1, static_cast<long>(inputIds.size()), 512};
        std::vector<int64_t> outCrossAttnShape = {12, 1, 500, 512};
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> outSelfAttnShape = {12, 1, static_cast<long>(inputIds.size()), 512};
        std::vector<int64_t> outCrossAttnShape = {12, 1, 300, 512};
#else
        std::vector<int64_t> outSelfAttnShape = {12, 1, static_cast<long>(inputIds.size()), 512};
        std::vector<int64_t> outCrossAttnShape = {12, 1, 1500, 512};
#endif

        //std::vector<float> decoderOutputLogits(vectorProduct(decoderOutputLogitsShape));
        size_t decoderOutputLogitsSize = vectorProduct(decoderOutputLogitsShape);
        float *decoderOutputLogits = new float [decoderOutputLogitsSize];

        size_t decoderOutputSelfAttnSize = vectorProduct(outSelfAttnShape);
        float *decoderOutputSelfAttn = new float [decoderOutputSelfAttnSize];
        size_t decoderOutputCrossAttnSize = vectorProduct(outCrossAttnShape);
        float *decoderOutputCrossAttn = new float [decoderOutputCrossAttnSize];

        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputLogits, decoderOutputLogitsSize, decoderOutputLogitsShape.data(), decoderOutputLogitsShape.size()));
        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputSelfAttn, decoderOutputSelfAttnSize, outSelfAttnShape.data(), outSelfAttnShape.size()));
        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputCrossAttn, decoderOutputCrossAttnSize, outCrossAttnShape.data(), outCrossAttnShape.size()));
        // inference
        decoder->Run(Ort::RunOptions{ nullptr }, decInputNames, decoderInputTensors.data(), 4, decOutputNames, decoderOutputTensors.data(), 3);

        shared_ptr<DecoderState> decoderState = make_shared<DecoderState>(
                decoderOutputSelfAttn, decoderOutputSelfAttnSize,
                decoderOutputCrossAttn, decoderOutputCrossAttnSize
        );

        shared_ptr<DecoderOutput> decoderOutput = make_shared<DecoderOutput>(
                decoderOutputLogits, decoderOutputLogitsSize
        );

        // clean up and return
        delete [] prevSelfAttnInputTensorValues;
        delete [] prevCrossAttnInputTensorValues;

        return make_pair(decoderState, decoderOutput);
    }

    int greedyDecodeToken(float* tensor, int idx, int numVocab) {
        int maxId = 0;
        float maxLogit = std::numeric_limits<float>::lowest();
        for (int i = 0; i < numVocab; i++) {
            float logit = tensor[numVocab * idx + i];
            if (logit > maxLogit) {
                maxLogit = logit;
                maxId = i;
            }
        }

        return maxId;
    }

    std::vector<int> beamDecodeToken(float* tensor, int idx, int numVocab, int beamSize, double temperature) {
        float* logitsPtr = (float*)&tensor[numVocab * idx];
        std::vector<int> kLargestTokens;

        for (int i = 0; i < beamSize; i++) {
            int largestToken = 0;
            float largestLogit = std::numeric_limits<float>::lowest();
            for (int j = 0; j < numVocab; j++) {
                if (std::find(kLargestTokens.begin(), kLargestTokens.end(), j) != kLargestTokens.end())
                    continue;

                if (logitsPtr[j] > largestLogit) {
                    largestLogit = logitsPtr[j];
                    largestToken = j;
                }
            }

            kLargestTokens.push_back(largestToken);
        }

        return kLargestTokens;
    }

    class BeamStructure {
    public:
        // Essential elements used in inferenceStepThread()

        int fromBeamNumber;
        bool deepCopyPending;   // whether or not the decoderState needs to be deep copied

        shared_ptr<DecoderState> decoderState;
        int lastOutTokenId;
        int nextStep;
        shared_ptr<DecoderOutput> lastOutput;

        bool dontPopBack = false;

        // Essential elements used for updateBeams() and pruneBeans()
        vector<size_t> ids;         // new tokens
        vector<double> probs;       // new tokens raw probabilities
        vector<double> probsTemp;   // temperature-corrected new tokens probabilities       [currently not used!]
        vector<double> probsPrompt; // prompt token probabilities (size = prompt.size())    [currently not used!]
        double logProb; //  [currently not used]

        // defualt constructor
        BeamStructure()
        { }

        // copy constructor
        BeamStructure(BeamStructure& b) {
            this->ids = b.ids;
            this->probs = b.probs;
            this->probsTemp = b.probsTemp;
            this->probsPrompt = b.probsPrompt;
            this->logProb = b.logProb;

            this->decoderState = b.decoderState;
            this->lastOutTokenId = b.lastOutTokenId;
            this->nextStep = b.nextStep;

            //this->lastOutput = b.lastOutput;
            // make a deep copy of output
            this->lastOutput = make_shared<DecoderOutput>(*(b.lastOutput));
        }
    };

    vector<shared_ptr<BeamStructure>> beams;

    void updateBeamPromptProbabilities(shared_ptr<Whisper::BeamStructure> beam, vector<size_t> &prompt, shared_ptr<DecoderOutput> decOut) {
        if (prompt.size() == 0)
            return;

        int numTokens;

        numTokens = decOut->decoderOutputLogitsSize / numVocab;

        int startIndex = 2; // TODO: adjust based on prompt with special tokens
        float* logits = (float*)decOut->decoderOutputLogits;

        for (int i = startIndex; i < numTokens - 1; i++) {
            double prob = (double)computeTokenProbability(i, prompt[i - startIndex], logits, 1.0);
            beam->probsPrompt.push_back(prob);
            //__android_log_print(ANDROID_LOG_INFO, "UHO2", "Prompt probability is %.2f", prob);
        }
    }

    bool bestBeamCompressionRatioCheck(Whisper& whisper, vector<size_t> &prompt, shared_ptr<Whisper::BeamStructure> beam0, int& highComprIdx)
    {
        //return true; // TESTING ONLY

        // (NEW - 27.7.2025) MERGED PROMPT WITH NEW IDS
        vector<size_t> ids = prompt;
        ids.insert(ids.end(), beam0->ids.begin(), beam0->ids.end());
        // returns false if there's high compression ratio, otherwise if all is okay it returns true
        string s1 = "";
        for (int i = 0; i < ids.size(); i++) {
            std::string tmp = whisper.mapIdToUTF8[ids[i]];
            for (char c : tmp)
                s1 += c;
        }

        double peakComprRatio;
        int peakStrIndex, peakTokenIndex;
        pair<double, int> gzipOutput = peakGzipCompressionRatio(s1);
        peakComprRatio = gzipOutput.first;
        peakStrIndex = gzipOutput.second;

        if (peakComprRatio >= maxComprRatio) {
            // From peak compression ratio's index inside merged string (prompt+new ids),
            // look at which token index this corresponds to. This way we can truncate poor transcription.
            string s2 = "";
            for (int i = 0; i < ids.size(); i++) {
                std::string tmp = whisper.mapIdToUTF8[ids[i]];
                for (char c : tmp)
                    s2 += c;
                if(s2.length() >= peakStrIndex) {
                    peakTokenIndex = i;
                    break;
                }
            }
            highComprIdx = std::max(peakTokenIndex - (int)prompt.size(), 0);    // we won't pop from prompt, so we limit values to a range

            __android_log_print(ANDROID_LOG_INFO, "UHO2", "Best beam compression ratio (%.2f) is poor. PeakStrIndex=%i\nText is \"%s\"\nNew text is \"%s\"", peakComprRatio, peakStrIndex, s1.c_str(), s2.c_str());
            return false;
        }
        return true;
    }

    // Original algorithm designed to detect repetitive (single- or multi-word) hallucinations
    // It does this by defining a window of last N tokens, and computing N/power(set(N_tokens)),
    // For example, if it gets N_tokens=[100,200,300,100,200,300] then power(set(N_tokens))=power({100,200,300})=3,
    // in which case "compression" ratio is 6/3 = 2.0. N is varied from 2 to at most len(prompt+newIds).
    tuple<int, double, int, string> detectHallucinations(vector<size_t>& prompt, vector<size_t>& newIds) {
        string hallucinationString = "";
        int hallucinationFingerprint = 0;
        double peakRatio = 1.0;
        int peakIndex = -1, newIdsPeakIndex = -1;

        vector<size_t> seq = prompt;
        seq.insert(seq.end(), newIds.begin(), newIds.end());

        for (int i = seq.size() - 2; i >= 0; i--) {
            vector<int> tmp1 = vector<int>(seq.begin() + i, seq.end());
            set<int> tmp2 = set<int>(seq.begin() + i, seq.end());
            double ratio = (double)tmp1.size() / (double)tmp2.size();
            //printf("tmp1/tmp2 = %.2f\n", ratio);
            if (ratio >= peakRatio) {
                peakRatio = ratio;
                peakIndex = i;
            }
        }
        //printf("peak index = %i", peakIndex);

        if (peakIndex != -1) {
            for(int i = peakIndex; i < seq.size(); i++) {
                hallucinationFingerprint += seq[i];
                hallucinationString += mapIdToUTF8[seq[i]];
            }
            newIdsPeakIndex = std::max((int)peakIndex - (int)prompt.size(), 0);
        }

        return make_tuple(newIdsPeakIndex, peakRatio, hallucinationFingerprint, hallucinationString);
    }

    // Returns the Levenshtein distance between word1 and word2 (source: https://github.com/guilhermeagostinelli/levenshtein/blob/master/levenshtein.cpp)
    int levenshteinDist(string word1, string word2) {
        int size1 = word1.size();
        int size2 = word2.size();
        int verif[size1 + 1][size2 + 1]; // Verification matrix i.e. 2D array which will store the calculated distance.

        // If one of the words has zero length, the distance is equal to the size of the other word.
        if (size1 == 0)
            return size2;
        if (size2 == 0)
            return size1;

        // Sets the first row and the first column of the verification matrix with the numerical order from 0 to the length of each word.
        for (int i = 0; i <= size1; i++)
            verif[i][0] = i;
        for (int j = 0; j <= size2; j++)
            verif[0][j] = j;

        // Verification step / matrix filling.
        for (int i = 1; i <= size1; i++) {
            for (int j = 1; j <= size2; j++) {
                // Sets the modification cost.
                // 0 means no modification (i.e. equal letters) and 1 means that a modification is needed (i.e. unequal letters).
                int cost = (word2[j - 1] == word1[i - 1]) ? 0 : 1;

                // Sets the current position of the matrix as the minimum value between a (deletion), b (insertion) and c (substitution).
                // a = the upper adjacent value plus 1: verif[i - 1][j] + 1
                // b = the left adjacent value plus 1: verif[i][j - 1] + 1
                // c = the upper left adjacent value plus the modification cost: verif[i - 1][j - 1] + cost
                verif[i][j] = min(
                        min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                        verif[i - 1][j - 1] + cost
                );
            }
        }

        // The last position of the matrix will contain the Levenshtein distance.
        return verif[size1][size2];
    }

    double stringSimilarity(string s1, string s2) {
        if(s1.empty() || s2.empty())
            return 0.0;

        return (1.0 - (double)levenshteinDist(s1, s2)/(double)std::max(s1.length(), s2.length()));
    }

    vector<int> findHighestRankTokens(float *logits, int rankTokensSize) {
        vector<int> out;
        float lastMaxLogit = std::numeric_limits<float>::max();

        for(int i = 0; i < rankTokensSize; i++) {
            float currMaxLogit = std::numeric_limits<float>::lowest();
            int currMaxLogitIndex = 0;
            for(int j = 0; j < numVocab; j++) {
                if(logits[j] > currMaxLogit && logits[j] < lastMaxLogit) {
                    currMaxLogit = logits[j];
                    currMaxLogitIndex = j;
                }
            }
            lastMaxLogit = currMaxLogit;
            out.push_back(currMaxLogitIndex);
        }

        return out;
    }

    shared_ptr<Whisper::BeamStructure> additiveInference(whisper_mel &mel, vector<size_t> &prompt, int beamSize=4, float temperature=1.0f, int groupLogProb=5, double groupLogP_thresh=-2.0)
    {
        // Token that shall be suppressed when executing "additiveInference" next time.
        // This is useful, because sometimes we get repeated hallcinations, as in "on je bil bil bil bil"
        // The 4th token "bil" should now be suppressed in next call to additiveInference, so that we move on with decoding!
        static int suppressNextTime = 0;
        int firstRankToken1Index = -1;

        static int prevHallucFingerprint = 0;
        static string prevHallucString = "";

        int maxSeqLen = 75;

        vector<float> encOut;

        auto t0 = chrono::high_resolution_clock::now();
        runEncoder(mel, encOut);
        auto t1 = chrono::high_resolution_clock::now();
        wprintf(L"Encoder forward pass took %i ms\n", chrono::duration_cast<chrono::milliseconds>(t1 - t0).count());

        auto decFwdOut = decoderForwardPass(encOut, prompt);
        shared_ptr<DecoderState> decState = decFwdOut.first;
        shared_ptr<DecoderOutput> decOut = decFwdOut.second;
        shared_ptr<Whisper::BeamStructure> beam = make_shared<Whisper::BeamStructure>();

        /*
        // NEW CODE HERE (removes low probability IDs and does another decoder forward pass with modified prompt!)
        // (EXPERIMENTAL -- doesn't yield best results, it seems that prompting with old tokens is still better)
        updateBeamPromptProbabilities(beam, prompt, decOut);
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW FOR PROMPT) prompt.size()=%i, probsPrompt.size()=%i", prompt.size(), beam->probsPrompt.size());
        double sumLogProb = 0.0;
        int groupSize = 5;
        while (prompt.size() >= groupSize) {
            double groupLogP = 0.0;
            for (int j = 0; j < groupSize; j++)
                groupLogP += (double)log(beam->probsPrompt[j]);
            if (groupLogP <= groupLogP_thresh) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW FOR PROMPT) Poping from prompt, groupLogP=%.2f < groupLogP_thresh=%.2f", groupLogP, groupLogP_thresh);
                prompt.erase(prompt.begin());
            }
            else {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW FOR PROMPT) Not poping anymore");
                break;
            }
        }
        decFwdOut = decoderForwardPass(encOut, prompt);
        decState = decFwdOut.first;
        decOut = decFwdOut.second;
        beam->probsPrompt.clear();
        */

        beam->decoderState = decState;
        beam->nextStep = prompt.size() + 3; // +2
        beam->lastOutTokenId = (prompt.size() > 0) ? prompt[prompt.size() - 1] : 50359; // (50359 == <|transcribe|>)
        beam->logProb = 0.0;
        beams.push_back(beam);

        for (int i = 0; i < maxSeqLen; i++) {
            vector<thread> threads(beams.size());

            auto t0 = chrono::high_resolution_clock::now();
            for (int j = 0; j < beams.size(); j++)
                threads[j] = thread(&Whisper::inferenceStepThread, this, beams[j], std::ref(encOut));
            for (int j = 0; j < beams.size(); j++)
                threads[j].join();

            // suppress first tokens
            vector<int> tokensToSuppress = {220, 50256, 50257, 13, 2411, 1097, 485, 7}; // 220 = "blank_token", 50257 = "EOT", 13 = ".", 2411 = " .", 1097 = " ...", 485 = "...", 7 = "("
            //if(suppressNextTime > 0)
            //    tokensToSuppress.push_back(suppressNextTime);

            if(i == 0) {
                for (int tok: tokensToSuppress) {
                    for(int j = 0; j < beams.size(); j++) {
                        float *logitsPtr = beams[j]->lastOutput->decoderOutputLogits;
                        logitsPtr[tok] = std::numeric_limits<float>::lowest();
                    }
                }
            }

            if(suppressNextTime != 0) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "Yes we have to suppress token %i", suppressNextTime);
                for(int j = 0; j < beams.size(); j++) {
                    int tok = suppressNextTime;
                    float *logitsPtr = beams[j]->lastOutput->decoderOutputLogits;
                    logitsPtr[tok] = std::numeric_limits<float>::lowest();
                }
                suppressNextTime = 0;   // reset token to suppress, so that it doesn't get suppressed in subsequent calls to our function
            }

            updateBeams(beamSize, temperature);
            pruneBeams(beamSize);

            // token 220 is " " (space), whereas 11 is "," (comma) -- when we have {..., 11, 220, ...}, we continue decoding
            bool isFirstToken = (i == 0); //(prompt.empty() && i == 0);
            if(isTimestampToken(beams[0]->lastOutTokenId) ||
                (beams[0]->lastOutTokenId == 220 && (isFirstToken?true:(beams[0]->ids[beams[0]->ids.size() - 2] != 11))) ||
                (beams[0]->probs[beams[0]->probs.size()-1] < 0.1 && !isFirstToken))    // end decoding if token probability is *really* low (and its not the first token)
            {
                //__android_log_print(ANDROID_LOG_INFO, "UHO2", "ADIN DEBUG 1");
                // we also end if we detect token 220
                int tmp = beams[0]->lastOutTokenId;
                int tmpProb = beams[0]->probs[beams[0]->probs.size()-1];
                //__android_log_print(ANDROID_LOG_INFO, "UHO2", "ADIN DEBUG 2");

                beams[0]->ids.pop_back();
                beams[0]->probs.pop_back();
                beams[0]->probsTemp.pop_back();
                //__android_log_print(ANDROID_LOG_INFO, "UHO2", "ADIN DEBUG 3");

                beams[0]->lastOutTokenId = special_tokens_start;
                beams[0]->ids.push_back(special_tokens_start);
                beams[0]->probs.push_back(1.0);
                beams[0]->probsTemp.push_back(1.0);
                //__android_log_print(ANDROID_LOG_INFO, "UHO2", "ADIN DEBUG 4");

                /*if(tmpProb < 0.1)
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "EXTREMELY LOW PROBABILITY TOKEN %i, breaking", tmp);
                else
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "SPECIAL TOKEN %i, breaking", tmp);*/
                break;
            }

            if (beams[0]->lastOutTokenId == special_tokens_start) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "EOT TOKEN, breaking");
                break;
            }

            // UNCOMMENT THE FOLLOWING FOR HALLUCINATION PREVENTION
            // check sum of log probabilities only once, because beginnings of hallucinations are often low prob
            /*double sumLogProb = 0.0;
            if ((i+1) == groupLogProb) {
                for(int j = 0; j < groupLogProb; j++)
                    sumLogProb += (double)log(beams[0]->probs[j]);

                if (sumLogProb <= groupLogP_thresh) {
                    for(int u = 0; u < beams[0]->probs.size(); u++)
                        __android_log_print(ANDROID_LOG_INFO, "UHO2", "LOW PROB ID %i (%s): %.2f\n", beams[0]->ids[u], mapIdToUTF8[beams[0]->ids[u]].c_str(), beams[0]->probs[u]);
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "EARLY EXIT DUE TO LOW PROB (hallucinations?) sum=%.2f\n", sumLogProb);
                    beams[0]->ids = {};
                    beams[0]->probs = {};
                    beams[0]->probsTemp = {};
                    break;
                }
            }*/

            //if((i+1) >= 2) { // if at least two tokens have been decoded
            //    beams[0]-
            //}
            /*
            int numEqual = 1;
            int lastId = beams[0]->ids[beams[0]->ids.size()-1];
            for(int j = beams[0]->ids.size()-2; j >= 0; j--) {
                int tmpId = beams[0]->ids[beams[0]->ids.size()-j];
                if(tmpId == lastId)
                    numEqual++;
                else
                    break;
            }
            if(numEqual == 2) {
                float *logitsPtr = beams[0]->lastOutput->decoderOutputLogits;
                logitsPtr[lastId] = std::numeric_limits<float>::lowest();
                int altToken = greedyDecodeToken(logitsPtr, 0, numVocab);
                beams[0]->lastOutTokenId = altToken;
                beams[0]->ids[beams[0]->ids.size()-1] = altToken;
                beams[0]->probs[beams[0]->probs.size() - 1] = (double)computeTokenProbability(0, altToken, logitsPtr, 1.0);
                // todo: update probsTemp and logProb (although they're currently not in use in code)
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "Detected repeats 2. Setting last token to %i (%s)", altToken, mapIdToUTF8[altToken].c_str());
            }*/

            /*
            // NEW CODE TO DETECT _simple_ HALLUCINATIONS (3 repeated last tokens)
            int numRepeatDetection = 3;
            if((i+1) >= numRepeatDetection) {
                bool lastTokensEqual = true;
                int lastId = beams[0]->ids[beams[0]->ids.size()-1];
                for(int j = beams[0]->ids.size()-2; j >=0; j--) {
                    if(lastId != beams[0]->ids[j])
                        lastTokensEqual = false;
                    else
                        break;
                }

                if(lastTokensEqual == true) {
                    //suppressNextTime = lastId;

                    for(int j = 0; j < numRepeatDetection - 1; j++) {   // TODO: numRepeatDetection-1 because 1 token is good
                        beams[0]->ids.pop_back();
                        beams[0]->probs.pop_back();
                        beams[0]->probsTemp.pop_back();
                    }

                    beams[0]->lastOutTokenId = special_tokens_start;
                    beams[0]->ids.push_back(special_tokens_start);
                    beams[0]->probs.push_back(1.0);
                    beams[0]->probsTemp.push_back(1.0);

                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) DETECTED REPEATED HALLUCINATION in last few tokens -- Removing and BREAKING\n");
                    break;
                }
            }*/

            /*
            // NEW EXPERIMENTAL CODE (27.7.2025) -- not for production
            int rankTokensSize = 50;
            vector<int> rankTokens;
            rankTokens = findHighestRankTokens(beams[0]->lastOutput->decoderOutputLogits, rankTokensSize);
            for(int j = 0; j < rankTokens.size(); j++) {
                if(rankTokens[j] == 220) { //isTimestampToken(rankTokens[j])) {
                    double p220 = computeTokenProbability(0, 220, beams[0]->lastOutput->decoderOutputLogits, 1.0);
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) BLANK TOKEN 220 with rank %i (p=%.2f)", j, p220);
                    break;
                }
            }

            if(rankTokens[1] == 220) {
                if(firstRankToken1Index == -1)
                    firstRankToken1Index = i;
            }

            if(rankTokens[1] == 220) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) BREAKING DECODING, because 220 has rank 1");
                //beams[0]->ids.pop_back();
                //beams[0]->probs.pop_back();
                //beams[0]->probsTemp.pop_back();
                beams[0]->lastOutTokenId = special_tokens_start;
                beams[0]->ids.push_back(special_tokens_start);
                beams[0]->probs.push_back(1.0);
                beams[0]->probsTemp.push_back(1.0);
                break;
            }*/

            // If last token is " ..." (1097), we break. Only hallucinations seem to happen this way.
            if(beams[0]->lastOutTokenId == 1097) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) BREAKING DECODING, found \" ...\"");
                beams[0]->ids.pop_back();
                beams[0]->probs.pop_back();
                beams[0]->probsTemp.pop_back();

                beams[0]->lastOutTokenId = special_tokens_start;
                beams[0]->ids.push_back(special_tokens_start);
                beams[0]->probs.push_back(1.0);
                beams[0]->probsTemp.push_back(1.0);
                break;
            }

            // NEW CODE TO DETECT HALLUCINATIONS (21.7.2025): detects for last groupLogProb tokens
            double sumLogProb = 0.0;
            if ((i+1) >= groupLogProb) {
                for(int j = 0; j < groupLogProb; j++)
                    sumLogProb += (double)log(beams[0]->probs[beams[0]->probs.size() - 1 - j]);

                if (sumLogProb <= groupLogP_thresh) {
                    //for(int u = 0; u < beams[0]->probs.size(); u++)
                    //    __android_log_print(ANDROID_LOG_INFO, "UHO2", "LOW PROB ID %i (%s): %.2f\n", beams[0]->ids[u], mapIdToUTF8[beams[0]->ids[u]].c_str(), beams[0]->probs[u]);
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) DETECTED LOW PROB in last few tokens -- sum=%.2f\n", sumLogProb);

                    // This will signalize to next part of code to reset audio
                    beams[0]->ids = {};
                    beams[0]->probs = {};
                    beams[0]->probsTemp = {};

                    /*for(int j = 0; j < groupLogProb && beams[0]->ids.size() > 0; j++) {
                        beams[0]->ids.pop_back();
                        beams[0]->probs.pop_back();
                        beams[0]->probsTemp.pop_back();
                    }

                    beams[0]->lastOutTokenId = special_tokens_start;
                    beams[0]->ids.push_back(special_tokens_start);
                    beams[0]->probs.push_back(1.0);
                    beams[0]->probsTemp.push_back(1.0);*/
                    break;
                }
            }

            // What follows is original code using our "detectHallucinations" function to detect repetitive hallucinations.
            // Read description of detectHallucinations function above for more info on how the function detects a sort of "compression ratio".
            // Features:
            //  - first we compute promptComprRatio, i.e. peak compression ratio of prompt (looking at last N tokens of prompt, where we vary N from 2 to len(prompt))
            //  - then, similarly, we compute comprRatio, i.e. peak compression ratio of merged prompt with best beam newly decoded ids (beams[0]->ids) that follow prompt (again looking at last N tokens, varying N from 2 to len(mergedTokens))
            //  - if comprRatio is high enough (e.g. >= 1.75), this is a red flag (e.g. comprRatio=2.0 would indicate 2 repetitions of same tokens)
            //      - function detectHallucination will give us the string that was hallucinated
            //          --> this string is then compared using the hallucinated string from previous calls to "additiveInference".
            //              When this previous string was e.g. an empty string "", function stringSimilarity will return a low value,
            //              which will lead to execution of "else" statement that removes hallucinations and adds "<|endoftext|>" (special_tokens_start).
            //              When hallucination extends even into the prompt,, peakComprIndex will be 0, which will remove all tokens that we
            //              currently decoded (because its all hallucinatory, even extending into prompt).
            //      - if there was a hallucination detected in previous iteration (i-1) and is again detected in current (i)
            //          then stringSimilarity will return a higher value (but always between 0 and 1), and we will assume
            //          that it's a false detection of hallucination, as it is unlikely that for 2 different audios (ensured by VAD)
            //          we would have the same hallucination

            vector<size_t> tmpEmpty = {};
            tuple<int, double, int, string> detectOutPrompt = detectHallucinations(prompt, tmpEmpty);
            double promptComprRatio = get<1>(detectOutPrompt);

            tuple<int, double, int, string> detectOut = detectHallucinations(prompt, beams[0]->ids);
            int peakComprIndex = get<0>(detectOut);
            double comprRatio = get<1>(detectOut);
            int hallucFingerprint = get<2>(detectOut);
            string hallucString = get<3>(detectOut);
            int hallucLength = (int)beams[0]->ids.size() - peakComprIndex;

            // *prompt* can't have higher compression ratio than *text with prompt*, because repeat-hallucinations don't stop repeating
            // -- if it stops repeating _after_ prompt, then apparently decoding is successful, not hallucinatory.
            if(comprRatio >= 1.75 && hallucLength >= 6 && (promptComprRatio < comprRatio))
            {
                double strSim = stringSimilarity(hallucString, prevHallucString);
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "DETECTED HIGH COMPRESSION (%.2f AT PEAK INDEX %i, fingerprint %i, strSim %.2f). Rectifying and breaking.", comprRatio, peakComprIndex, hallucFingerprint, strSim);
                //if(hallucFingerprint == prevHallucFingerprint) {
                if(strSim >= 0.70 && comprRatio < 5.0) {
                    // We can afford to let falsely detected hallucination to pass, because speech is recognized in 2 iterations always
                    // with _new_ speech audio (this is guaranteed by VAD). Hallucinations for 2 different audios on same place are unlikely.
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "--> HOWEVER, high-compression strings in last 2 iterations are 'equal' (sim=%.2f). FALSELY DETECTED HALLUCINATION. In any case, BREAKING", strSim);
                    beams[0]->dontPopBack = true;

                    beams[0]->lastOutTokenId = special_tokens_start;
                    beams[0]->ids.push_back(special_tokens_start);
                    beams[0]->probs.push_back(1.0);
                    beams[0]->probsTemp.push_back(1.0);

                    prevHallucString = hallucString;

                    break;
                } else {
                    //prevHallucFingerprint = hallucFingerprint;
                    prevHallucString = hallucString;

                    // remove only repeated part of hallucination (e.g. "je bilo je bilo" becomes "je bilo")
                    if (peakComprIndex > 0)
                        peakComprIndex = peakComprIndex + (int) (((double) beams[0]->ids.size() - (double) peakComprIndex) / (double)comprRatio);

                    for (int j = i; j >= peakComprIndex && j >= 0; j--) {
                        beams[0]->ids.pop_back();
                        beams[0]->probs.pop_back();
                        beams[0]->probsTemp.pop_back();
                    }

                    beams[0]->lastOutTokenId = special_tokens_start;
                    beams[0]->ids.push_back(special_tokens_start);
                    beams[0]->probs.push_back(1.0);
                    beams[0]->probsTemp.push_back(1.0);
                    break;
                }
            }

            // gzip compression ratio code; currently not in use, blocked by "0 == 1" condition in if statement
            int peakComprTokenIndex = 0;
            if (0 == 1 && bestBeamCompressionRatioCheck(*this, prompt, beams[0], peakComprTokenIndex) == false) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "BEST BEAM COMPRESSION RATIO IS DANGEROUS, break-ing");
                //__android_log_print(ANDROID_LOG_INFO, "UHO2", "(ALSO last token probability = %.2f)", (double)beams[0]->probs[beams[0]->probs.size()-1]);

                // (PRE-27.7.2025 CODE)
                //beams[0]->ids = {};
                //beams[0]->probs = {};
                //beams[0]->probsTemp = {};

                // (27.7.2025 TESTING CODE)
                /*if(firstRankToken1Index == -1) {
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "CAN'T RECTIFY. breaking");
                    beams[0]->ids = {};
                    beams[0]->probs = {};
                    beams[0]->probsTemp = {};
                } else {
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATTEMPTED RECTIFICATION. breaking");
                    for(int j = i; j >= (firstRankToken1Index-1) && j >= 0; j--) {
                        beams[0]->ids.pop_back();
                        beams[0]->probs.pop_back();
                        beams[0]->probsTemp.pop_back();
                    }

                    beams[0]->lastOutTokenId = special_tokens_start;
                    beams[0]->ids.push_back(special_tokens_start);
                    beams[0]->probs.push_back(1.0);
                    beams[0]->probsTemp.push_back(1.0);
                    break;
                }*/

                // (28.7.2025 TESTING CODE)
                // peakComprTokenIndex is index inside newly decoded id's, which marks the beginning of hallucinations
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATTEMPTED RECTIFICATION#2 (peakComprTokenIndex=%i). breaking", peakComprTokenIndex);
                for(int j = i; j >= (peakComprTokenIndex-1) && j >= 0; j--) {
                    beams[0]->ids.pop_back();
                    beams[0]->probs.pop_back();
                    beams[0]->probsTemp.pop_back();
                }
                beams[0]->lastOutTokenId = special_tokens_start;
                beams[0]->ids.push_back(special_tokens_start);
                beams[0]->probs.push_back(1.0);
                beams[0]->probsTemp.push_back(1.0);
                break;
            }

            if (i == maxSeqLen-1)
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "\n\nWOW GOT THROUGH ALL TOKENS?\n\n");

            auto t1 = chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "Single inference step took %lli ms (%s, p=%.2f)\n", chrono::duration_cast<chrono::milliseconds>(t1-t0).count(), mapIdToUTF8[beams[0]->lastOutTokenId].c_str(), beams[0]->probs[beams[0]->probs.size()-1]);
        }

        /*
        for (int i = 0; i < beams.size(); i++) {
            shared_ptr<Whisper::BeamStructure> beam = beams[i];
            for (int j = 0; j < beam->ids.size(); j++)
                wprintf(L"%s@%.2f ", id_to_vocab[beam->ids[j]].c_str(), beam->timestampsIds[j]);
            wprintf(L"\n");
        }
        */

        // add probabilities for original prompt tokens to the beam we return
        updateBeamPromptProbabilities(beams[0], prompt, decOut);

        return beams[0];
    }

    void updateBeams(int beamSize, double temperature)
    {
        // update beam info...
        vector<shared_ptr<BeamStructure>> newBeams;

        for (int i = 0; i < beams.size(); i++) {
            //int numVocab = beams[i]->lastOutput.shape()[1];

            // beamDecodeToken takes about 0.5ms
            vector<int> beamTokens = beamDecodeToken((float*)beams[i]->lastOutput->decoderOutputLogits, 0, numVocab, beamSize, temperature);
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "beamTokens[0]=%i", beamTokens[0]);
            //vector<int> beamTokens = beamDecodeToken((float*)beams[i]->lastOutput.buffer(), 0, numVocab, beamSize, temperature);

            // for each of "numBeams", create numBeams new beams, totaling numBeams^2 new beams (they replace old 'beams')
            for (int j = 0; j < beamSize; j++) {
                shared_ptr<BeamStructure> newBeam;

                newBeam = make_shared<BeamStructure>(*(beams[i].get()));
                newBeam->nextStep++;
                newBeam->lastOutTokenId = beamTokens[j];
                newBeam->ids.push_back(beamTokens[j]);
                newBeam->fromBeamNumber = i;

                float* logits = (float*)beams[i]->lastOutput->decoderOutputLogits;

                // computing both prob and probTemp totals in about 0.5ms
                double prob = (double)computeTokenProbability(0, beamTokens[j], logits, 1.0);   // cpu intensive (almost 0.5ms)
                double probTemp = (double)computeTokenProbability(0, beamTokens[j], logits, temperature);   // cpu intensive (almost 0.5ms)

                __android_log_print(ANDROID_LOG_INFO, "UHO2", "updateBeams probability %.2f", prob);

                newBeam->probs.push_back(prob);
                newBeam->probsTemp.push_back(probTemp);
                newBeam->logProb += log(probTemp);

                newBeams.push_back(newBeam);
            }
        }

        beams = newBeams;
    }

    void pruneBeams(int numBeams) {
        vector<int> beamsRequringDeepCopy;  // perfrom beam copy only if necessary

        std::sort(beams.begin(), beams.end(), [](shared_ptr<BeamStructure> const& b1, shared_ptr<BeamStructure> const& b2) -> bool { return b1->logProb > b2->logProb; });
        beams = vector<shared_ptr<BeamStructure>>(beams.begin(), beams.begin() + numBeams);

        for (shared_ptr<BeamStructure>& beam : beams)
        {
            if (std::find(beamsRequringDeepCopy.begin(), beamsRequringDeepCopy.end(), beam->fromBeamNumber) == beamsRequringDeepCopy.end())
                beamsRequringDeepCopy.push_back(beam->fromBeamNumber);
            else {
                //wprintf(L"\tRequiring copy = %i\n", beam->fromBeamNumber);
                // Make a deep copy of decoder state

                shared_ptr<DecoderState> copyDecoderState;
                copyDecoderState = make_shared<DecoderState>(*(beam->decoderState));
                beam->decoderState = copyDecoderState;

                //beam->decoderState = make_shared<DecoderState>(*(beam->decoderState));

                //beam->decoderState = make_shared<ctranslate2::layers::DecoderState>(*(beam->decoderState));
            }
        }
    }

    // single step thread
    void inferenceStepThread(shared_ptr<BeamStructure> beam, vector<float>& encoderOutput)
    {
        int tid = gettid();
        setpriority(PRIO_PROCESS, tid, -10);

        auto stepOutput = inferenceStep(beam->decoderState, beam->lastOutTokenId, beam->nextStep);
        beam->decoderState = stepOutput.first;
        beam->lastOutput = stepOutput.second;
    }

    // single step
    pair<shared_ptr<DecoderState>, shared_ptr<DecoderOutput>> inferenceStep(shared_ptr<DecoderState> decoderState, int32_t startId, int step)
    {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // VERY IMPORTANT (without empty encoder input, the inference is very slow)
        auto encoderOutputTensorValues = std::vector<float>();
#ifdef TINY_MODEL
        std::vector<int64_t> encoderOutputShape = { 1, 0, 384 };
#else
        std::vector<int64_t> encoderOutputShape = { 1, 0, 512 };
#endif
        // RUN DECODER

        // TODO: Make sure this "step" is correct parameter for input
#ifdef TINY_MODEL
        std::vector<int64_t> prevSelfAttnShape = {8, 1, step, 384};    // initial dimensions (later its (8, 1, numInTokens, 384))
        std::vector<int64_t> prevCrossAttnShape = {8, 1, 1500, 384};   // initial dimensions (later its (8, 1, 1500, 384))
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> prevSelfAttnShape = {12, 1, step, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 500, 512};   // initial dimensions (later its (12, 1, 500, 512))
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> prevSelfAttnShape = {12, 1, step, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 300, 512};   // initial dimensions (later its (12, 1, 300, 512))
#else   // base model
        std::vector<int64_t> prevSelfAttnShape = {12, 1, step, 512};    // initial dimensions (later its (12, 1, numInTokens, 512))
        std::vector<int64_t> prevCrossAttnShape = {12, 1, 1500, 512};   // initial dimensions (later its (12, 1, 1500, 512))
#endif
        std::vector<Ort::Value> decoderInputTensors;
        const char* decInputNames[] = { "tokens", "audio", "cache_self_attn", "cache_cross_attn" };

        //std::vector<int64_t> inputIds = { 50258, 50305, 50359 };
        //std::vector<int64_t> inputIdsShape = { 1, 3 };
        std::vector<int64_t> inputIds = { startId };
        std::vector<int64_t> inputIdsShape = {1, 1};

        //int step = 3;
        int vocabSize = numVocab; //51865;
        std::vector<Ort::Value> decoderOutputTensors;
        const char* decOutputNames[] = { "logits", "new_cache_self_attn", "new_cache_cross_attn" };

        //std::vector<float> selfAttnInputTensorValues(vectorProduct(selfAttnShape));
        //std::vector<float> crossAttnInputTensorValues(vectorProduct(crossAttnShape));

        //size_t prevSelfAttnInputTensorSize = vectorProduct(prevSelfAttnShape);
        //float *prevSelfAttnInputTensorValues = new float [prevSelfAttnInputTensorSize];

        //size_t prevCrossAttnInputTensorSize = vectorProduct(prevCrossAttnShape);
        //float *prevCrossAttnInputTensorValues = new float [prevCrossAttnInputTensorSize];

        // prepare input tensors
        decoderInputTensors.clear();
        decoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, inputIds.data(), inputIds.size(), inputIdsShape.data(), inputIdsShape.size()));
        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, encoderOutputTensorValues.data(), encoderOutputTensorValues.size(), encoderOutputShape.data(), encoderOutputShape.size()));

        //decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, prevSelfAttnInputTensorValues, prevSelfAttnInputTensorSize, prevSelfAttnShape.data(), prevSelfAttnShape.size()));
        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderState->decoderOutputSelfAttn, decoderState->decoderOutputSelfAttnSize, prevSelfAttnShape.data(), prevSelfAttnShape.size()));

        //decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, prevCrossAttnInputTensorValues, prevCrossAttnInputTensorSize, prevCrossAttnShape.data(), prevCrossAttnShape.size()));
        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderState->decoderOutputCrossAttn, decoderState->decoderOutputCrossAttnSize, prevCrossAttnShape.data(), prevCrossAttnShape.size()));

        // now prepare output tensors
        decoderOutputTensors.clear();
        std::vector<int64_t> decoderOutputLogitsShape = {1, 1, vocabSize }; //{ 1, (step==3)?3:1, vocabSize };
#ifdef TINY_MODEL
        std::vector<int64_t> outSelfAttnShape = {8, 1, step + 1, 384};
        std::vector<int64_t> outCrossAttnShape = {8, 1, 1500, 384};
#elifdef CTX10_BASE_MODEL
        std::vector<int64_t> outSelfAttnShape = {12, 1, step + 1, 512};
        std::vector<int64_t> outCrossAttnShape = {12, 1, 500, 512};
#elifdef CTX6_BASE_MODEL
        std::vector<int64_t> outSelfAttnShape = {12, 1, step + 1, 512};
        std::vector<int64_t> outCrossAttnShape = {12, 1, 300, 512};
#else
        std::vector<int64_t> outSelfAttnShape = {12, 1, step + 1, 512};
        std::vector<int64_t> outCrossAttnShape = {12, 1, 1500, 512};
#endif
        size_t decoderOutputLogitsSize = vectorProduct(decoderOutputLogitsShape);
        float *decoderOutputLogits = new float[decoderOutputLogitsSize];

        size_t decoderOutputSelfAttnSize = vectorProduct(outSelfAttnShape);
        float *decoderOutputSelfAttn = new float [decoderOutputSelfAttnSize];
        size_t decoderOutputCrossAttnSize = vectorProduct(outCrossAttnShape);
        float *decoderOutputCrossAttn = new float [decoderOutputCrossAttnSize];

        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputLogits, decoderOutputLogitsSize, decoderOutputLogitsShape.data(), decoderOutputLogitsShape.size()));
        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputSelfAttn, decoderOutputSelfAttnSize, outSelfAttnShape.data(), outSelfAttnShape.size()));
        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, decoderOutputCrossAttn, decoderOutputCrossAttnSize, outCrossAttnShape.data(), outCrossAttnShape.size()));
        // inference
        try {
            decoder->Run(Ort::RunOptions{ nullptr }, decInputNames, decoderInputTensors.data(), 4, decOutputNames, decoderOutputTensors.data(), 3);
        } catch (const Ort::Exception& e) {
            //std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ONNX exception: %s", e.what());
        }

        // prepare values for next iteration
        shared_ptr<DecoderState> outDecoderState = make_shared<DecoderState>(
                decoderOutputSelfAttn, decoderOutputSelfAttnSize,
                decoderOutputCrossAttn, decoderOutputCrossAttnSize
        );
        shared_ptr<DecoderOutput> outDecoderOutput = make_shared<DecoderOutput>(
                decoderOutputLogits,
                decoderOutputLogitsSize
        );

        return make_pair(outDecoderState, outDecoderOutput);
    }

    // ##########################################################################################################

    whisper_mel computeSpectrogramFromBuff(vector<float> buff)
    {
        whisper_mel mel = { 0 };
        buff = PreemphasizeAndNormalize(buff);    // no need to normalize, we did it already in getNextStreamMel
        buff.resize(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE, 0);
        if (!log_mel_spectrogram(buff.data(), (int)buff.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, 4, g_filters, mel)) {
            puts("error with spectrogram");
            return mel;
        }
        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "First mel data is %.2f", (double)mel.data[0]);
        return mel;
    }

    whisper_mel computeSpectrogram(AAssetManager *manager, string filePath)
    {
        whisper_mel mel = { 0 };

        //vector<short> pcmShort;
        vector<float> pcmFloat;
        //uint32_t numEl;
        //uint32_t sampleRate;

        //ReadWaveIntoBuffer(filePath.c_str(), pcmShort, &numEl, &sampleRate, 0);
        //for (int i = 0; i < numEl; i++)
        //    pcmFloat.push_back((float)pcmShort[i] / 32768.0);
        pcmFloat = readWaveAsset(manager, (char*)filePath.c_str());
        pcmFloat.resize(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE, 0);

        if (!log_mel_spectrogram(pcmFloat.data(), (int)pcmFloat.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, 4, g_filters, mel)) {
            puts("error with spectrogram");
            return mel;
        }
        puts("done computing");
        return mel;
    }

    vector<float> readWaveAsset(AAssetManager *manager, char *assetFileName) {
        AAsset *asset = AAssetManager_open(manager, assetFileName, AASSET_MODE_BUFFER);
        uint8_t *buffer = (uint8_t*)AAsset_getBuffer(asset);
        long len = AAsset_getLength(asset);

        // skip 44 bytes (headers and everything), assume we have 16kHz mono WAV file
        short *v = (short*)((unsigned long)buffer + 44);
        unsigned long vSize = (len - 44)/2;

        // convert to floating point
        vector<float> ret;
        for(int i = 0; i < vSize; i++)
            ret.push_back((float)v[i] / 32768.0f);

        return ret;
    }

    vector<float> PreemphasizeAndNormalize(vector<float>& in)
    {
        float filter_factor = 0.95f;
        vector<float> dout;

        int num = in.size();
        float dmax;

        dout = vector<float>(num);

        dmax = 0.0f;
        for (int k = 1; k < (int)num; k++) {
            dout[k] = (float)in[k] - filter_factor * (float)in[k - 1];
            if (fabs(dout[k]) > dmax)
                dmax = fabs(dout[k]);
        }

        for (int k = 0; k < (int)num; k++)
            dout[k] = (float)(dout[k] / dmax);

        return dout;
    }

    void dft(const std::vector<float>& in, std::vector<float>& out) {
        int N = (int)in.size();

        out.resize(N * 2);

        for (int k = 0; k < N; k++) {
            float re = 0;
            float im = 0;

            for (int n = 0; n < N; n++) {
                float angle = (float)(2 * M_PI * k * n / (float)N);
                re += in[n] * cos(angle);
                im -= in[n] * sin(angle);
            }

            out[k * 2 + 0] = re;
            out[k * 2 + 1] = im;
        }
    }

    void fft(const std::vector<float>& in, std::vector<float>& out) {
        out.resize(in.size() * 2);

        int N = (int)in.size();

        if (N == 1) {
            out[0] = in[0];
            out[1] = 0;
            return;
        }

        if (N % 2 == 1) {
            dft(in, out);
            return;
        }

        std::vector<float> even;
        std::vector<float> odd;

        for (int i = 0; i < N; i++) {
            if (i % 2 == 0) {
                even.push_back(in[i]);
            }
            else {
                odd.push_back(in[i]);
            }
        }

        std::vector<float> even_fft;
        std::vector<float> odd_fft;

        fft(even, even_fft);
        fft(odd, odd_fft);

        for (int k = 0; k < N / 2; k++) {
            float theta = (float)(2.0f * M_PI * k / (float)N);

            float re = cos(theta);
            float im = -sin(theta);

            float re_odd = odd_fft[2 * k + 0];
            float im_odd = odd_fft[2 * k + 1];

            out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
            out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

            out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
            out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
        }
    }

    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L92-L124
    bool log_mel_spectrogram(
            const float* samples,
            const int n_samples,
            const int sample_rate,
            const int fft_size,
            const int fft_step,
            const int n_mel,
            const int n_threads,
            const whisper_filters& filters,
            whisper_mel& mel) {

        // Hanning window
        std::vector<float> hann;
        hann.resize(fft_size);
        for (int i = 0; i < fft_size; i++) {
            hann[i] = (float)(0.5 * (1.0 - cos((2.0 * M_PI * i) / (fft_size))));
        }

        mel.n_mel = n_mel;
        mel.n_len = (n_samples) / fft_step;
        mel.data.resize(mel.n_mel * mel.n_len);

        const int n_fft = 1 + fft_size / 2;

        //printf("%s: n_samples = %d, n_len = %d\n", __func__, n_samples, mel.n_len);
        //printf("%s: recording length: %f s\n", __func__, (float) n_samples/sample_rate);

        std::vector<std::thread> workers(n_threads);
        for (int iw = 0; iw < n_threads; ++iw) {
            workers[iw] = std::thread([&](int ith) {
                int tid = gettid();
                setpriority(PRIO_PROCESS, tid, -10);

                std::vector<float> fft_in;
                fft_in.resize(fft_size);
                for (int i = 0; i < fft_size; i++) {
                    fft_in[i] = 0.0;
                }

                std::vector<float> fft_out;
                fft_out.resize(2 * fft_size);

                for (int i = ith; i < mel.n_len; i += n_threads) {
                    const int offset = i * fft_step;

                    // apply Hanning window
                    for (int j = 0; j < fft_size; j++) {
                        if (offset + j < n_samples) {
                            fft_in[j] = hann[j] * samples[offset + j];
                        }
                        else {
                            fft_in[j] = 0.0;
                        }
                    }

                    // FFT -> mag^2
                    fft(fft_in, fft_out);

                    for (int j = 0; j < fft_size; j++) {
                        fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
                    }
                    for (int j = 1; j < fft_size / 2; j++) {
                        //if (i == 0) {
                        //    printf("%d: %f %f\n", j, fft_out[j], fft_out[fft_size - j]);
                        //}
                        fft_out[j] += fft_out[fft_size - j];
                    }
                    if (i == 0) {
                        //for (int j = 0; j < fft_size; j++) {
                        //    printf("%d: %e\n", j, fft_out[j]);
                        //}
                    }

                    // mel spectrogram
                    for (int j = 0; j < mel.n_mel; j++) {
                        double sum = 0.0;

                        for (int k = 0; k < n_fft; k++) {
                            if(isnan(fft_out[k]))   // <---- had to add this, otherwise log_mel_spectrogram() returns -nan entries
                                continue;

                            sum += fft_out[k] * filters.data[j * n_fft + k];
                        }
                        if (sum < 1e-10) {
                            sum = 1e-10;
                        }

                        sum = log10(sum);

                        mel.data[j * mel.n_len + i] = (float)sum;
                    }
                }
            }, iw);
        }

        for (int iw = 0; iw < n_threads; ++iw) {
            workers[iw].join();
        }

        // clamping and normalization
        double mmax = -1e20;
        for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
            if (mel.data[i] > mmax) {
                mmax = mel.data[i];
            }
        }
        //printf("%s: max = %f\n", __func__, mmax);

        mmax -= 8.0;

        for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
            if (mel.data[i] < mmax) {
                mel.data[i] = (float)mmax;
            }

            mel.data[i] = (mel.data[i] + 4.0f) / 4.0f;
        }

        return true;
    }

    template <typename T>
    T vectorProduct(const std::vector<T>& v)
    {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
};

#endif //UHO_WHISPER_H
