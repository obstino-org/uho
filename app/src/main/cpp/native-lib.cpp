// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

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
#include <mutex>
#include <math.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

//#include <corecrt_math_defines.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include "gzip/compress.hpp"
#include "whisper.h"
#include "whisper_realfeed.h"
#include "stream.h"

#define STEP_SETTING_SPEED 0
#define STEP_SETTING_ACCURACY 1
double stepSpeed = 1.0;
double stepAccuracy = 2.0;

using namespace std;

shared_ptr<WhisperRealFeed> wrf = nullptr;
shared_ptr<MainStream> stream = nullptr;

void newTextCallback(string text) {
    stream->addNewText(text);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_obstino_uho_MainService_nativeGetNewText(
        JNIEnv* env,
        jobject mainService) {
    //return env->NewStringUTF(stream->getNewText().c_str());

    // TODO: if necessary, take utf8TextToWstring into caller Java code
    string outText = stream->getNewText();
    wstring outTextCorrected = Whisper::utf8TextToWstring(outText);
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return env->NewStringUTF(converter.to_bytes(outTextCorrected.c_str()).c_str());
}

// interface between Java code that records audio and C++ code that will use that audio
extern "C" JNIEXPORT void JNICALL
Java_com_obstino_uho_MainService_nativeAddNewAudio(
        JNIEnv* env,
        jobject mainService,
        jfloatArray audioArray) {
    jsize audioSize = env->GetArrayLength(audioArray);
    jfloat *audio = env->GetFloatArrayElements(audioArray, nullptr);

    vector<float> audioVect = vector<float>(audio, audio + audioSize);
    stream->addNewAudio(audioVect);

    env->ReleaseFloatArrayElements(audioArray, audio, 0);
}

extern "C" JNIEXPORT void JNICALL
Java_com_obstino_uho_MainService_nativeStartASR(
        JNIEnv* env,
        jobject mainService,
        jobject assetManager,
        jdouble stepSetting)
{
    AAssetManager *manager = AAssetManager_fromJava(env, assetManager);
    if(wrf != nullptr)
        return;

    __android_log_print(ANDROID_LOG_INFO, "UHO2", "Creating new MainStream object");
    stream = make_shared<MainStream>();
    __android_log_print(ANDROID_LOG_INFO, "UHO2", "Creating new WhisperRealFeed object");
    wrf = make_shared<WhisperRealFeed>(manager);

    // set frame step (bridge code)
    __android_log_print(ANDROID_LOG_INFO, "UHO2", "stepSetting = %.2f", stepSetting);
    wrf->config.frameStepSeconds = stepSetting;

    __android_log_print(ANDROID_LOG_INFO, "UHO2", "Calling whisperBeginFeed");
    wrf->whisperBeginFeed(newTextCallback, stream);
    __android_log_print(ANDROID_LOG_INFO, "UHO2", "Called OK.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_obstino_uho_MainService_nativeStopASR(
        JNIEnv* env,
        jobject mainService,
        jobject assetManager) {
    wrf->whisperStopFeed();

    while(wrf->looperThread.joinable()) {   // TODO: check if it's sufficient to check joinable
        this_thread::sleep_for(chrono::milliseconds (100));
    }

    wrf = nullptr;  // TODO: make sure this is ok to do
}
