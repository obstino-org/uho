// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef STREAM_H
#define STREAM_H

#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include "portaudio.h"
#include "pa_win_wasapi.h"

using namespace std;

static inline char* stristr(const char* haystack, const char* needle) // source: https://github.com/troglobit/toolbox/blob/master/stristr.c
{
    char* pptr = (char*)needle;   // Pattern to search for
    char* start = (char*)haystack; // Start with a bowl of hay 
    char* sptr;                      // Substring pointer        
    int   slen = strlen(haystack); // Total size of haystack   
    int   plen = strlen(needle);   // Length of our needle     

    // while string length not shorter than pattern length
    for (; slen >= plen; start++, slen--)
    {
        // find start of pattern in string
        while (toupper(*start) != toupper(*needle))
        {
            start++;
            slen--;
            // if pattern longer than string 
            if (slen < plen)
            {
                return NULL;
            }
        }

        sptr = start;
        pptr = (char*)needle;
        while (toupper(*sptr) == toupper(*pptr))
        {
            sptr++;
            pptr++;
            // if end of pattern then pattern was found
            if ('\0' == *pptr)
            {
                return start;
            }
        }
    }

    return NULL;
}

class AudioStream
{
public:
    double sr = 16000.0;
    unsigned long framesPerBuffer = (unsigned long)(sr * 0.050);
    deque<float> buffer;
    mutex bufferLock;
    PaStream* stream;

    vector<float> getNextAudioChunk(double audioChunkSeconds = 3.0) {
        vector<float> out;
        int desiredSize;

        desiredSize = (int)(sr * audioChunkSeconds);

        bufferLock.lock();

        if (buffer.size() < desiredSize) {
            bufferLock.unlock();
            return vector<float>();
        }

        //if (desiredSize > 0.0) {
            //out.insert(out.begin(), buffer.begin(), buffer.begin() + desiredSize);
            //buffer.erase(buffer.begin(), buffer.begin() + desiredSize);
        //}
        //else {
            // if desiredSize = 0.0, we capture all samples available
            out.insert(out.begin(), buffer.begin(), buffer.end());
            buffer.clear();
        //}

        bufferLock.unlock();

        return out;
    }
    
    int streamCallbackMethod(
        const void* input, void* output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData)
    {
        float* fInput = (float*)input;

        bufferLock.lock();
        for (int i = 0; i < frameCount; i++)
            buffer.push_back(fInput[i]);
        bufferLock.unlock();

        return paContinue;
    }

    static int staticStreamCallback(
        const void* input, void* output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData)
    {
        //puts("got called at least once");
        AudioStream* audioStream = (AudioStream*)userData;
        return audioStream->streamCallbackMethod(input, output, frameCount, timeInfo, statusFlags, userData);
    }

    AudioStream(bool useMic = false) {
        Pa_Initialize();
        PaDeviceIndex inputDevice;

        if (useMic == true) {
            inputDevice = Pa_GetDefaultInputDevice();
        }
        else {
            inputDevice = -1;
            for (int i = 0; i < Pa_GetDeviceCount(); i++) {
                const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
                if (deviceInfo != nullptr) {
                    const PaHostApiInfo* hostApiInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
                    if (hostApiInfo != nullptr && hostApiInfo->type == paWASAPI && stristr(deviceInfo->name, "loopback")) {
                        inputDevice = i;
                        break;
                    }
                }
            }
            if (inputDevice == -1) {
                puts("Error: no loopback device found");
                return;
            }
        }

        // Loop through host APIs and find WASAPI
        const PaDeviceInfo* info = Pa_GetDeviceInfo(inputDevice);
        if (!info) {
            puts("Error getting DeviceInfo");
            return;
        }

        PaWasapiStreamInfo wasapiStreamInfo;
        PaStreamParameters inputParameters;

        inputParameters.channelCount = 1;
        inputParameters.device = inputDevice;
        inputParameters.hostApiSpecificStreamInfo = nullptr;
        inputParameters.sampleFormat = paFloat32;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputDevice)->defaultLowOutputLatency;

        if (useMic == false) {
            wasapiStreamInfo.size = sizeof(PaWasapiStreamInfo);
            wasapiStreamInfo.hostApiType = paWASAPI;
            wasapiStreamInfo.version = 1;
            wasapiStreamInfo.flags = paWinWasapiAutoConvert;
            inputParameters.hostApiSpecificStreamInfo = &wasapiStreamInfo;
        }

        PaError err1 = Pa_OpenStream(&stream,
            &inputParameters,
            nullptr,
            sr,
            framesPerBuffer,
            paClipOff,
            &AudioStream::staticStreamCallback,
            this);

        if (err1 != paNoError) {
            printf("Error opening stream: %s\n", Pa_GetErrorText(err1));
            return;
        }

        PaError err2 = Pa_StartStream(stream);
        if (err2 != paNoError) {
            printf("Error starting stream: %s\n", Pa_GetErrorText(err2));
            return;
        }
    }

    ~AudioStream() {
        Pa_StopStream(stream);
        Pa_Terminate();
    }
};

#endif
