// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef UHO_STREAM_H
#define UHO_STREAM_H

#include <iostream>
#include <vector>
#include <mutex>
#include <string>

using namespace std;

class MainStream {
public:
    mutex newTextLock;
    mutex newAudioLock;

    int vadChunkSize = 512;
    vector<float> audio;
    string text;

    string getNewText() {
        string textOut;

        newTextLock.lock();
        textOut = text;
        text = "";
        newTextLock.unlock();

        return textOut;
    }

    void addNewText(string _text) {
        newTextLock.lock();
        text += _text;
        newTextLock.unlock();
    }

    void addNewAudio(vector <float>& _audio) {
        newAudioLock.lock();
        audio.insert(audio.end(), _audio.begin(), _audio.end());
        newAudioLock.unlock();
    }

    vector<float> getNextAudioChunk(double audioChunkSeconds = 2.0) {
        double sr = 16000.0;
        int desiredSize = (int)(sr * audioChunkSeconds);
        static vector<float> out;   // (12.8.2025 - changed to static)

        newAudioLock.lock();
        // if number of samples is less than audioChunkSeconds, return empty buffer
        if (audio.size() < desiredSize) {
            newAudioLock.unlock();
            return vector<float>();
        }
        // get all available samples
        out.insert(out.end(), audio.begin(), audio.end());
        audio.clear();
        newAudioLock.unlock();

        // return out;  // (12.8.2025 - removed this line)

        // 12.8.2025: Added functionality to truncate 'out', so that it will be divisible by vadChunkSize
        // if it wasn't divisible, leave remaining samples for next iteration of getNextAudioChunk
        int diff = out.size() % vadChunkSize;
        vector<float> actualOut = vector<float>(out.begin(), out.end() - diff);
        out.erase(out.begin(), out.end() - diff);

        return actualOut;
    }

    MainStream() { }
};

#endif //UHO_STREAM_H
