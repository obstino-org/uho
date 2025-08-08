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
        vector<float> out;

        newAudioLock.lock();
        // if number of samples is less than audioChunkSeconds, return empty buffer
        if (audio.size() < desiredSize) {
            newAudioLock.unlock();
            return vector<float>();
        }
        // get all available samples
        out.insert(out.begin(), audio.begin(), audio.end());
        audio.clear();
        newAudioLock.unlock();

        return out;
    }

    MainStream() { }
};

#endif //UHO_STREAM_H
