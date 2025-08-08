// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef WHISPER_REALFEED_H
#define WHISPER_REALFEED_H

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <deque>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <random>
#include "whisper.h"
#include "vad.h"
#include "whisper.h"
#include "stream.h"

using namespace std;

class WhisperRealFeed {
public:
	struct WRFConfig {
		int sr = 16000;
		double maxSilenceResetTime = 3.0;
        double zeroSamplesSeconds = 2.0; // 2.0 (use 2.0 normally) // 1.500 <---- defines how many seconds of silence sound buffer has at end (helps prevent hallucinations)
#ifdef CTX10_BASE_MODEL
		double fullSizeSeconds = 10.0;	// buffer size (should stay fixed to 10.0)
		double windowSizeSeconds = 10.0; // buffer window size (anything outside this window is silence) -- not used
		double frameStepSeconds = 1.0; //2.0; // 3.0;   (how much seconds of audio we capture at a time; larger step means more accuracy, but lower speed of inference)
#elifdef CTX6_BASE_MODEL
        double fullSizeSeconds = 6.0;	// buffer size (should stay fixed to 6.0)
		double windowSizeSeconds = 6.0; // buffer window size (anything outside this window is silence) -- not used
		double frameStepSeconds = 2.0; //2.0; // 3.0;
#else
        double fullSizeSeconds = 30.0;	// buffer size (should stay fixed to 30.0)
		double windowSizeSeconds = 30.0;	// buffer window size (anything outside this window is silence) <<< very important for hallucination prevention, better than setting to 30.0 seconds
		double frameStepSeconds = 2.0; //2.0; // 3.0;
#endif

		// Additive inference
		int numBeams = 1;
		float temperature = 1.0f;
		int groupLogProb = 5; //5;
		double groupLogP_thresh = (double)groupLogProb * (-1.0); //(-0.8); //(-1.4);
		int backPopExtra = 2;//3;	// <-- very important, as the last word might have been corrupted by audio clipping at end
        bool popBackUntilWholeWord = false;
        bool addWhiteNoise = false;
	};

	WRFConfig config;
	SileroVAD vad;
	Whisper whisper;

	WhisperRealFeed(AAssetManager *manager): whisper(manager), vad(manager) { }
	WhisperRealFeed(AAssetManager *manager, WRFConfig config) : whisper(manager), vad(manager), config(config) {}

    /*
    // Experimental, not for use in production
    int removeHighCompressionIds(deque<size_t>& ids, std::map<int, std::string>& id2vocab)
	{
		int numRemoved = 0;
		std::string s1;
		double peakComprRatio;

		do {
			s1 = "";
			for (int i = 0; i < ids.size(); i++) {
				std::string tmp = id2vocab[ids[i]];
				for (char c : tmp)
					s1 += c;
			}
			peakComprRatio = peakGzipCompressionRatio(s1);
			if (peakComprRatio >= 2.0) {
				numRemoved++;
				ids.pop_back();
			}
		} while (ids.size() > 0 && peakComprRatio >= 2.0);

		if (numRemoved > 0) {
			int popExtra = 4;
			for (int i = 0; i < popExtra && ids.size() > 0; i++) {
				numRemoved++;
				ids.pop_back();
			}

			//wprintf(L"(!Warning!) Poor ratio, tried to rectify it.\n");
		}

		return numRemoved;
	}*/

	bool appendTextModifyPrompt(string& textASR, vector<size_t>& prompt, shared_ptr<Whisper::BeamStructure> beam, int groupSize = 5, double groupLogP_thresh = -3.0, int backPopExtra = 2, bool shouldResetPrompt = false)
	{
		int numPopedBack = 0;
		deque<double> mergedPromptProbs;
		deque<size_t> mergedPromptIds;

		// The following condition can occur when whisper.additiveInference returns an empty prompt, and empty ids
		// (likely because it detected hallucinations/low probability id's at beginning of decoding)
		// Here we therefore return true (as opposed to false, which would trigger subsequent audio resetting),
		// which avoids errors having being made in the heuristics that follow in this function
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 1");
		if (beam->ids.size() == 0)
			return true;    // if it returns false, it will reset prompt and buffer

		mergedPromptProbs = deque<double>(beam->probsPrompt.begin(), beam->probsPrompt.end());
		mergedPromptProbs.insert(mergedPromptProbs.end(), beam->probs.begin(), beam->probs.end());

		mergedPromptIds = deque<size_t>(prompt.begin(), prompt.end());
		mergedPromptIds.insert(mergedPromptIds.end(), beam->ids.begin(), beam->ids.end());

        if(beam->probsPrompt.size() != prompt.size())
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "FATAL ERROR 1");
        if(beam->probs.size() != beam->ids.size())
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "FATAL ERROR 2");

        // pop low probability ids at beginning
		// (but don't pop if we had an empty prompt -- because otherwise beginnings of transcription get corrupted often (they're low prob becuse of no context))

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 2");
        if (prompt.size() != 0) {
            // NEW CODE (23.7.2025) -- pops single tokens from prompt (experimental, not for use in production)
            /*while(mergedPromptIds.size() > beam->ids.size()) {
                if(log(mergedPromptProbs[0]) <= groupLogP_thresh/(double)groupSize) {
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) Poping single from prompt");
                    mergedPromptIds.pop_front();
                    mergedPromptProbs.pop_front();
                } else
                    break;
            }*/

            while ((mergedPromptIds.size() - beam->ids.size()) >= groupSize) {
                double groupLogP = 0.0;
                for (int j = 0; j < groupSize; j++)
                    groupLogP += (double)log(mergedPromptProbs[j]);

                if (groupLogP <= groupLogP_thresh) {
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "Poping from front, groupLogP=%.2f < groupLogP_thresh=%.2f", groupLogP, groupLogP_thresh);

                    // (NEW 30.7.2025) if there's only groupSize tokens in prompt, pop them all
                    if(mergedPromptIds.size() - beam->ids.size() == groupSize)
                    {
                        mergedPromptIds.erase(mergedPromptIds.begin(), mergedPromptIds.begin() + groupSize);
                        mergedPromptProbs.erase(mergedPromptProbs.begin(), mergedPromptProbs.begin() + groupSize);
                    }
                    else {
                        mergedPromptIds.pop_front();
                        mergedPromptProbs.pop_front();
                    }

                    //mergedPromptIds.pop_front();
                    //mergedPromptProbs.pop_front();
                }
                else {
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "Not poping anymore");
                    break;
                }
            }
        }

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 3");
		// pop low probability ids at end
		if ((numPopedBack < beam->ids.size()) && mergedPromptIds[mergedPromptIds.size() - 1] == special_tokens_start) {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 3.1");
			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
		}
		else {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 3.2");
			/*wprintf(L"Something's going on (repetitions/hallucinations)\n");
			int numRemoved = removeHighCompressionIds(mergedPromptIds, whisper.mapIdToUTF8);
			for (int i = 0; i < numRemoved; i++)
				mergedPromptProbs.pop_back();
			numPopedBack += numRemoved;
			 */

			// For now, in case the last token isn't special_tokens_start (<|endoftext|>), we'll just assume
			// that hallucinations or repetitions are present, and we'll reset prompt & audio by returning false.
			// For now, we decided not to try to fix text with repetitions;
			// However, if you decide to try that approach, you may uncomment the lines above

            __android_log_print(ANDROID_LOG_INFO, "UHO2", "Halucinacije ali pa preveč tokenov");
            return false;
		}

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 4");
		if ((numPopedBack < beam->ids.size()) && isTimestampToken(mergedPromptIds[mergedPromptIds.size() - 1])) {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 4.1");
			//double endTimestamp = getTokenTimestamp(mergedPromptIds[mergedPromptIds.size() - 1]);
			//wprintf(L"EndTimestamp: %.2f sec\n", endTimestamp);
			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
		}

		// Pop low probability IDs
		while ((beam->ids.size() - numPopedBack) >= groupSize) {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 5.1");
			double groupLogP = 0.0;
			for (int j = 0; j < groupSize; j++)
				groupLogP += log(mergedPromptProbs[mergedPromptProbs.size() - 1 - j]);

			if (groupLogP <= groupLogP_thresh) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 5.2");
				mergedPromptIds.pop_back();
				mergedPromptProbs.pop_back();
				numPopedBack++;
			}
			else
				break;
		}

		// Pop back until we find alphabetical characters
		//		If we have for example transcription with () at end, e.g. "hello world, my na() ..."
		//		then we will remove ids so we obtain "hello world, my na"

		int size = mergedPromptIds.size();
		for (int i = size - 1; i >= 0; i--) {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 6.1");
			// check if last character corresponding to output sequence is alphabetical
			string idTextSoFar;
			for (int j = 0; j <= i; j++)
				idTextSoFar += whisper.mapIdToUTF8[mergedPromptIds[j]];
			wstring wstringIdTextSoFar = whisper.utf8TextToWstring(idTextSoFar);
			if (wstringIdTextSoFar.length() <= 0)	// just in case something went wrong
				break;

            if (iswalpha(wstringIdTextSoFar[wstringIdTextSoFar.length() - 1]))	// found alphabetical character, stop poping ids
                break;

            // As long as we have text that is longer than prompt, we can pop newly decoded tokens
            if((numPopedBack+1) < beam->ids.size()) {   // +1 because we want at least one token...
                mergedPromptIds.pop_back();
                mergedPromptProbs.pop_back();
                numPopedBack++;
            } else {
                break;
            }
			//wprintf(L"Poping back (i=%i)\n", i);
		}

        // If prompt reset is triggered by long silence time (3 seconds), then decode those last IDs (don't pop extra when shouldResetPrompt=true)
        if (shouldResetPrompt == false) {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 7.1");
            // (conditionally) pop extra IDs from back
            if(beam->dontPopBack == false) {
                for (int i = 0; i < backPopExtra; i++)
                {
                    // don't pop beyond current prompt!
                    if ((numPopedBack + 1) >= beam->ids.size())    // "+1" is there, because we want at least one token
                        break;
                    mergedPromptIds.pop_back();
                    mergedPromptProbs.pop_back();
                    numPopedBack++;
                    __android_log_print(ANDROID_LOG_INFO, "UHO2", "ATMP DEBUG 7.2");
                }
            }

            if(config.popBackUntilWholeWord == true) {
                // Pop tokens until there is a token with " " (space) symbol in its first character.
                // This might help make better predictions for next word, because it starts to predict whole word instead of part of the word.
                while(numPopedBack < beam->ids.size())
                {
                    string sTok = whisper.mapIdToUTF8[mergedPromptIds[mergedPromptIds.size()-1]];
                    wstring wTok = whisper.utf8TextToWstring(sTok);
                    mergedPromptIds.pop_back();
                    mergedPromptProbs.pop_back();
                    numPopedBack++;
                    if(wTok.length() > 0 && wTok[0] == L' ') // Symbol Ġ -- this is the space symbol in vocab.json
                    {
                        __android_log_print(ANDROID_LOG_INFO, "UHO2", "(NEW) Found last space symbol in tokens, breaking out of loop");
                        break;
                    }
                }
            }
        }
        else {
            wprintf(L"No need to pop extra, audio contains a pause.\n");
        }

		if (!shouldResetPrompt)
			prompt = vector<size_t>(mergedPromptIds.begin(), mergedPromptIds.end());
		else {
			prompt = {};
		}

		for (int i = 0; i < (int)(beam->ids.size() - numPopedBack); i++)
			textASR += whisper.mapIdToUTF8[beam->ids[i]];

		return true;
	}

    /*
    // Old implementation of getNextStreamMel (not for use in production)
	whisper_mel getNextStreamMel(shared_ptr<MainStream> stream, bool& shouldResetPrompt, atomic<bool> &stopLooperThread) {
		whisper_mel out;

		static int cumulativeSilenceCount = 0;

		int fullSize = (int)((double)config.sr * config.fullSizeSeconds);
		int windowSize = (int)((double)config.sr * config.windowSizeSeconds);
		int frameStep = (int)((double)config.sr * config.frameStepSeconds);
		static deque<float> fullBuffer(fullSize, 0.0f);

		// An input usage should by default provide "shouldResetPrompt=false", whereas "true" signalizes to reset audio
		if (shouldResetPrompt == true) {
			for (int i = 0; i < fullBuffer.size(); i++)
				fullBuffer[i] = 0.0f;
			return whisper_mel();
		}

		// add new audio data to end of buffer
		vector<float> newBuffer;

		while (true && !stopLooperThread) {
			newBuffer = stream->getNextAudioChunk(config.frameStepSeconds);
			if (newBuffer.size() > 0)
				break;
			else
				std::this_thread::sleep_for(std::chrono::milliseconds (100)); //Pa_Sleep(100);
		}


		fullBuffer.insert(fullBuffer.end(), newBuffer.begin(), newBuffer.end());
		// truncate full buffer
		while (fullBuffer.size() > fullSize)
			fullBuffer.pop_front();

		// find beginning of audio position
		int fullBufferBeginPos = fullBuffer.size() - 1;
		for (int i = 0; i < fullBuffer.size(); i++) {
			if (fullBuffer[i] != 0.0f) {
				fullBufferBeginPos = i;
				break;
			}
		}

		vector<float> truncBuffer;

        // uncomment below to enable VAD silence removal
        vector<float> tmpFullBufferVect = vector<float>(fullBuffer.begin(), fullBuffer.end());
        tmpFullBufferVect = whisper.PreemphasizeAndNormalize(tmpFullBufferVect);
		vad.resetStates();
		for (int i = fullBufferBeginPos; i < tmpFullBufferVect.size() - vad.audioChunkSize; i += vad.audioChunkSize) {
			vector<float> chunk = vector<float>(tmpFullBufferVect.begin() + i, tmpFullBufferVect.begin() + i + vad.audioChunkSize);
			float detection = vad.forward(chunk);
			float vadThresh = 0.5f;
			if (detection > vadThresh) {
				truncBuffer.insert(truncBuffer.end(), chunk.begin(), chunk.end());
				cumulativeSilenceCount = 0;
			}
			else {
				cumulativeSilenceCount += vad.audioChunkSize;
			}
		}

		// If there is enduring silence for longer period (e.g. 3 seconds), reset audio and reset prompt
		double cumulativeSilenceTime = (double)cumulativeSilenceCount / (double)config.sr;
		if (cumulativeSilenceTime >= config.maxSilenceResetTime) {
			// Reset audio
			for (int i = 0; i < fullBuffer.size(); i++)
				fullBuffer[i] = 0.0f;
			// Signal to reset prompt
			shouldResetPrompt = true;
		}
		else {
			shouldResetPrompt = false;
		}

		// pad start with zeros
		if (truncBuffer.size() < fullSize)
			truncBuffer.insert(truncBuffer.begin(), fullSize - truncBuffer.size(), 0.0f);

        // leave only "windowSize" samples
        for(int i = 0; i < fullSize - windowSize; i++)
            truncBuffer[i] = 0.0f;

//#ifdef CTX_BASE_MODEL
        // Set last fraction of seconds of audio to zeros (helps hallucination prevention -- because we predict timestamp just before EOT token)
        int numNewZeroSamples = (int)((double)config.sr * 1.500);
        truncBuffer.erase(truncBuffer.begin(), truncBuffer.begin() + numNewZeroSamples);
        truncBuffer.insert(truncBuffer.end(), numNewZeroSamples, 0.0);
//#endif

		out = whisper.computeSpectrogramFromBuff(truncBuffer);
		return out;
	}*/

    // Reimplementation of getNextStreamMel (august 2025)
    // Features:
    //  - function waits until there are at least config.frameStepSeconds of voice activity; silent segments captured in loop don't count as voice activity
    //      - parameter numVadChunksBetweenSound controls how much time of silence is added to left and right 'margins' of recorded voice activity segments
    //      - if buffer that stores voice activity (noSilenceNewBuffer) contains config.maxSilenceResetTime (3) seconds of silence, and only silence,
    //        break out of loop, reset buffer, and signalize other code to reset prompt
    //      - break out of loop if there's been *some* voice activity, and at least 0.1 seconds of silence (somewhat experimental, may help show captions with a faster pace; might be removed later)
    //  - optionally add white noise (experimental feature, generally doesn't seem to be necessary)
    //  - sound buffer is left-aligned, such that final "computeBuffer" variable will have zero padding on right
    //      - computeBuffer has a limited size of (10 - config.zeroSamplesSeconds) seconds (for base model with 10s context window this is usually 8 seconds)
    //           --> this means that when buffer contains 7 seconds of audio, and we want to add e.g. 2 new seconds of audio,
    //               buffer will temporarily be 9 seconds long, and then 1 second of audio will be pop-ed from front of buffer
    //               so that buffer is 8 seconds long (or generally 10 - config.zeroSeconds). Afterwards, 2 seconds of padding will be
    //               added to this 8 second audio so that we have 10 second buffer. We use that final 10 second buffer to compute mel spectrogram.
    whisper_mel getNextStreamMel(shared_ptr<MainStream> stream, bool& shouldResetPrompt, atomic<bool> &stopLooperThread, bool &dontPopBackExtra) {
        whisper_mel out;

        static int cumulativeSilenceCount = 0;
        float vadThresh = 0.5f;

        int fullSize = (int)((double)config.sr * config.fullSizeSeconds);
        int windowSize = (int)((double)config.sr * config.windowSizeSeconds);
        int frameStep = (int)((double)config.sr * config.frameStepSeconds);
        //static deque<float> fullBuffer(fullSize, 0.0f);
        static deque<float> fullBuffer;
        static vector<float> computeBuffer; // this is always returned

        dontPopBackExtra = false;

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG1");
        // An input usage should by default provide "shouldResetPrompt=false", whereas "true" signalizes to reset audio
        if (shouldResetPrompt == true) {
            fullBuffer.clear();

            //for (int i = 0; i < fullBuffer.size(); i++)
            //    fullBuffer[i] = 0.0f;
            return whisper_mel();
        }
        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2");

        // add new audio data to end of buffer
        vector<float> newBuffer;

        /*
        // Original audio capture code, as long as there is new audio chunk, process it
        while (true && !stopLooperThread) {
            newBuffer = stream->getNextAudioChunk(config.frameStepSeconds);
            if (newBuffer.size() > 0)
                break;
            else
                std::this_thread::sleep_for(std::chrono::milliseconds (100)); //Pa_Sleep(100);
        }*/

        // New audio capture code (20.7.2025): we need at least config.frameStepSeconds of voice activity
        double silenceTime = 0.0;
        vector<float> noSilenceNewBuffer;
        vad.resetStates();

        while (true && !stopLooperThread) {
            //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.1");
            newBuffer = stream->getNextAudioChunk(0.25);
            if (newBuffer.size() == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds (50));    // (1.8.2025 -- changed from 100 to 50)
                continue;
            }

            //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2");
            //int sizeWithPadding = ((int)ceil((double)newBuffer.size()/(double)vad.audioChunkSize))*vad.audioChunkSize; // if we got e.g. 0.3s, then it is rounded to 0.32 s, which is divisible by vad.audioChunkSize (0.032 s)
            //newBuffer.resize(sizeWithPadding, 0.0f);
            vector <float> newBufferNorm = newBuffer; //whisper.PreemphasizeAndNormalize(newBuffer);
            vector <float> tmpNoSilenceNewBuffer;

            int numVoiceChunksAdded = 0;
            int silenceCount = 0;
            int numVadChunksBetweenSound = 2;   // 3
            for (int i = 0; i < newBuffer.size() - vad.audioChunkSize; i += vad.audioChunkSize) {
                vector<float> chunkNorm = vector<float>(newBufferNorm.begin() + i, newBufferNorm.begin() + i + vad.audioChunkSize);
                vector<float> chunk = vector<float>(newBuffer.begin() + i, newBuffer.begin() + i + vad.audioChunkSize);
                if(vad.forward(chunkNorm) > vadThresh) {
                    if(silenceCount >= numVadChunksBetweenSound) { // this happens when there was enduring silence, now suddenly theres voice activity
                        // in this case, we append 3 previous chunks (because in that timeframe, there might have been voice activity)
                        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2.1");
                        vector<float> prevChunks = vector<float>(
                                newBuffer.begin() + i - numVadChunksBetweenSound*vad.audioChunkSize,
                                newBuffer.begin() + i
                            );
                        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2.2");
                        tmpNoSilenceNewBuffer.insert(tmpNoSilenceNewBuffer.end(), prevChunks.begin(), prevChunks.end());
                        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2.3");
                    }
                    tmpNoSilenceNewBuffer.insert(tmpNoSilenceNewBuffer.end(), chunk.begin(), chunk.end());
                    //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2.4");
                    numVoiceChunksAdded++;
                    silenceTime = 0.0;
                    silenceCount = 0;
                } else {
                    //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2.5");
                    if(silenceCount < numVadChunksBetweenSound)
                        tmpNoSilenceNewBuffer.insert(tmpNoSilenceNewBuffer.end(), chunk.begin(), chunk.end());
                    //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.2.6");
                    silenceTime += (double)vad.audioChunkSize/(double)config.sr;
                    silenceCount++;
                }
            }
            //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.3");
            // insert last chunk (skipped by for loop if not divisible by vad.audioChunkSize) -- but only if there wasn't "all silence"
            tmpNoSilenceNewBuffer.insert(
                    tmpNoSilenceNewBuffer.end(),
                    newBuffer.begin() + newBuffer.size() - 1 - newBuffer.size()%vad.audioChunkSize,
                    newBuffer.end());
            //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG2.4");

            if(!noSilenceNewBuffer.empty() && silenceTime >= 0.1) {   // JUST FOR TESTING ! PROBABLY REMOVE LATER
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "BREAKING BECAUSE OF SMALL SILENCE");
                //dontPopBackExtra = true; // make it decode all ID's without popping back extra
                // (NOTE@01.08.2025: setting dontPopBackExtra to 'true' gave significant errors during real-time transcriptions)
                break;
            }

            if(silenceTime >= config.maxSilenceResetTime) {
                //fullBuffer.clear();   // (actually we clear fullBuffer later in this function, so that last tokens are decoded too)

                // Signal to reset prompt
                shouldResetPrompt = true;
                break;
            }

            if(numVoiceChunksAdded == 0)
                tmpNoSilenceNewBuffer.clear();
            else
                noSilenceNewBuffer.insert(noSilenceNewBuffer.end(), tmpNoSilenceNewBuffer.begin(), tmpNoSilenceNewBuffer.end());

            if(noSilenceNewBuffer.size() >= frameStep)
                break;
            else
                std::this_thread::sleep_for(std::chrono::milliseconds (50)); //100)); //Pa_Sleep(100);  // (1.8.2025 changed from 100 to 50)
        }

        if(stopLooperThread)
            return whisper_mel();

        newBuffer = noSilenceNewBuffer;

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG3");

        /*
        // Check with VAD if all new samples are silence. If yes, return empty mel()
        // Code that uses this just needs to check if prompt={} and mel() is empty, then it doesn't need to decode
        bool allSilent = true;
        vad.resetStates();
        for (int i = 0; i < newBuffer.size() - vad.audioChunkSize; i += vad.audioChunkSize) {
            vector<float> chunk = vector<float>(newBuffer.begin() + i, newBuffer.begin() + i + vad.audioChunkSize);
            float detection = vad.forward(chunk);
            float vadThresh = 0.5f;
            if (detection > vadThresh) {
                allSilent = false;
                break;
            }
        }
        if(allSilent)
            return whisper_mel();*/

        /*
        int whereToInsert = fullBuffer.size()-1;
        for(int i = fullBuffer.size()-1; i >= 0; i--) {
            if(fullBuffer[i] != 0.0)
                break;
            else
                whereToInsert = i;
        }*/

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG4");
        fullBuffer.insert(fullBuffer.end(), newBuffer.begin(), newBuffer.end());
        // truncate full buffer
        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG5");
        while (fullBuffer.size() > fullSize)
            fullBuffer.pop_front();

        /*
        // NEXT, VAD silence removal and 3 second silence detection
        vector<float> truncBuffer;
        vector<float> tmpFullBufferVect = vector<float>(fullBuffer.begin(), fullBuffer.end());
        //tmpFullBufferVect = whisper.PreemphasizeAndNormalize(tmpFullBufferVect);
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG6");
        vad.resetStates();
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG7");
        for (int i = 0; i < tmpFullBufferVect.size() - vad.audioChunkSize; i += vad.audioChunkSize) {
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG7.1 -- tmpFullBuffVect.size()=%i", tmpFullBufferVect.size());
            vector<float> chunk = vector<float>(tmpFullBufferVect.begin() + i, tmpFullBufferVect.begin() + i + vad.audioChunkSize);
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG7.2");
            float detection = vad.forward(chunk);
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG7.3");
            if (detection > vadThresh) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG7.4");
                truncBuffer.insert(truncBuffer.end(), chunk.begin(), chunk.end());
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG7.5");
                cumulativeSilenceCount = 0;
            }
            else {
                cumulativeSilenceCount += vad.audioChunkSize;
            }
        }
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG8");

        // If there is enduring silence for longer period (e.g. 3 seconds), reset audio and reset prompt
        double cumulativeSilenceTime = (double)cumulativeSilenceCount / (double)config.sr;
        if (cumulativeSilenceTime >= config.maxSilenceResetTime) {
            // Reset audio
            //for (int i = 0; i < fullBuffer.size(); i++)
            //    fullBuffer[i] = 0.0f;
            fullBuffer.clear();

            // Signal to reset prompt
            shouldResetPrompt = true;
        }
        else {
            shouldResetPrompt = false;
        }*/

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG9");

        computeBuffer = vector<float>(fullBuffer.begin(), fullBuffer.end());//vector<float>(truncBuffer.begin(), truncBuffer.end());
        if(silenceTime >= config.maxSilenceResetTime)
            fullBuffer.clear();

        if(config.addWhiteNoise == true) {
            // Code that adds some white noise (attempts to prevent situations when decoding stops too early (slov. zastoji))
            float peak = 0.0f;
            for(int i = 0; i < computeBuffer.size(); i++) {
                float val = abs(computeBuffer[i]);
                if(val > peak)
                    peak = val;
            }
            float mean = 0.0f;
            float stddev = peak * 0.001f;  // suggested not to set this too high (too much may increase hallucinations?)
            std::default_random_engine generator;
            std::normal_distribution<float> dist(mean, stddev);
            for(int i = 0; i < computeBuffer.size(); i++)
                computeBuffer[i] += dist(generator);
        }

//#ifdef CTX_BASE_MODEL
        // Set last fraction of seconds of audio to zeros (helps hallucination prevention -- because we predict timestamp just before EOT token)
        int numNewZeroSamples = (int)((double)config.zeroSamplesSeconds * (double)config.sr);
        //if(computeBuffer.size() >= fullSize) {
        //}

        // Buffer is like []
        // Audio (A) is   [A]
        //                [AA]
        //                  ...
        //                [AAAAAAAA]
        // When we resize the buffer we want to pad with 0's at end only if necessary, and remove A's from start only if necessary
        // Each A here is 1 second. Total max 10 A's.
        //

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG10");

        int computeBufferSize = computeBuffer.size();
        int delta = fullSize - computeBufferSize;
        if(delta > numNewZeroSamples) {
            // in this case there will anyway be more than numNewZeroSamples zeros padded to the end, which is our goal
            computeBuffer.resize(fullSize, 0.0);
        } else {
            // if computeBufferSize is equal to fullSize, then we erase "numNewZeroSamples" from beginning, and add same to end
            // if, however, it is e.g. equal to fullSize-1, we remove numNewZeroSamples-1 from beginning, and add numNewZeroSamples+1 to end
            // similar for other values
            computeBuffer.erase(computeBuffer.begin(), computeBuffer.begin() + numNewZeroSamples - delta);
            computeBuffer.resize(fullSize, 0.0);
        }

        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG11");
        //if(computeBufferSize < fullSize) {
        //    computeBuffer.resize(fullSize, 0.0);
        //}
        //fullBuffer.erase(fullBuffer.begin(), fullBuffer.begin() + numNewZeroSamples);
        //fullBuffer.insert(fullBuffer.end(), numNewZeroSamples, 0.0);
//#endif

        out = whisper.computeSpectrogramFromBuff(computeBuffer);
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "getNextStreamMel finished, fullBuffer length=%.2f sec, computeBufferSize=%i", (double)fullBuffer.size()/16000.0, computeBuffer.size());
        //__android_log_print(ANDROID_LOG_INFO, "UHO2", "DEBUG12");
        return out;
    }

    atomic<bool> stopLooperThread{ false };
	thread looperThread;

	bool whisperBeginFeed(void (*f)(string addedTextUTF8), shared_ptr<MainStream> stream) {
		if (looperThread.joinable())
			return false;	// signalize that thread is still running, and that whisperStopFeed should be called first

		stopLooperThread = false;
		looperThread = thread(&WhisperRealFeed::additiveInferenceLoopASR, this, f, stream);
		return true;
	}

	void whisperStopFeed() {
		stopLooperThread = true;
	}

	void additiveInferenceLoopASR(void (*fTextCallback)(string addedTextUTF8), shared_ptr<MainStream> stream) {
        int tid = gettid();
        setpriority(PRIO_PROCESS, tid, -10);
        //nice(10);   // increase thread priority

		// Reset static audio buffer when beginning inference
		bool initialShouldResetPrompt = true;
		bool initialDontPopBackExtra;
        getNextStreamMel(stream, initialShouldResetPrompt, stopLooperThread, initialDontPopBackExtra);

		vector<size_t> prompt = {};
		//string utf8textASR = "";
		//wstring textASR = L"";

		while (stopLooperThread == false) {
			double lastSpeechTime;
			bool shouldResetPrompt = false;
            bool dontPopBackExtra = true;

            auto t0 = chrono::high_resolution_clock::now();
            whisper_mel mel = getNextStreamMel(stream, shouldResetPrompt, stopLooperThread, dontPopBackExtra);
            auto t1 = chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "getNextStreamMel = %i ms", chrono::duration_cast<chrono::milliseconds>(t1-t0).count());

            if(mel.data.empty())
                break;

            if(prompt.empty() && shouldResetPrompt == true)
                continue;

            /*
            char tmp[4096] = "Prompt is:\n";
            if(prompt.size() == 0)
                strcat(tmp, "[Empty]");
            for(int i = 0; i < prompt.size(); i++)
                sprintf(tmp, "%s%s", tmp, whisper.mapIdToUTF8[prompt[i]].c_str());
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "\"%s\"", tmp);
             */

            auto a0 = chrono::high_resolution_clock::now();
			shared_ptr<Whisper::BeamStructure> beam = whisper.additiveInference(mel, prompt, config.numBeams, config.temperature, config.groupLogProb, config.groupLogP_thresh);
            auto a1 = chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "additiveInference = %i ms", chrono::duration_cast<chrono::milliseconds>(a1-a0).count());

            beam->dontPopBack = dontPopBackExtra;

            // New text is now contained in beam->ids

			string utf8AddedTextASR;
            auto b0 = chrono::high_resolution_clock::now();
			bool res = appendTextModifyPrompt(utf8AddedTextASR, prompt, beam, config.groupLogProb, config.groupLogP_thresh, config.backPopExtra, shouldResetPrompt);
            auto b1 = chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_INFO, "UHO2", "appendTextModifyPrompt = %i ms", chrono::duration_cast<chrono::milliseconds>(b1-b0).count());

			// New prompt is now contained in prompt variable

			if (res == false) {	// fail-proof fuse (in case of hallucinations)
				__android_log_print(ANDROID_LOG_INFO, "UHO2", "RES == FALSE. Resetting stuff");
                shouldResetPrompt = true;
				getNextStreamMel(stream, shouldResetPrompt, stopLooperThread, dontPopBackExtra);
				prompt = {};
			}

			if(shouldResetPrompt == true)
                fTextCallback(utf8AddedTextASR + ". ");
            else
                fTextCallback(utf8AddedTextASR);

			//textASR = whisper.utf8TextToWstring(utf8textASR);
			//wprintf(L"FULL TEXT: %s\n", textASR.c_str());

            /*for(int i = 0; i < prompt.size(); i++) {
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "(prompt) id=%lu (%s)", prompt[i], whisper.mapIdToUTF8[prompt[i]].c_str());
            }

            for(int i = 0; i < beam->ids.size(); i++)
                __android_log_print(ANDROID_LOG_INFO, "UHO2", "(beam->ids) id=%lu (%s)", beam->ids[i], whisper.mapIdToUTF8[beam->ids[i]].c_str());
            */

			whisper.beams.clear();	// (!) Very important (!)
		}

        __android_log_print(ANDROID_LOG_INFO, "UHO2", "Out of loop!");
		looperThread.detach();
        __android_log_print(ANDROID_LOG_INFO, "UHO2", "Detached looperThread");
	}
};

#endif
