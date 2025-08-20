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
#include <random>
#include <io.h>	// setmode UTF8
#include <fcntl.h>	// -||-
#include "whisper.h"
#include "vad.h"
#include "stream.h"

class WhisperRealFeed {
public:
	struct WRFConfig {
		int sr = 16000;
		double maxSilenceResetTime = 10.0; //3.0; //  (13.8.2025 -- changed to 10 seconds)
		double zeroSamplesSeconds = 2.0; // 2.0 (use 2.0 normally) // 1.500 <---- defines how many seconds of silence sound buffer has at end (helps prevent hallucinations)
		double textviewNewlineSilenceTime = 3.0;    // after this time passes, printout textview will display a few empty lines for visual clarity
#if defined(CTX10_BASE_MODEL)
		double fullSizeSeconds = 10.0;	// buffer size (should stay fixed to 10.0)
		double windowSizeSeconds = 10.0; // buffer window size (anything outside this window is silence) -- not used
		double frameStepSeconds = 1.0; //2.0; // 3.0;   (how much seconds of audio we capture at a time; larger step means more accuracy, but lower speed of inference)
#elif defined(CTX6_BASE_MODEL)
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

	WhisperRealFeed() { }
	WhisperRealFeed(WRFConfig config) : config(config) {}

	bool appendTextModifyPrompt(string& textASR, vector<size_t>& prompt, shared_ptr<Whisper::BeamStructure> beam, int groupSize = 5, double groupLogP_thresh = -3.0, int backPopExtra = 2, bool shouldResetPrompt = false)
	{
		int numPopedBack = 0;
		deque<double> mergedPromptProbs;
		deque<size_t> mergedPromptIds;

		// The following condition can occur when whisper.additiveInference returns an empty prompt, and empty ids
		// (likely because it detected hallucinations/low probability id's at beginning of decoding)
		// Here we therefore return true (as opposed to false, which would trigger subsequent audio resetting),
		// which avoids errors having being made in the heuristics that follow in this function
		wprintf(L"ATMP DEBUG 1\n");

		if (shouldResetPrompt == true) {  // (13.8.2025 -- added this if condition)
			prompt = {};
			return true;
		}

		if (beam->ids.size() == 0)
			return true;    // if it returns false, it will reset prompt and buffer

		mergedPromptProbs = deque<double>(beam->probsPrompt.begin(), beam->probsPrompt.end());
		mergedPromptProbs.insert(mergedPromptProbs.end(), beam->probs.begin(), beam->probs.end());

		mergedPromptIds = deque<size_t>(prompt.begin(), prompt.end());
		mergedPromptIds.insert(mergedPromptIds.end(), beam->ids.begin(), beam->ids.end());

		if (beam->probsPrompt.size() != prompt.size())
			wprintf(L"FATAL ERROR 1\n");
		if (beam->probs.size() != beam->ids.size())
			wprintf(L"FATAL ERROR 2\n");

		// pop low probability ids at beginning
		// (but don't pop if we had an empty prompt -- because otherwise beginnings of transcription get corrupted often (they're low prob becuse of no context))

		wprintf(L"ATMP DEBUG 2\n");
		if (prompt.size() != 0) {
			// NEW CODE (23.7.2025) -- pops single tokens from prompt (experimental, not for use in production)
			/*while(mergedPromptIds.size() > beam->ids.size()) {
				if(log(mergedPromptProbs[0]) <= groupLogP_thresh/(double)groupSize) {
					wprintf(L"(NEW) Poping single from prompt");
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
					//wprintf(L"Poping from front, groupLogP=%.2f < groupLogP_thresh=%.2f\n", groupLogP, groupLogP_thresh);

					// (NEW 30.7.2025) if there's only groupSize tokens in prompt, pop them all
					if (mergedPromptIds.size() - beam->ids.size() == groupSize)
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
					wprintf(L"Not poping anymore\n");
					break;
				}
			}
		}

		wprintf(L"ATMP DEBUG 3\n");
		// pop low probability ids at end
		if ((numPopedBack < beam->ids.size()) && mergedPromptIds[mergedPromptIds.size() - 1] == special_tokens_start) {
			wprintf(L"ATMP DEBUG 3.1\n");
			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
		}
		else {
			wprintf(L"ATMP DEBUG 3.2\n");
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

			wprintf(L"Halucinacije ali pa preveč tokenov\n");
			return false;
		}

		wprintf(L"ATMP DEBUG 4\n");
		if ((numPopedBack < beam->ids.size()) && isTimestampToken(mergedPromptIds[mergedPromptIds.size() - 1])) {
			wprintf(L"ATMP DEBUG 4.1\n");
			//double endTimestamp = getTokenTimestamp(mergedPromptIds[mergedPromptIds.size() - 1]);
			//wprintf(L"EndTimestamp: %.2f sec\n", endTimestamp);
			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
		}

		// Pop low probability IDs
		while ((beam->ids.size() - numPopedBack) >= groupSize) {
			wprintf(L"ATMP DEBUG 5.1\n");
			double groupLogP = 0.0;
			for (int j = 0; j < groupSize; j++)
				groupLogP += log(mergedPromptProbs[mergedPromptProbs.size() - 1 - j]);

			if (groupLogP <= groupLogP_thresh) {
				wprintf(L"ATMP DEBUG 5.2\n");
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
			wprintf(L"ATMP DEBUG 6.1\n");
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
			if ((numPopedBack + 1) < beam->ids.size()) {   // +1 because we want at least one token...
				mergedPromptIds.pop_back();
				mergedPromptProbs.pop_back();
				numPopedBack++;
			}
			else {
				break;
			}
			//wprintf(L"Poping back (i=%i)\n", i);
		}

		// If prompt reset is triggered by long silence time (3 seconds), then decode those last IDs (don't pop extra when shouldResetPrompt=true)
		if (0 == 0 || shouldResetPrompt == false) {     // (13.8.2025 -- removed shouldResetPrompt condition)
			wprintf(L"ATMP DEBUG 7.1\n");
			// (conditionally) pop extra IDs from back
			if (beam->dontPopBack == false) {
				for (int i = 0; i < backPopExtra; i++)
				{
					// don't pop beyond current prompt!
					if ((numPopedBack + 1) >= beam->ids.size())    // "+1" is there, because we want at least one token
						break;

					mergedPromptIds.pop_back();
					mergedPromptProbs.pop_back();
					numPopedBack++;
					wprintf(L"ATMP DEBUG 7.2\n");
				}
			}

			if (config.popBackUntilWholeWord == true) {
				// Pop tokens until there is a token with " " (space) symbol in its first character.
				// This might help make better predictions for next word, because it starts to predict whole word instead of part of the word.
				while (numPopedBack < beam->ids.size())
				{
					string sTok = whisper.mapIdToUTF8[mergedPromptIds[mergedPromptIds.size() - 1]];
					wstring wTok = whisper.utf8TextToWstring(sTok);
					mergedPromptIds.pop_back();
					mergedPromptProbs.pop_back();
					numPopedBack++;
					if (wTok.length() > 0 && wTok[0] == L' ') // Symbol Ġ -- this is the space symbol in vocab.json
					{
						wprintf(L"(NEW) Found last space symbol in tokens, breaking out of loop\n");
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
	
	float findPeak(vector<float>& newBuffer, deque<float>& fullBuffer) {
		float peakNewBuffer = std::max(
			*std::max_element(newBuffer.begin(), newBuffer.end()),
			abs(*std::min_element(newBuffer.begin(), newBuffer.end()))
		);

		float peakFullBuffer = 0.0f;
		if (!fullBuffer.empty()) {
			peakFullBuffer = std::max(
				*std::max_element(fullBuffer.begin(), fullBuffer.end()),
				abs(*std::min_element(fullBuffer.begin(), fullBuffer.end()))
			);
		}

		float peak = std::max(peakNewBuffer, peakFullBuffer);
		return peak;
	}

	vector<float> normalize(vector<float>& chunk, float peak) {
		if (peak == 0.0)
			return chunk;

		vector<float> out;
		for (int i = 0; i < chunk.size(); i++)
			out.push_back(chunk[i] / peak);

		return out;
	}

	// Buffer normalization
	// Can be used as Automatic Gain Control (AGC), based on moving average filtering, by setting useMovingGain to "true"
	
	vector<float> normalize(vector<float>& chunk) {
		bool useMovingGain = false;
		float minGain1 = 10.0f;
		static float movingGain = 10.0f;

		float max = 0.0;
		for (int i = 0; i < chunk.size(); i++) {
			float c = abs(chunk[i]);
			if (c > max)
				max = c;
		}

		if (max == 0.0)
			return chunk;

		float memory = 0.5f;
		float maxGain = 1.0f / max;
		float minGain2 = std::min(minGain1, maxGain);
		movingGain = std::clamp(memory * movingGain + (1.0f - memory) * maxGain, minGain2, maxGain);
		//wprintf(L"movingGain = %.2f VS maxGain = %.2f\n", movingGain, maxGain);

		vector<float> out;
		for (int i = 0; i < chunk.size(); i++) {
			if(useMovingGain)
				out.push_back(chunk[i] * movingGain); 
			else
				out.push_back(chunk[i] * maxGain);
		}

		return out;
	}

    // Reimplementation of getNextStreamMel (august 2025)
    // Features:
    //  - function waits until there are at least config.frameStepSeconds of voice activity; silent segments captured in loop don't count as voice activity
    //      - parameter maxSoundActive controls how much time of silence is added to left and right 'margins' of recorded voice activity segments
    //      - if buffer that stores voice activity (noSilenceNewBuffer) contains config.maxSilenceResetTime seconds of silence, and only silence,
    //        break out of loop, reset buffer, and signalize other code to reset prompt
    //      - break out of loop if there's been *some* voice activity, and at least 0.1 seconds of silence (shows captions with a faster pace)
    //  - optionally add white noise (experimental feature, generally doesn't seem to be necessary)
    //  - sound buffer is left-aligned, such that final "computeBuffer" variable will have zero padding on right
    //      - computeBuffer has a limited size of (10 - config.zeroSamplesSeconds) seconds (for base model with 10s context window this is usually 8 seconds)
    //           --> this means that when buffer contains 7 seconds of audio, and we want to add e.g. 2 new seconds of audio,
    //               buffer will temporarily be 9 seconds long, and then 1 second of audio will be pop-ed from front of buffer
    //               so that buffer is 8 seconds long (or generally 10 - config.zeroSeconds). Afterwards, 2 seconds of padding will be
    //               added to this 8 second audio so that we have 10 second buffer. We use that final 10 second buffer to compute mel spectrogram.
    whisper_mel getNextStreamMel(AudioStream* stream, bool& shouldResetPrompt, atomic<bool>& stopLooperThread, bool& dontPopBackExtra, double& outSilenceTime) {
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

        static bool prevBreakBecauseOfFramestep = false;
        vector<float> noSilenceNewBuffer;
        int maxSoundActive = 3;
        static double silenceTime = 0.0;
        outSilenceTime = silenceTime;
        static int soundActive = -maxSoundActive;
        static deque<vector<float>> prev3Chunks;
        if (!prev3Chunks.size()) {
            for (int i = 0; i < maxSoundActive; i++)
                prev3Chunks.push_back(vector<float>(vad.audioChunkSize, 0.0f));
        }

        while (true && !stopLooperThread) {
            newBuffer = stream->getNextAudioChunk(0.25);
            if (newBuffer.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));    // (1.8.2025 -- changed from 100 to 50)
                continue;
            }

			//vector<float> newBufferNorm = normalize(newBuffer, findPeak(newBuffer, fullBuffer));
			vector<float> newBufferNorm = normalize(newBuffer);

            for (int i = 0; i < newBuffer.size() - vad.audioChunkSize; i += vad.audioChunkSize) {
                vector<float> chunk = vector<float>(newBuffer.begin() + i, newBuffer.begin() + i + vad.audioChunkSize);
				vector<float> chunkNorm = vector<float>(newBufferNorm.begin() + i, newBufferNorm.begin() + i + vad.audioChunkSize);
                float v = vad.forward(chunkNorm);
                if (v > 0.5f) {
                    // prepend code
                    if (soundActive < 0) {
                        int numToPrepend = -soundActive;
                        for (int j = 0; j < numToPrepend; j++) {
                            noSilenceNewBuffer.insert(
                                noSilenceNewBuffer.end(),
                                prev3Chunks[maxSoundActive - j - 1].begin(),
                                prev3Chunks[maxSoundActive - j - 1].end()
                            );
                        }
                    }
                    soundActive = maxSoundActive;
                }
                else {
                    soundActive = std::max(soundActive - 1, -maxSoundActive);
                }

                // append code
                if (v > 0.5f || soundActive >= 0) {
                    noSilenceNewBuffer.insert(noSilenceNewBuffer.end(), chunk.begin(), chunk.end());
                    silenceTime = 0.0;
                }
                else {
                    silenceTime += (double)vad.audioChunkSize / (double)config.sr;
                    outSilenceTime = silenceTime;   // return silenceTime parameter, to display a few empty lines in textview if there's been silence
                }

                prev3Chunks.pop_front();
                prev3Chunks.push_back(chunk);
            }

            wprintf(L"noSilenceNewBuffer size = %.3f seconds, silenceTime = %.2f\n", (double)noSilenceNewBuffer.size() / (double)config.sr, silenceTime);

            if ((!noSilenceNewBuffer.empty() || prevBreakBecauseOfFramestep) && silenceTime >= 0.1) {
                wprintf(L"BREAKING because of SHORT silence\n");
                dontPopBackExtra = true; // make it decode all ID's without popping back extra
                // (NOTE@01.08.2025: setting dontPopBackExtra to 'true' gave significant errors during real-time transcriptions)
                prevBreakBecauseOfFramestep = false;
                break;
            }

            if (silenceTime >= config.maxSilenceResetTime) {
                wprintf(L"BREAKING because of LONG silence\n");
                fullBuffer.clear();   // (actually we clear fullBuffer later in this function, so that last tokens are decoded too)
                shouldResetPrompt = true;   // Signal to reset prompt
                // we set silenceTime back to 0 in subsequent code
                prevBreakBecauseOfFramestep = false;
                break;
            }

            if (noSilenceNewBuffer.size() >= frameStep) {
                wprintf(L"BREAKING because of frameStep\n");
                prevBreakBecauseOfFramestep = true;
                break;
            }
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // (1.8.2025 changed from 100 to 50)
        }

        if (stopLooperThread)
            return whisper_mel();

        newBuffer = noSilenceNewBuffer;


        //wprintf(L"DEBUG4");
        fullBuffer.insert(fullBuffer.end(), newBuffer.begin(), newBuffer.end());
        // truncate full buffer
        //wprintf(L"DEBUG5");
        while (fullBuffer.size() > fullSize)
            fullBuffer.pop_front();

        computeBuffer = vector<float>(fullBuffer.begin(), fullBuffer.end());//vector<float>(truncBuffer.begin(), truncBuffer.end());
        if (silenceTime >= config.maxSilenceResetTime) {
            silenceTime = 0.0;
            fullBuffer.clear();
        }

        if (config.addWhiteNoise == true) {
            // Code that adds some white noise (attempts to prevent situations when decoding stops too early (slov. zastoji))
            float peak = 0.0f;
            for (int i = 0; i < computeBuffer.size(); i++) {
                float val = abs(computeBuffer[i]);
                if (val > peak)
                    peak = val;
            }

            float mean = 0.0f;
            float stddev = peak * 0.001f;  // suggested not to set this too high (too much may increase hallucinations?)
            std::default_random_engine generator;
            std::normal_distribution<float> dist(mean, stddev);
            for (int i = 0; i < computeBuffer.size(); i++)
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

        //wprintf(L"DEBUG10");

        int computeBufferSize = computeBuffer.size();
        int delta = fullSize - computeBufferSize;
        if (delta > numNewZeroSamples) {
            // in this case there will anyway be more than numNewZeroSamples zeros padded to the end, which is our goal
            computeBuffer.resize(fullSize, 0.0);
        }
        else {
            // if computeBufferSize is equal to fullSize, then we erase "numNewZeroSamples" from beginning, and add same to end
            // if, however, it is e.g. equal to fullSize-1, we remove numNewZeroSamples-1 from beginning, and add numNewZeroSamples+1 to end
            // similar for other values
            computeBuffer.erase(computeBuffer.begin(), computeBuffer.begin() + numNewZeroSamples - delta);
            computeBuffer.resize(fullSize, 0.0);
        }

        out = whisper.computeSpectrogramFromBuff(computeBuffer);
        wprintf(L"getNextStreamMel finished, fullBuffer length=%.2f sec, computeBufferSize=%i\n", (double)fullBuffer.size() / 16000.0, computeBuffer.size());
        return out;
    }

	atomic<bool> stopLooperThread{ false };
	thread looperThread;

	bool whisperBeginFeed(void (*f)(string addedTextUTF8), bool micOrStream) {
		if (looperThread.joinable())
			return false;	// signalize that thread is still running, and that whisperStopFeed should be called first

		stopLooperThread = false;
		looperThread = thread(&WhisperRealFeed::additiveInferenceLoopASR, this, f, micOrStream);
		return true;
	}

	void whisperStopFeed() {
		stopLooperThread = true;
	}

	void additiveInferenceLoopASR(void (*fTextCallback)(string addedTextUTF8), bool micOrStream) {
		AudioStream stream(micOrStream);

		// Reset static audio buffer when beginning inference
		bool initialShouldResetPrompt = true;
		bool initialDontPopBackExtra;
		double initialSilenceTime;
		getNextStreamMel(&stream, initialShouldResetPrompt, stopLooperThread, initialDontPopBackExtra, initialSilenceTime);

		vector<size_t> prompt = {};
		//string utf8textASR = "";
		//wstring textASR = L"";

		while (stopLooperThread == false) {
			double lastSpeechTime;
			bool shouldResetPrompt = false;
			bool dontPopBackExtra = true;
			double silenceTime = 0.0;

			auto t0 = chrono::high_resolution_clock::now();
			whisper_mel mel = getNextStreamMel(&stream, shouldResetPrompt, stopLooperThread, dontPopBackExtra, silenceTime);
			auto t1 = chrono::high_resolution_clock::now();
			wprintf(L"getNextStreamMel = %i ms\n", chrono::duration_cast<chrono::milliseconds>(t1 - t0).count());

			if (mel.data.empty())
				break;

			if (prompt.empty() && shouldResetPrompt == true)
				continue;

			/*
			char tmp[4096] = "Prompt is:\n";
			if(prompt.size() == 0)
				strcat(tmp, "[Empty]");
			for(int i = 0; i < prompt.size(); i++)
				sprintf(tmp, "%s%s", tmp, whisper.mapIdToUTF8[prompt[i]].c_str());
			wprintf(L"\"%s\"", tmp);
			 */

			auto a0 = chrono::high_resolution_clock::now();
			shared_ptr<Whisper::BeamStructure> beam = whisper.additiveInference(mel, prompt, config.numBeams, config.temperature, config.groupLogProb, config.groupLogP_thresh);
			auto a1 = chrono::high_resolution_clock::now();
			wprintf(L"additiveInference = %i ms\n", chrono::duration_cast<chrono::milliseconds>(a1 - a0).count());

			// sometimes whisper.additiveInference sets beam->dontPopBack to true (when false detect hallucinations).
			// in such a case we ignore "dontPopBackExtra" (obtained from getNextStreamMel)
			if (!beam->dontPopBack)
				beam->dontPopBack = dontPopBackExtra;

			// New text is now contained in beam->ids

			string utf8AddedTextASR;
			auto b0 = chrono::high_resolution_clock::now();
			bool res = appendTextModifyPrompt(utf8AddedTextASR, prompt, beam, config.groupLogProb, config.groupLogP_thresh, config.backPopExtra, shouldResetPrompt);
			auto b1 = chrono::high_resolution_clock::now();
			wprintf(L"appendTextModifyPrompt = %i ms\n", chrono::duration_cast<chrono::milliseconds>(b1 - b0).count());

			// New prompt is now contained in prompt variable

			if (res == false) {	// fail-proof fuse (in case of hallucinations)
				wprintf(L"RES == FALSE. Resetting stuff\n");
				shouldResetPrompt = true;
				getNextStreamMel(&stream, shouldResetPrompt, stopLooperThread, dontPopBackExtra, silenceTime);
				prompt = {};
			}

			/*if(shouldResetPrompt == true)
				fTextCallback(utf8AddedTextASR + ". ");
			else
				fTextCallback(utf8AddedTextASR);*/
			wprintf(L"silenceTime = %.2f\n", silenceTime);
			if (/*!utf8AddedTextASR.empty() && */silenceTime >= config.textviewNewlineSilenceTime) {
				wprintf(L"(callback with *)\n");
				fTextCallback("*" + utf8AddedTextASR);
			}
			else
				fTextCallback(utf8AddedTextASR);

			//textASR = whisper.utf8TextToWstring(utf8textASR);
			//wprintf(L"FULL TEXT: %s\n", textASR.c_str());

			/*for(int i = 0; i < prompt.size(); i++) {
				wprintf(L"(prompt) id=%lu (%s)", prompt[i], whisper.mapIdToUTF8[prompt[i]].c_str());
			}

			for(int i = 0; i < beam->ids.size(); i++)
				wprintf(L"(beam->ids) id=%lu (%s)", beam->ids[i], whisper.mapIdToUTF8[beam->ids[i]].c_str());
			*/

			whisper.beams.clear();	// (!) Very important (!)
		}

		wprintf(L"Out of loop!\n");
		looperThread.detach();
		wprintf(L"Detached looperThread\n");
	}
};

#endif
