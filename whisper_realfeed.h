// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef WHISPER_REALFEED_H
#define WHISPER_REALFEED_H

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <ctranslate2/layers/transformer.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <io.h>	// setmode UTF8
#include <fcntl.h>	// -||-
#include "whisper.h"
#include "vad.h"
#include "stream.h"

class WhisperRealFeed {
public:
	struct WRFConfig {
		int sr = 16000;
		double maxSilenceResetTime = 3.0;
		double fullSizeSeconds = 30.0;	// buffer size (should stay fixed to 30.0)
		double windowSizeSeconds = 30.0;	// buffer window size (anything outside this window is silence)
		double frameStepSeconds = 2.0; // 3.0;	// <<-- if this parameter is 0, we capture all available samples

		// Additive inference
		int numBeams = 1;
		float temperature = 1.0f;
		int groupLogProb = 5;
		double groupLogP_thresh = (double)groupLogProb * (-0.8);
		int backPopExtra = 3;	// <-- very important, as the last word might have been corrupted by audio clipping at end
	};

	WRFConfig config;
	SileroVAD vad;
	Whisper whisper;

	WhisperRealFeed() { }
	WhisperRealFeed(WRFConfig config) : config(config) {}

	int removeHighCompressionIds(deque<size_t>& ids, std::map<int, std::wstring>& id2vocab)
	{
		int numRemoved = 0;
		std::string s1;
		double peakComprRatio;

		do {
			s1 = "";
			for (int i = 0; i < ids.size(); i++) {
				std::wstring tmp = id2vocab[ids[i]];
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
	}

	bool appendTextModifyPrompt(Whisper& whisper, string& textASR, vector<size_t>& prompt, shared_ptr<Whisper::BeamStructure> beam, int groupSize = 5, double groupLogP_thresh = -3.0, int backPopExtra = 2, bool shouldResetPrompt = false)
	{
		int numPopedBack = 0;
		deque<double> mergedPromptProbs;
		deque<size_t> mergedPromptIds;

		// The following condition can occur when whisper.additiveInference returns an empty prompt, and empty ids
		// (likely because it detected hallucinations/low probability id's at beginning of decoding)
		// Here we therefore return true (as opposed to false, which would trigger subsequent audio resetting),
		// which avoids errors having being made in the heuristics that follow in this function
		if (beam->ids.size() == 0)
			return true;

		mergedPromptProbs = deque(beam->probsPrompt.begin(), beam->probsPrompt.end());
		mergedPromptProbs.insert(mergedPromptProbs.end(), beam->probs.begin(), beam->probs.end());

		mergedPromptIds = deque(prompt.begin(), prompt.end());
		mergedPromptIds.insert(mergedPromptIds.end(), beam->ids.begin(), beam->ids.end());

		// pop low probability ids at beginning
		// (but don't pop if we had an empty prompt -- because otherwise beginnings of transcription get corrupted often (they're low prob becuse of no context))
		if (prompt.size() != 0) {
			while (mergedPromptIds.size() >= groupSize) {
				double groupLogP = 0.0;
				for (int j = 0; j < groupSize; j++)
					groupLogP += log(mergedPromptProbs[j]);

				if (groupLogP <= groupLogP_thresh) {
					mergedPromptIds.pop_front();
					mergedPromptProbs.pop_front();
				}
				else
					break;
			}
		}

		// pop low probability ids at end
		if (mergedPromptIds.size() > 0 && mergedPromptIds[mergedPromptIds.size() - 1] == special_tokens_start) {
			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
		}
		else {
			/*
			wprintf(L"Something's going on (repetitions/hallucinations)\n");
			int numRemoved = removeHighCompressionIds(mergedPromptIds, whisper.id_to_vocab);
			for (int i = 0; i < numRemoved; i++)
				mergedPromptProbs.pop_back();
			numPopedBack += numRemoved;
			*/

			// For now, in case the last token isn't special_tokens_start, we'll just assume
			// that hallucinations or repetitions are present, and we'll reset prompt & audio by returning false.
			// For now, we decided not to try to fix text with repetitions;
			// However, if you decide to try that approach, you may uncomment the lines above
			return false;
		}

		if (mergedPromptIds.size() > 0 && isTimestampToken(mergedPromptIds[mergedPromptIds.size() - 1])) {
			//double endTimestamp = getTokenTimestamp(mergedPromptIds[mergedPromptIds.size() - 1]);
			//wprintf(L"EndTimestamp: %.2f sec\n", endTimestamp);
			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
		}

		// Pop low probability IDs
		while (mergedPromptIds.size() >= groupSize) {
			double groupLogP = 0.0;
			for (int j = 0; j < groupSize; j++)
				groupLogP += log(mergedPromptProbs[mergedPromptProbs.size() - 1 - j]);

			if (groupLogP <= groupLogP_thresh) {
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
			// check if last character corresponding to output sequence is alphabetical
			string idTextSoFar;
			for (int j = 0; j <= i; j++)
				idTextSoFar += whisper.mapIdToUTF8[mergedPromptIds[j]];
			wstring wstringIdTextSoFar = whisper.utf8TextToWstring(idTextSoFar);
			if (wstringIdTextSoFar.length() <= 0)	// just in case something went wrong
				break;
			if (iswalpha(wstringIdTextSoFar[wstringIdTextSoFar.length() - 1]))	// found alphabetical character, stop poping ids
				break;

			mergedPromptIds.pop_back();
			mergedPromptProbs.pop_back();
			numPopedBack++;
			//wprintf(L"Poping back (i=%i)\n", i);
		}

		// If prompt reset is triggered by long silence time (3 seconds), then decode those last IDs (don't pop extra when shouldResetPrompt=true)
		if (shouldResetPrompt == false) {
			// Pop extra IDs from back
			for (int i = 0; i < backPopExtra && mergedPromptIds.size() > 0; i++)
			{
				// don't pop beyond current prompt!
				if ((numPopedBack + 1) >= beam->ids.size())
					break;

				mergedPromptIds.pop_back();
				mergedPromptProbs.pop_back();
				numPopedBack++;
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

	whisper_mel getNextStreamMel(AudioStream* stream, bool& shouldResetPrompt, atomic<bool> &stopLooperThread) {
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
				Pa_Sleep(100);
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

		vad.resetStates();
		for (int i = fullBufferBeginPos; i < fullBuffer.size() - vad.audioChunkSize; i += vad.audioChunkSize) {
			vector<float> chunk = vector<float>(fullBuffer.begin() + i, fullBuffer.begin() + i + vad.audioChunkSize);
			float detection = vad.forward(chunk);
			float vadThresh = 0.0f;
			if (detection >= vadThresh) {
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
		out = Whisper::computeSpectrogramFromBuff(truncBuffer);
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
		getNextStreamMel(&stream, initialShouldResetPrompt, stopLooperThread);

		vector<size_t> prompt = {};
		//string utf8textASR = "";
		//wstring textASR = L"";

		while (stopLooperThread == false) {
			double lastSpeechTime;
			bool shouldResetPrompt = false;
			whisper_mel mel = getNextStreamMel(&stream, shouldResetPrompt, stopLooperThread);

			shared_ptr<Whisper::BeamStructure> beam = whisper.additiveInference(whisper, mel, prompt, config.numBeams, config.temperature, config.groupLogProb, config.groupLogP_thresh);

			// New text is now contained in beam->ids

			string utf8AddedTextASR;
			bool res = appendTextModifyPrompt(whisper, utf8AddedTextASR, prompt, beam, config.groupLogProb, config.groupLogP_thresh, config.backPopExtra, shouldResetPrompt);

			// New prompt is now contained in prompt variable

			if (res == false) {	// fail-proof fuse (in case of hallucinations)
				shouldResetPrompt = true;
				getNextStreamMel(&stream, shouldResetPrompt, stopLooperThread);
				prompt = {};
			}

			fTextCallback(utf8AddedTextASR);
			//textASR = whisper.utf8TextToWstring(utf8textASR);
			//wprintf(L"FULL TEXT: %s\n", textASR.c_str());

			whisper.beams.clear();	// (!) Very important (!)
		}

		looperThread.detach();
	}
};

#endif
