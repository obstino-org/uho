// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.
// Credit goes to "https://github.com/nyadla-sys/whisper.cpp" for log_mel_spectrogram function along with dft, fft, filters_vocab_gen.bin usage and code

#ifndef WHISPER_H
#define WHISPER_H
//#include <windows.h>
#include <map>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <locale>
#include <codecvt>
#include <deque>
#include <regex>
#include <math.h>
//#include <corecrt_math_defines.h>
#include "whisper_tokens.h"
#include "gzip/compress.hpp"

#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <ctranslate2/layers/transformer.h>

#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_N_MEL       80
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_MEL_LEN     3000
#define WHISPER_STEP_MS 2700
#define WHISPER_LENGTH_MS 2*WHISPER_STEP_MS

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

double peakGzipCompressionRatio(string s) {
    double peakCompressionRatio = 0.0;
    int chunkSize = 3;
    std::reverse(s.begin(), s.end());
    for (int i = 0; i < s.length(); i += chunkSize) {
        string subStr = s.substr(0, i);
        string subStrZip = gzip::compress(subStr.c_str(), subStr.length());
        double compressionRatio = gzipCompressionRatio(subStr, subStrZip);
        //printf("Compression ratio: %.2f\n", compressionRatio);
        if (compressionRatio > peakCompressionRatio)
            peakCompressionRatio = compressionRatio;
    }
    return peakCompressionRatio;
}

class Beam {
public:
    struct TokenInfo {
    public:
        int tokenId;
        double origProbability;		// probability without temperature
        double tempProbability;		// probability using temperature

        bool operator==(const TokenInfo& t2) const {
            return (
                tokenId == t2.tokenId &&
                origProbability == t2.origProbability &&
                tempProbability == t2.tempProbability
                );
        }
    };

    ctranslate2::layers::DecoderState decoderState;
    int step;
    deque<TokenInfo> outputTokenInfo;
    
    Beam() {
    }

    double getLogProbability() {
        double logP = 0;
        for (int i = 0; i < outputTokenInfo.size(); i++)
            logP += log(outputTokenInfo[i].tempProbability);

        return logP;
    }

    double getCompressionRatio(std::map<int, std::wstring>& id2vocab) {
        std::string s1 = "", c1;
        for (int i = 0; i < outputTokenInfo.size(); i++) {
            std::wstring tmp = id2vocab[outputTokenInfo[i].tokenId];
            for (char c : tmp)
                s1 += c;
        }

        // at this point we just compress
        double comprRatio = peakGzipCompressionRatio(s1);
        return comprRatio;
    }
};

class Whisper {
public:
    std::shared_ptr <const ctranslate2::models::Model> model;
    std::shared_ptr <ctranslate2::models::WhisperReplica> wModel;
    std::string filtersPath = "./filters_vocab_gen.bin";

    int32_t n_vocab;
    std::map <int, wstring> id_to_vocab;
    map <int, string> mapIdToUTF8;

    vector<size_t> tokensToSuppress;

    Whisper()
    {
        //std::string path(env);
        
        std::string path = ".\\";  // <-- best model, before overfitting occurred
        
        model = ctranslate2::models::WhisperModel::load(path);
        wModel = ctranslate2::models::WhisperReplica::create_from_model(*model);

        auto fin = std::ifstream(filtersPath, std::ios::binary);
        uint32_t magic = 0;
        fin.read((char*)&magic, sizeof(magic));

        // load mel filters
        fin.read((char*)&g_filters.n_mel, sizeof(g_filters.n_mel));
        fin.read((char*)&g_filters.n_fft, sizeof(g_filters.n_fft));

        g_filters.data.resize(g_filters.n_mel * g_filters.n_fft);
        fin.read((char*)g_filters.data.data(), g_filters.data.size() * sizeof(float));

        fin.close();

        // Read vocabulary from json into utf8 integer-to-string map called "mapIdToUTF8"
        ifstream fVocab("vocab.json");

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

        fVocab.close();
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

        return softmaxNumerator / softmaxDenominator;
    }

    void runEncoder(whisper_mel& mel, ctranslate2::StorageView & encoderOutput) {
        const ctranslate2::Device device = wModel->_model->device();
        const ctranslate2::DataType encDtype = wModel->_encoder->output_type();
        encoderOutput = ctranslate2::StorageView(encDtype, device);
        
        ctranslate2::StorageView inFeatures(ctranslate2::Shape({ 1, 80, 3000 }), mel.data);
        inFeatures.move_to(device, encDtype);

        (*(wModel->_encoder))(inFeatures, encoderOutput);
    }

    pair<shared_ptr<ctranslate2::layers::DecoderState>, ctranslate2::StorageView> decoderForwardPass(ctranslate2::StorageView &encoderOutput, std::vector<size_t> basicPromptIds)
    {
        //std::vector<size_t> prompt = { 50258, 50305, 50359, 50363 };  // == <|startoftranscript|><|sl|><|transcribe|><|notimestamps|>
        std::vector<size_t> prompt = { 50258, 50305, 50359 };    // == <|startoftranscript|><|sl|><|transcribe|>

        // append basic prompt ids, except for last one, which will later be used as startId during additive inference!
        if(basicPromptIds.size() > 0)
            prompt.insert(prompt.end(), basicPromptIds.begin(), std::prev(basicPromptIds.end()));

        std::vector<std::vector<size_t>> prompts({ prompt });

        shared_ptr<ctranslate2::layers::DecoderState> state = make_shared<ctranslate2::layers::DecoderState>(wModel->_decoder->initial_state());
        state->emplace("memory", encoderOutput);

        wModel->_decoder->update_output_layer(wModel->_model->preferred_size_multiple());
        ctranslate2::StorageView inputs = ctranslate2::layers::make_sequence_inputs(prompts, wModel->_decoder->device());
        
        ctranslate2::StorageView outputs(wModel->_decoder->output_type(), wModel->_decoder->device());

        //wModel->_decoder->forward_prompt(inputs, *(state.get()), &outputs);
        (*(wModel->_decoder))(0, inputs, *(state.get()), &outputs, nullptr);
        
        return make_pair(state, outputs);
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

        // suppress tokens first
        for (int i = 0; i < tokensToSuppress.size(); i++) {
            logitsPtr[tokensToSuppress[i]] = std::numeric_limits<float>::lowest();
        }

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
        
        shared_ptr<ctranslate2::layers::DecoderState> decoderState;
        int lastOutTokenId;
        int nextStep;
        ctranslate2::StorageView lastOutput;

        // Essential elements used for updateBeams() and pruneBeans()
        vector<size_t> ids;         // new tokens
        vector<double> probs;       // new tokens raw probabilities
        vector<double> probsTemp;   // temperature-corrected new tokens probabilities
        vector<double> probsPrompt; // prompt token probabilities (size = prompt.size())
        double logProb;

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
            this->lastOutput = b.lastOutput;
        }
    };

    vector<shared_ptr<BeamStructure>> beams;

    void updateBeamPromptProbabilities(shared_ptr<Whisper::BeamStructure> beam, vector<size_t> &prompt, ctranslate2::StorageView &decOut) {
        if (prompt.size() == 0)
            return;

        int numTokens, numVocab;
        
        numTokens = decOut.shape()[1];
        numVocab = decOut.shape()[2];
        
        //wprintf(L"numTokens=%i\n", numTokens);

        int startIndex = 2;     // TODO: adjust based on prompt with special tokens
        float* logits = (float*)decOut.buffer();

        for (int i = startIndex; i < numTokens; i++) {
            double prob = (double)computeTokenProbability(i, prompt[i - startIndex], logits, 1.0);
            beam->probsPrompt.push_back(prob);
        }
    }

    bool bestBeamCompressionRatioCheck(Whisper& whisper, shared_ptr<Whisper::BeamStructure> beam0)
    {
        // returns false if there's high compression ratio, otherwise if all is okay it returns true
        string s1 = "";
        for (int i = 0; i < beam0->ids.size(); i++) {
            std::string tmp = whisper.mapIdToUTF8[beam0->ids[i]];
            for (char c : tmp)
                s1 += c;
        }
        
        double peakComprRatio = peakGzipCompressionRatio(s1);
        if (peakComprRatio >= 2.0) {
            return false;
        }
        return true;
    }

    shared_ptr<Whisper::BeamStructure> additiveInference(Whisper &whisper, whisper_mel &mel, vector<size_t> &prompt, int beamSize=4, float temperature=1.0f, int groupLogProb=5, double groupLogP_thresh=-2.0)
    {
        //int beamSize = 4;
        //float temperature = 1.0f;
        int maxSeqLen = 75;

        ctranslate2::StorageView encOut;

        //auto t0 = chrono::high_resolution_clock::now();
        //auto t0Enc = chrono::high_resolution_clock::now();
        runEncoder(mel, encOut);
        //auto t1Enc = chrono::high_resolution_clock::now();
        //FILE* file = fopen("tmp.txt", "a");
        //fwprintf(file, L"Encoder forward pass took %i ms\n", chrono::duration_cast<chrono::milliseconds>(t1Enc - t0Enc).count());
        //fclose(file);

        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));   // TODO REMOVE THIS !! ONLY FOR TESTING xD
        //auto t1 = chrono::high_resolution_clock::now();
        //wprintf(L"Encoder forward pass took %i ms\n", chrono::duration_cast<chrono::milliseconds>(t1 - t0).count());

        //wprintf(L"Inside here, prompt size is %i\n", prompt.size());
        auto decFwdOut = decoderForwardPass(encOut, prompt);

        shared_ptr<ctranslate2::layers::DecoderState> decState = decFwdOut.first;
        ctranslate2::StorageView decOut = decFwdOut.second;

        shared_ptr<Whisper::BeamStructure> beam = make_shared<Whisper::BeamStructure>();
        beam->decoderState = decState;
        beam->nextStep = prompt.size() + 2; //+ 3;
        beam->lastOutTokenId = (prompt.size() > 0) ? prompt[prompt.size() - 1] : 50359; //50363;
        beam->logProb = 0.0;

        beams.push_back(beam);

        for (int i = 0; i < maxSeqLen; i++) {
            vector<thread> threads(beams.size());

            for (int j = 0; j < beams.size(); j++)
                threads[j] = thread(&Whisper::inferenceStepThread, this, beams[j]);
            for (int j = 0; j < beams.size(); j++)
                threads[j].join();

            updateBeams(beamSize, temperature);
            pruneBeams(beamSize);

            if (bestBeamCompressionRatioCheck(whisper, beams[0]) == false) {
                beams[0]->ids = {};
                beams[0]->probs = {};
                beams[0]->probsTemp = {};
                break;
            }

            if (beams[0]->lastOutTokenId == special_tokens_start)
                break;

            // HALLUCINATION PREVENTION
            // check sum of log probabilities only once, because beginnings of hallucinations are often low prob
            double sumLogProb = 0.0;
            if ((i+1) == groupLogProb) {
                for(int j = 0; j < groupLogProb; j++)
                    sumLogProb += log(beams[0]->probs[j]);
                
                if (sumLogProb <= groupLogP_thresh) {
                    //wprintf(L"EARLY EXIT DUE TO LOW PROB (hallucinations?)\n");
                    beams[0]->ids = {};
                    beams[0]->probs = {};
                    beams[0]->probsTemp = {};
                    break;
                }
            }

        }

        // add probabilities for original prompt tokens to the beam we return
        updateBeamPromptProbabilities(beams[0], prompt, decOut);

        return beams[0];
    }

    void updateBeams(int beamSize, double temperature)
    {
        // update beam info...
        vector<shared_ptr<BeamStructure>> newBeams;

        for (int i = 0; i < beams.size(); i++) {
            int numVocab = beams[i]->lastOutput.shape()[1];
            
            // beamDecodeToken takes about 0.5ms
            vector<int> beamTokens = beamDecodeToken((float*)beams[i]->lastOutput.buffer(), 0, numVocab, beamSize, temperature);
            
            // for each of "numBeams", create numBeams new beams, totaling numBeams^2 new beams (they replace old 'beams')
            for (int j = 0; j < beamSize; j++) {
                shared_ptr<BeamStructure> newBeam;
                
                newBeam = make_shared<BeamStructure>(*(beams[i].get()));
                newBeam->nextStep++;
                newBeam->lastOutTokenId = beamTokens[j];
                newBeam->ids.push_back(beamTokens[j]);
                newBeam->fromBeamNumber = i;

                float* logits = (float*)beams[i]->lastOutput.buffer();

                // computing both prob and probTemp totals in about 0.5ms
                double prob = (double)computeTokenProbability(0, beamTokens[j], logits, 1.0);   // cpu intensive (almost 0.5ms)
                double probTemp = (double)computeTokenProbability(0, beamTokens[j], logits, temperature);   // cpu intensive (almost 0.5ms)

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
        
        //wprintf(L"Hello\n");
        for (shared_ptr<BeamStructure>& beam : beams)
        {
            if (std::find(beamsRequringDeepCopy.begin(), beamsRequringDeepCopy.end(), beam->fromBeamNumber) == beamsRequringDeepCopy.end())
                beamsRequringDeepCopy.push_back(beam->fromBeamNumber);
            else {
                //wprintf(L"\tRequiring copy = %i\n", beam->fromBeamNumber);
                beam->decoderState = make_shared<ctranslate2::layers::DecoderState>(*(beam->decoderState));
            }
        }
    }

    // single step thread
    void inferenceStepThread(shared_ptr<BeamStructure> beam)
    {
        beam->lastOutput = inferenceStep(*(beam->decoderState), beam->lastOutTokenId, beam->nextStep);
    }

    // single step
    ctranslate2::StorageView inferenceStep(ctranslate2::layers::DecoderState &decoderState, int32_t startId, int step)
    {
        int startStep = step;
        ctranslate2::Device device = wModel->_decoder->device();
        ctranslate2::StorageView sampleFrom({ 1 }, ctranslate2::DataType::INT32);
        ctranslate2::StorageView output(wModel->_decoder->output_type(), device);
        float* logitsData;
        int numVocab;
        int maxTokenId;

        sampleFrom.at<int32_t>(0) = startId;
        (*(wModel->_decoder))(step, sampleFrom.to(device), decoderState, &output, nullptr);
        return output;
    }

    static whisper_mel computeSpectrogramFromBuff(vector<float> buff)
    {
        whisper_mel mel = { 0 };
        buff = PreemphasizeAndNormalize<float>(buff);
        buff.resize(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE, 0);
        if (!log_mel_spectrogram(buff.data(), buff.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, 4, g_filters, mel)) {
            wprintf(L"error with spectrogram\n");
            return mel;
        }
        return mel;
    }

    whisper_mel computeSpectrogram(string filePath)
    {
        whisper_mel mel = { 0 };

        vector<short> pcmShort;
        vector<float> pcmFloat;
        uint32_t numEl;
        uint32_t sampleRate;
        ReadWaveIntoBuffer(filePath.c_str(), pcmShort, &numEl, &sampleRate, 0);
        for (int i = 0; i < numEl; i++)
            pcmFloat.push_back((float)pcmShort[i] / 32768.0);
        pcmFloat.resize(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE, 0);

        if (!log_mel_spectrogram(pcmFloat.data(), pcmFloat.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, 1, g_filters, mel)) {
            puts("error with spectrogram");
            return mel;
        }

        //puts("done computing");
        return mel;
    }

    bool ReadWaveIntoBuffer(const char* szFile, vector<short>& sout, uint32_t* numEl, uint32_t* sampleRate, uint32_t force_num_samples)	// assumes we have 1 channel, 16 bit
    {
        FILE* f;
        uint32_t Subchunk1Size, Subchunk2Size;
        char buff[100] = { 0 };
        fopen_s(&f, szFile, "rb");
        if (!f)
            return false;

        fseek(f, 16, 0);
        fread(&Subchunk1Size, sizeof(uint32_t), 1, f);
        fseek(f, 24, 0);
        fread(sampleRate, sizeof(uint32_t), 1, f);
        fseek(f, 16 + 4 + Subchunk1Size + 4, 0);
        fread(&Subchunk2Size, sizeof(uint32_t), 1, f);
        if (force_num_samples > 0) {
            sout = vector<short>(force_num_samples, 0);
            uint32_t read_size = std::min(force_num_samples * 2, Subchunk2Size);
            fread(&sout[0], 1, read_size, f);
            *numEl = force_num_samples;
        }
        else {
            sout = vector<short>(Subchunk2Size / 2, 0);
            fread(&sout[0], 1, Subchunk2Size, f);
            *numEl = Subchunk2Size / 2;
        }

        fclose(f);

        return true;
    }

    template <class T>
    static vector<float> PreemphasizeAndNormalize(vector<T>& in)
    {
        double filter_factor = 0.95;
        vector<float> dout;

        int num = in.size();
        float dmax;

        dout = vector<float>(num);

        dmax = 0.0f;
        for (int k = 1; k < (int)num; k++) {
            dout[k] = (float)in[k] - filter_factor * (double)in[k - 1];
            if (fabs(dout[k]) > dmax)
                dmax = fabs(dout[k]);
        }

        for (int k = 0; k < (int)num; k++)
            dout[k] = (float)(dout[k] / dmax);

        return dout;
    }

    static void dft(const std::vector<float>& in, std::vector<float>& out) {
        int N = in.size();

        out.resize(N * 2);

        for (int k = 0; k < N; k++) {
            float re = 0;
            float im = 0;

            for (int n = 0; n < N; n++) {
                float angle = 2 * M_PI * k * n / N;
                re += in[n] * cos(angle);
                im -= in[n] * sin(angle);
            }

            out[k * 2 + 0] = re;
            out[k * 2 + 1] = im;
        }
    }

    static void fft(const std::vector<float>& in, std::vector<float>& out) {
        out.resize(in.size() * 2);

        int N = in.size();

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
            float theta = 2 * M_PI * k / N;

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
    static bool log_mel_spectrogram(
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
            hann[i] = 0.5 * (1.0 - cos((2.0 * M_PI * i) / (fft_size)));
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
                            sum += fft_out[k] * filters.data[j * n_fft + k];
                        }
                        if (sum < 1e-10) {
                            sum = 1e-10;
                        }

                        sum = log10(sum);

                        mel.data[j * mel.n_len + i] = sum;
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
                mel.data[i] = mmax;
            }

            mel.data[i] = (mel.data[i] + 4.0) / 4.0;
        }

        return true;
    }

    template <typename T>
    T vectorProduct(const std::vector<T>& v)
    {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
};

#endif