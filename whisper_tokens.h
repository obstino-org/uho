// Copyright (c) obstino-org. All rights reserved.
// Licensed under the MIT License.

#ifndef WHISPER_TOKENS_H
#define WHISPER_TOKENS_H

#include <vector>
#include <string>
using namespace std;

int special_tokens_start = 50257;

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

bool isPrintToken(int tokenId) {
    return (tokenId >= 0 && tokenId < special_tokens_start);
}

bool isTimestampToken(int tokenId) {
    return (tokenId >= timestamp_token_start && tokenId <= timestamp_token_last);
}

double getTokenTimestamp(int tokenId) {
    return (double)(tokenId - timestamp_token_start) * 0.02;
}

// timestamp_token_start == <|0.0|>
// timestamp_token_start + 1500 == <|30.0|>

//[f"<|{i * 0.02:.2f}|>" for i in range(1501)]

#endif