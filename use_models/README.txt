You'll want to run "fetch_and_convert_models.ipynb" first.
Afterwards, if you wish, test model inference using "test_models.ipynb".

Additionally (e.g. for Android app), you may wish to convert our pretrained Whisper base model
to ONNX format that uses KV caching and **10 second context window**.
You can do that using "convert2onnx_reduce_ctx_10s.ipynb".
This approach significantly increases model inference speed -- especially the encoder, but also decoder.
(Android app is optimized for 10s context model)

ANDROID APP:
	• After model conversion, make sure to copy ctx10encoder.onnx and ctx10decoder.onnx to "UHO_Android\app\src\main\assets" folder.
WINDOWS APP:
	• After model conversion, make sure to copy CONTENTS (not folder itself) from "ct2_base_artur_best" folder (CTranslate2 model)
	  into same directory as UHO Windows executable file (i.e. "UHO_Windows\x64\Release")
