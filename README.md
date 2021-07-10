# OpenVINO Real-time version of 'Noise Suppression Demo'

### Description:
Intel OpenVINO Toolkit 2021.4 introduced a new deep-learning based audio noise suppression demo. The demo takes an audio input file and produce a noise-suppressed output audio file. The input audio file must meet specific audio file format (WAV, 16KHz, Mono or 2ch).  
 I modified the demo program to make it possible to process real-time audio stream. Now you can use a microphone as 
input and check noise-supprssed audio with a headphone. The demo program can enable / disable noise suppression function so that you can check the effect immediately.      

Intel OpenVINO Toolkit 2021.4にディープラーニングベースのオーディオノイズサプレッションデモプログラムが追加されました。このデモプログラムはオーディオファイルを入力とし、ノイズ抑制したオーディオファイルを出力します。入力オーディオファイルは特定のフォーマットである必要があります(WAV, 16KHz, 1ch or 2ch)。  
今回、このデモプログラムを改造し、リアルタイムに音声ストリームを処理できるようにしました。マイクを使って音声を入力し、ヘッドフォンでノイズ抑制効果を確認できます。簡単に効果の確認ができるようにノイズ抑制をON/OFFすることも可能です。  

---
### Prerequisites:
- OpenVINO 2021.4  
- Python 3.x
- Python modues : pyaudio, opencv-python, numpy

---
### How to run:

1. Install OpenVINO
 Follow the '[Get Started Guide](https://docs.openvinotoolkit.org/latest/index.html)' instruction to install and setup OpenVINO 2021.4.

2. Install Python prerequisites
```sh
python3 -m pip install -r requirements.in
```

3. Obtain OMZ model
 Use '`model downloader`' in OpenVINO to download the `noise-suppressioin-poconetlike-0001` model.  
 ```sh
 python3 $INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name noise-suppression-poconetlike-0001
 ```

4. Setup audio device  
 Connect a microphone and a headphone. Setup audio configuration of your computer.

5. Run demo program
```sh
python3 noise_suppression_demo-rt.py -m intel/noise-suppression-poconetlike-0001/FP16/noise-suppression-poconetlike-0001.xml -d CPU
```  
* `noise_suppression_demo-rt-mt.py` is a **multi-threaded version** of the same demo. Audio recording and playback portion are run in isolated threads from the main inference thread.    

---
### Remarks:   
This program opens a small dummy window using OpenCV. The reason is to use `cv2.waitKey()` to read real-time key input. No other reason exists :-).  

---

### Disclaimer:
Provided as-is.

