#!/usr/bin/env python3
 
"""
 Copyright (c) 2021 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
      http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
 
import pyaudio
import cv2
 
import numpy as np
 
from openvino.inference_engine import IECore, Blob
 
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
                      required=True, type=Path)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on. "
                           "Default value is CPU",
                      default="CPU", type=str)
    return parser
 
def main():
    dummy_image = np.zeros((32 ,32 ,3), dtype=np.uint8)
    cv2.imshow('dummy_window', dummy_image)
 
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
 
    log.info("Initializing Inference Engine")
    ie = IECore()
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))
 
    # read IR
    model_xml = args.model
    model_bin = model_xml.with_suffix(".bin")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)
 
    # check input and output names
    input_shapes = {k: v.input_data.shape for k, v in ie_encoder.input_info.items()}
    input_names = list(ie_encoder.input_info.keys())
    output_names = list(ie_encoder.outputs.keys())
 
    assert "input" in input_names, "'input' is not presented in model"
    assert "output" in output_names, "'output' is not presented in model"
    state_inp_names = [n for n in input_names if "state" in n]
    state_param_num = sum(np.prod(input_shapes[n]) for n in state_inp_names)
    log.info("state_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))
 
    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)
 
    #sample_inp = wav_read(args.input)
 
    input_size = input_shapes["input"][1]
    res = None
 
    audio = pyaudio.PyAudio()
    # get_format_from_width(2), paInt16
    record_stream   = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input =True, frames_per_buffer=input_size)
    playback_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True, frames_per_buffer=input_size)
 
    samples_out = []
    samples_times = []
 
    print('OpenVINO Audio Noise Suppression Demo')
    print('<ESC>   : Exit program')
    print('<SPACE> : Enabling / Disabling noise suppression')
    print()
    print('Noise suppression is enabled')
 
    key = -1
    noise_suppress_flag = True
 
    while key != 27:
        key = cv2.waitKey(1)
        if key == ord(' '):
            noise_suppress_flag = False if noise_suppress_flag else True
            if noise_suppress_flag:
                print('Noise suppression is enabled')
            else:
                print('Noise suppression is diabled')
 
        input = np.frombuffer(record_stream.read(num_frames=input_size), dtype=np.int16)
        normalized_input = input.astype(np.float32) * (1.0 / np.iinfo(np.int16).max)
 
        #forms input
        inputs = {"input": normalized_input[None, :]}
    
        #add states to input
        for n in state_inp_names:
            if res:
                inputs[n] = res[n.replace('inp', 'out')].buffer
            else:
                #on the first iteration fill states by zeros
                inputs[n] = np.zeros(input_shapes[n], dtype=np.float32)
 
        # Set inputs manually through InferRequest functionality to speedup
        infer_request_ptr = ie_encoder_exec.requests[0]
        for n, data in inputs.items():
            info_ptr = ie_encoder.input_info[n]
            blob = Blob(info_ptr.tensor_desc, data)
            infer_request_ptr.set_blob(n, blob, info_ptr.preprocess_info)
 
        # infer by IE
        infer_request_ptr.infer()
        res = infer_request_ptr.output_blobs
 
        if noise_suppress_flag:
            output_audio = (res['output'].buffer[0] * np.iinfo(np.int16).max).astype(np.int16)
        else:
            output_audio = input
        playback_stream.write(frames=output_audio, num_frames=input_size)
 
    record_stream.stop_stream()
    record_stream.close()
    playback_stream.stop_stream()
    playback_stream.close()
    audio.terminate()
 
if __name__ == '__main__':
    sys.exit(main() or 0)
 