# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Benchmark script for ImageNet models on GPU.
see README.md for the usage and results of this script.
"""
import argparse
import threading
import sys
import time
import glob

import numpy as np

import os, os.path as osp
import tvm
import tvm.contrib.graph_runtime as runtime

from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata, download

from tvm import relay

try:
    import onnx
    import tarfile
    onnx_enabled = True
except:
    onnx_enabled = False

try:
    from PIL import Image
    image_input_enabled = True
except:
    image_input_enabled = False

from util import get_network

def get_onnx_network(network, batch_size):
    netname = network.split('onnx/')[1]

    nets = {'inception_v1' :
            ['https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz',
             'data_0',
             (1, 3, 224, 224)],
            'resnet-50' :
            ['https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz',
             'gpu_0/data_0',
             (1, 3, 224, 224)],
            'mobilenet_v2' :
            ['https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz',
             'data',
             (1, 3, 224, 224)]}

    onnx_url, input_name, input_shape = nets[netname]

    onnx_tar_gz = osp.split(onnx_url)[1]
    onnx_path = download_testdata(onnx_url, onnx_tar_gz, module='onnx')

    mdir = osp.dirname(onnx_path) + "/" + onnx_tar_gz.replace(".tar.gz", "")
    mpath = glob.glob(mdir + "/*.onnx")
    if len(mpath) < 1:
        tar = tarfile.open(onnx_path)
        tar.extractall(osp.dirname(onnx_path))
        tar.close()
        mpath = glob.glob(mdir + "/*.onnx")
    assert len(mpath) == 1
    onnx_model = onnx.load(mpath[0])

    shape_dict = {input_name: input_shape}
    net, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    return (net, params, input_shape, input_name)

def load_and_transform_image(input_f, network_name, input_shape):

    img = Image.open(input_f).resize(input_shape[2:4])

    img = img.convert('RGB')
    img = np.array(img).astype(float)
    # Default 0.0 to 255.0 per channel.

    if network_name in ('onnx/resnet-50', 'onnx/mobilenet_v2'):
        # Normalize to 0 .. 1.0 per channel.
        img /= np.array([255.0, 255.0, 255.0])

    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :].astype('float32')
    return tvm.nd.array(img)

def benchmark(network, target):

    if network.startswith('onnx/'):
        net, params, input_shape, input_name = get_onnx_network(network, batch_size=1)
    else:
        net, params, input_shape, output_shape = get_network(network, batch_size=1)
        input_name = 'data'

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(net, target=target, params=params)

    # create runtime
    ctx = tvm.context(str(target), 0)
    module = runtime.create(graph, lib, ctx)

    if input_fn is None:
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    else:
        data_tvm = load_and_transform_image(input_fn, network, input_shape)

    data_set_start = time.clock()
    module.set_input(input_name, data_tvm)
    module.set_input(**params)
    data_set_end = time.clock()
    t = data_set_end - data_set_start

    # verification run, if requested
    if args.verify_index:
        module.run()
        tvm_out = module.get_output(0)
        top1_index = np.argmax(tvm_out.asnumpy()[0])
        if top1_index == args.verify_index:
            print("Verify: Labelled %d as expected.\n" % top1_index)
        else:
            synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
            synset_name = 'imagenet1000_clsid_to_human.txt'
            synset_path = download_testdata(synset_url, synset_name, module='data')
            with open(synset_path) as f:
                synset = eval(f.read())
            print("Verify: Labelled %d (%s) while expecting %d (%s).\n" %
                  (top1_index, synset[top1_index], args.verify_index, synset[args.verify_index]),
                  file=sys.stderr)

    # evaluate
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=args.repeat)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
    print("best: %.2f ms + input setting %.2f ms = %.2f ms total" % (np.min(prof_res), t * 1000, np.min(prof_res) + t * 1000))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    onnx_networks = []
    if onnx_enabled:
        onnx_networks = ['onnx/inception_v1', 'onnx/resnet-50', 'onnx/mobilenet_v2']

    parser.add_argument("--network", type=str, choices=
                        ['resnet-18', 'resnet-34', 'resnet-50',
                         'vgg-16', 'vgg-19', 'densenet-121', 'inception_v3',
                         'mobilenet', 'squeezenet_v1.0', 'squeezenet_v1.1'] + onnx_networks,
                        help='The name of neural network')

    parser.add_argument("--model", type=str,
                        choices=['1080ti', 'titanx', 'tx2', 'gfx900'], default='1080ti',
                        help="The model of the test device. If your device is not listed in "
                             "the choices list, pick the most similar one as argument.")
    parser.add_argument("--repeat", type=int, default=600)
    parser.add_argument("--target", type=str,
                        choices=['cuda', 'opencl', 'rocm', 'nvptx', 'metal'], default='cuda',
                        help="The tvm compilation target")
    parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")

    if image_input_enabled:
        parser.add_argument("--verify-index", type=int, default=None,
                            help="Verify that the given image was recognized with the given label index.")
        parser.add_argument("--input-img", default=None, type=argparse.FileType('rb'))


    args = parser.parse_args()

    dtype = 'float32'

    if args.network is None:
        networks = ['resnet-50', 'mobilenet', 'vgg-19', 'inception_v3']
    else:
        networks = [args.network]

    target = tvm.target.create('%s -model=%s' % (args.target, args.model))

    input_fn = None
    if image_input_enabled:
        if args.verify_index and args.input_img is None:
            print("Must give an input image with a known label to verify.")
            sys.os.exit(1)
        input_fn = args.input_img

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")

    for network in networks:
        if args.thread == 1:
            benchmark(network, target)
        else:
            threads = list()
            for n in range(args.thread):
                thread = threading.Thread(target=benchmark, args=([network, target]), name="thread%d" % n)
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
