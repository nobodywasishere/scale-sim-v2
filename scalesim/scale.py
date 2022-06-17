#!/usr/bin/env python
import argparse

from scalesim.scale_sim import scalesim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="../topologies/conv_nets/test.csv",
                        help="Path to the topology file"
                        )
    parser.add_argument('-c', metavar='Config file', type=str,
                        default="../configs/scale.cfg",
                        help="Path to the config file"
                        )
    parser.add_argument('-p', metavar='log dir', type=str,
                        default="../test_runs",
                        help="Path to log dir"
                        )
    parser.add_argument('-i', metavar='input type', type=str,
                        default="conv",
                        help="Type of input topology, gemm: MNK, conv: conv"
                        )
    parser.add_argument('-o', metavar='.npy files output dir', type=str,
                        default="../",
                        help="Path to .npy files"
                        )
    parser.add_argument('-s', metavar='sampling rate', type=int,
                        default=1,
                        help="sampling rate for activity map and buffer"
                        )

    args = parser.parse_args()
    topology = args.t
    config = args.c
    logpath = args.p
    inp_type = args.i
    output_path = args.o
    sampling_rate = args.s

    gemm_input = False
    if inp_type == 'gemm':
        gemm_input = True

    s = scalesim(save_disk_space=True, verbose=True,
                 config=config,
                 output_path=output_path,
                 sampling_rate=sampling_rate,
                 topology=topology,
                 input_type_gemm=gemm_input
                 )
    s.run_scale(top_path=logpath)
