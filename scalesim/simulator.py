import os

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.single_layer_sim import single_layer_sim as layer_sim
import numpy as np
import math


class simulator:
    def __init__(self):
        self.conf = cfg()
        self.topo = topo()
        self.output_path = "../"

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        self.num_layers = 0

        self.single_layer_sim_object_list = []

        self.params_set_flag = False
        self.all_layer_run_done = False

        #xuesi
        self.cycles_per_sample = 1

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   top_path="./",
                   output_path="../",
                   verbosity=True,
                   save_trace=True,
                   sampling_rate=1
                   ):

        self.conf = config_obj
        self.topo = topo_obj
        self.output_path = output_path

        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()

        self.params_set_flag = True

        #xuesi
        self.cycles_per_sample = sampling_rate

    #
    def run(self):
        assert self.params_set_flag, 'Simulator parameters are not set'

        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim()
            this_layer_sim.set_params(layer_id=i,
                                 config_obj=self.conf,
                                 topology_obj=self.topo,
                                 verbose=self.verbose,
                                 sampling_rate=self.cycles_per_sample,
                                 output_path=self.output_path)

            self.single_layer_sim_object_list.append(this_layer_sim)

        if not os.path.isdir(self.top_path):
            cmd = 'mkdir ' + self.top_path
            os.system(cmd)

        report_path = self.top_path + '/' + self.conf.get_run_name()

        if not os.path.isdir(report_path):
            cmd = 'mkdir ' + report_path
            os.system(cmd)

        self.top_path = report_path

        # 2. Run each layer
        # TODO: This is parallelizable
        for single_layer_obj in self.single_layer_sim_object_list:

            if self.verbose:
                layer_id = single_layer_obj.get_layer_id()
                print('\nRunning Layer ' + str(layer_id))

            single_layer_obj.run()

            if self.verbose:
                comp_items = single_layer_obj.get_compute_report_items()
                comp_cycles = comp_items[0]
                stall_cycles = comp_items[1]
                util = comp_items[2]
                mapping_eff = comp_items[3]
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) +'%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) +'%')

                avg_bw_items = single_layer_obj.get_bandwidth_report_items()
                avg_ifmap_bw = avg_bw_items[3]
                avg_filter_bw = avg_bw_items[4]
                avg_ofmap_bw = avg_bw_items[5]
                print('Average IFMAP DRAM BW: ' + "{:.3f}".format(avg_ifmap_bw) + ' words/cycle')
                print('Average Filter DRAM BW: ' + "{:.3f}".format(avg_filter_bw) + ' words/cycle')
                print('Average OFMAP DRAM BW: ' + "{:.3f}".format(avg_ofmap_bw) + ' words/cycle')

            if self.save_trace:
                if self.verbose:
                    print('Saving traces: ', end='')
                single_layer_obj.save_traces(self.top_path)
                if self.verbose:
                    print('Done!')

        self.all_layer_run_done = True

        self.generate_reports()

    #
    def generate_reports(self):
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        compute_report_name = self.top_path + '/COMPUTE_REPORT.csv'
        compute_report = open(compute_report_name, 'w')
        header = 'LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,\n'
        compute_report.write(header)

        bandwidth_report_name = self.top_path + '/BANDWIDTH_REPORT.csv'
        bandwidth_report = open(bandwidth_report_name, 'w')
        header = 'LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, '
        header += 'Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n'
        bandwidth_report.write(header)

        detail_report_name = self.top_path + '/DETAILED_ACCESS_REPORT.csv'
        detail_report = open(detail_report_name, 'w')
        header = 'LayerID, '
        header += 'SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, '
        header += 'SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, '
        header += 'SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, '
        header += 'DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, '
        header += 'DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, '
        header += 'DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n'
        detail_report.write(header)

        buffer_bw_array = np.zeros((3, 0), dtype=np.float64)
        for lid in range(len(self.single_layer_sim_object_list)):
            single_layer_obj = self.single_layer_sim_object_list[lid]
            compute_report_items_this_layer = single_layer_obj.get_compute_report_items()
            log = str(lid) +', '
            log += ', '.join([str(x) for x in compute_report_items_this_layer])
            log += ',\n'
            compute_report.write(log)

            bandwidth_report_items_this_layer = single_layer_obj.get_bandwidth_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bandwidth_report_items_this_layer])
            log += ',\n'
            bandwidth_report.write(log)

            detail_report_items_this_layer = single_layer_obj.get_detail_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in detail_report_items_this_layer])
            log += ',\n'
            detail_report.write(log)

            # xuesi's code starts
            lid_ifmap_sram_start_cycle = detail_report_items_this_layer[0]
            lid_ifmap_sram_stop_cycle = detail_report_items_this_layer[1]
            lid_ifmap_sram_reads = detail_report_items_this_layer[2]
            lid_filter_sram_start_cycle = detail_report_items_this_layer[3]
            lid_filter_sram_stop_cycle = detail_report_items_this_layer[4]
            lid_filter_sram_reads = detail_report_items_this_layer[5]
            lid_ofmap_sram_start_cycle = detail_report_items_this_layer[6]
            lid_ofmap_sram_stop_cycle = detail_report_items_this_layer[7]
            lid_ofmap_sram_writes = detail_report_items_this_layer[8]

            lid_cycle_min = min(lid_ifmap_sram_start_cycle, lid_filter_sram_start_cycle, lid_ofmap_sram_start_cycle)
            lid_cycle_max = max(lid_ifmap_sram_stop_cycle, lid_filter_sram_stop_cycle, lid_ofmap_sram_stop_cycle)

            lid_buffer_bw_array_sampled = np.zeros((3, math.ceil((int(lid_cycle_max) - int(lid_cycle_min) + 1) / self.cycles_per_sample)), dtype=np.float64)          

            lid_ifmap_bw_per_sample = lid_ifmap_sram_reads / (lid_ifmap_sram_stop_cycle - lid_ifmap_sram_start_cycle) * self.cycles_per_sample
            lid_filter_bw_per_sample = lid_filter_sram_reads / (lid_filter_sram_stop_cycle - lid_filter_sram_start_cycle) * self.cycles_per_sample
            lid_ofmap_bw_per_sample = lid_ofmap_sram_writes / (lid_ofmap_sram_stop_cycle - lid_ofmap_sram_start_cycle) * self.cycles_per_sample

            lid_ifmap_array = np.full((1, (math.ceil(int(lid_ifmap_sram_stop_cycle) / self.cycles_per_sample) - math.floor(int(lid_ifmap_sram_start_cycle) / self.cycles_per_sample))), lid_ifmap_bw_per_sample) 
            lid_filter_array = np.full((1, (math.ceil(int(lid_filter_sram_stop_cycle) / self.cycles_per_sample) - math.floor(int(lid_filter_sram_start_cycle) / self.cycles_per_sample))), lid_filter_bw_per_sample)
            lid_ofmap_array = np.full((1, (math.ceil(int(lid_ofmap_sram_stop_cycle) / self.cycles_per_sample) - math.floor(int(lid_ofmap_sram_start_cycle) / self.cycles_per_sample))), lid_ofmap_bw_per_sample)

            lid_buffer_bw_array_sampled[0, math.floor(int(lid_ifmap_sram_start_cycle) / self.cycles_per_sample) : math.ceil(int(lid_ifmap_sram_stop_cycle) / self.cycles_per_sample)] = lid_ifmap_array
            lid_buffer_bw_array_sampled[1, math.floor(int(lid_filter_sram_start_cycle) / self.cycles_per_sample) : math.ceil(int(lid_filter_sram_stop_cycle) / self.cycles_per_sample)] = lid_filter_array
            lid_buffer_bw_array_sampled[2, math.floor(int(lid_ofmap_sram_start_cycle) / self.cycles_per_sample) : math.ceil(int(lid_ofmap_sram_stop_cycle) / self.cycles_per_sample)] = lid_ofmap_array
            
            buffer_bw_array = np.append(buffer_bw_array, lid_buffer_bw_array_sampled, axis = 1)

        print("size of buffer_bw_array")
        print(buffer_bw_array.shape)
        buffer_bw_array_filename = self.output_path + "/bufferBandwidthArray.npy"
        np.save(buffer_bw_array_filename, buffer_bw_array)

        # xuesi's code ends

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()

    #
    def get_total_cycles(self):
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        total_cycles = 0
        for layer_obj in self.single_layer_sim_object_list:
            cycles_this_layer = int(layer_obj.get_compute_report_items[0])
            total_cycles += cycles_this_layer

        return total_cycles

