import time
import os
import shutil
import math
import numpy as np
from tqdm import tqdm

from scalesim.memory.read_buffer import read_buffer as rdbuf
from scalesim.memory.read_buffer_estimate_bw import ReadBufferEstimateBw as rdbuf_est
from scalesim.memory.read_port import read_port as rdport
from scalesim.memory.write_buffer import write_buffer as wrbuf
from scalesim.memory.write_port import write_port as wrport

from copy import deepcopy as dc
# import seaborn as sns

class double_buffered_scratchpad:
    def __init__(self):
        self.layer_id = 0

        self.ifmap_buf = rdbuf()
        self.filter_buf = rdbuf()
        self.ofmap_buf =wrbuf()

        self.ifmap_port = rdport()
        self.filter_port = rdport()
        self.ofmap_port = wrport()

        self.verbose = True
        self.output_path = "../"

        self.ifmap_trace_matrix = np.zeros((1,1), dtype=np.int)
        self.filter_trace_matrix = np.zeros((1,1), dtype=np.int)
        self.ofmap_trace_matrix = np.zeros((1,1), dtype=np.int)

        # Metrics to gather for generating run reports
        self.total_cycles = 0
        self.compute_cycles = 0
        self.stall_cycles = 0

        self.avg_ifmap_dram_bw = 0
        self.avg_filter_dram_bw = 0
        self.avg_ofmap_dram_bw = 0

        self.ifmap_sram_start_cycle = 0
        self.ifmap_sram_stop_cycle = 0
        self.filter_sram_start_cycle = 0
        self.filter_sram_stop_cycle = 0
        self.ofmap_sram_start_cycle = 0
        self.ofmap_sram_stop_cycle = 0

        self.ifmap_dram_start_cycle = 0
        self.ifmap_dram_stop_cycle = 0
        self.ifmap_dram_reads = 0
        self.filter_dram_start_cycle = 0
        self.filter_dram_stop_cycle = 0
        self.filter_dram_reads = 0
        self.ofmap_dram_start_cycle = 0
        self.ofmap_dram_stop_cycle = 0
        self.ofmap_dram_writes = 0

        self.estimate_bandwidth_mode = False,
        self.traces_valid = False
        self.params_valid_flag = True

        #maz
        #self.per_mac_compute_array_for_all_steps

        #xuesi
        self.cycles_per_sample = 1

    #
    def set_params(self,
                   verbose=True,
                   layer_id = 0,
                   estimate_bandwidth_mode=False,
                   word_size=1,
                   ifmap_buf_size_bytes=2, filter_buf_size_bytes=2, ofmap_buf_size_bytes=2,
                   rd_buf_active_frac=0.5, wr_buf_active_frac=0.5,
                   ifmap_backing_buf_bw=1, filter_backing_buf_bw=1, ofmap_backing_buf_bw=1,
                   sampling_rate=1,
                   output_path="../"):

        #breakpoint()
        self.estimate_bandwidth_mode = estimate_bandwidth_mode
        self.cycles_per_sample = sampling_rate
        self.output_path = output_path

        if self.estimate_bandwidth_mode:
            self.ifmap_buf = rdbuf_est()
            self.filter_buf = rdbuf_est()

            self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
                                      total_size_bytes=ifmap_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_default_bw=ifmap_backing_buf_bw)

            self.filter_buf.set_params(backing_buf_obj=self.filter_port,
                                       total_size_bytes=filter_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_default_bw=filter_backing_buf_bw)
        else:
            self.ifmap_buf = rdbuf()
            self.filter_buf = rdbuf()

            self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
                                      total_size_bytes=ifmap_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_bw=ifmap_backing_buf_bw)

            self.filter_buf.set_params(backing_buf_obj=self.filter_port,
                                       total_size_bytes=filter_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_bw=filter_backing_buf_bw)

        self.ofmap_buf.set_params(backing_buf_obj=self.ofmap_port,
                                  total_size_bytes=ofmap_buf_size_bytes,
                                  word_size=word_size,
                                  active_buf_frac=wr_buf_active_frac,
                                  backing_buf_bw=ofmap_backing_buf_bw)

        self.verbose = verbose
        self.layer_id = layer_id

        self.params_valid_flag = True

    #
    def set_read_buf_prefetch_matrices(self,
                                       ifmap_prefetch_mat=np.zeros((1,1)),
                                       filter_prefetch_mat=np.zeros((1,1))
                                       ):

        self.ifmap_buf.set_fetch_matrix(ifmap_prefetch_mat)
        self.filter_buf.set_fetch_matrix(filter_prefetch_mat)

    #
    def reset_buffer_states(self):

        self.ifmap_buf.reset()
        self.filter_buf.reset()
        self.ofmap_buf.reset()

    # The following are just shell methods for users to control each mem individually
    def service_ifmap_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        out_cycles_arr_np = self.ifmap_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_filter_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        out_cycles_arr_np = self.filter_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_ofmap_writes(self,
                             incoming_requests_arr_np,  # 2D array with the requests
                             incoming_cycles_arr):

        out_cycles_arr_np = self.ofmap_buf.service_writes(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #maz
    def compute_step(self, array, row_num, col_num):
        for i in range(row_num):
            for j in range(col_num-1, -1, -1):
                array[i][j]['ifmap'] = array[i][j-1]['ifmap']
        for j in range(col_num):
            for i in range(row_num-1, -1, -1):
                array[i][j]['filter'] = array[i-1][j]['filter']
        return array

    def service_memory_requests(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        r""" needs short description

        Parameters
        ----------
        ifmap_demand_mat
            which ifmap addresses are used in each step
        filter_demand_mat
            which filter addresses are used in each step
        ofmap_demand_mat
            which ofmap addresses are used in each step
        """
        assert self.params_valid_flag, 'Memories not initialized yet'

        #breakpoint()

        #maz
        keys = ['ifmap', 'filter', 'is_active']
        values = [-1, -1, 0]
        mac_compute_dict = dict(zip(keys, values))
        systolic_array_row_number = ifmap_demand_mat.shape[1]
        systolic_array_col_number = filter_demand_mat.shape[1]
        per_mac_compute_array_for_one_step = []
        
        for i in range(systolic_array_row_number):
            mac_compute_row = []
            for j in range(systolic_array_col_number):
                mac_compute_row.append(mac_compute_dict.copy())
            per_mac_compute_array_for_one_step.append(mac_compute_row)
        per_mac_compute_array_for_all_steps = []
        
        # maz
        # index 0 of ofmap_demand_mat shows n * m 
        # where n is the total_number_of_folds (row_folds + col_folds)
        # and m is the number_of_compute_steps_in_each_fold
        ofmap_lines = ofmap_demand_mat.shape[0]

        for i in range(systolic_array_row_number):
            mac_compute_row = []
            for j in range(systolic_array_col_number):
                mac_compute_row.append(mac_compute_dict.copy())
            per_mac_compute_array_for_one_step.append(mac_compute_row)
        per_mac_compute_array_for_all_steps = []


        self.total_cycles = 0
        self.stall_cycles = 0
        self.steps = 0

        # maz & xuesi
        # hit latency values seem to be dependant on whether we are using
        # estimated BW or not.
        # However I did not find it to be set to a value other than 1.
        ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
        filter_hit_latency = self.ifmap_buf.get_hit_latency()

        ifmap_serviced_cycles = []
        filter_serviced_cycles = []
        ofmap_serviced_cycles = []
        
        outputDir = self.output_path
        if (self.layer_id == 0):
            activityArray = np.zeros([10, systolic_array_row_number, systolic_array_col_number], dtype=np.uint16)
            curr_cycle = 0 # variable used to index into persistant activity array
        else:
            try:
                activityArray_fname = outputDir + "/activityArray.npy"
                activityArray = np.load(activityArray_fname)                
            except FileNotFoundError:
                print("Error: Expected activityArray at {} but not found.".format(activityArray_fname))
                exit()
            try:
                curr_cycle_fname = "{}/curr_cycle.npy".format(outputDir)
                curr_cycle = np.load(curr_cycle_fname)[0]
            except FileNotFoundError:
                print("Error: Expected curr_cycle at {} but not found.".format(curr_cycle_fname))
                exit()
                
        pbar_disable = not self.verbose
        for i in tqdm(range(ofmap_lines), disable=pbar_disable):
            
            if (activityArray.shape[0]-1 < math.floor(curr_cycle/self.cycles_per_sample)):
                [samples, rows, cols] = activityArray.shape
                activityArray.resize([samples+10, rows, cols])
            #breakpoint()
            cycle_arr = np.zeros((1,1)) + i + self.stall_cycles

            ifmap_demand_line = ifmap_demand_mat[i, :].reshape((1,ifmap_demand_mat.shape[1]))
            ifmap_cycle_out = self.ifmap_buf.service_reads(incoming_requests_arr_np=ifmap_demand_line,
                                                            incoming_cycles_arr=cycle_arr)
            ifmap_serviced_cycles += [ifmap_cycle_out[0]]
            ifmap_stalls = ifmap_cycle_out[0] - cycle_arr[0] - ifmap_hit_latency

            filter_demand_line = filter_demand_mat[i, :].reshape((1, filter_demand_mat.shape[1]))
            filter_cycle_out = self.filter_buf.service_reads(incoming_requests_arr_np=filter_demand_line,
                                                           incoming_cycles_arr=cycle_arr)
            filter_serviced_cycles += [filter_cycle_out[0]]
            filter_stalls = filter_cycle_out[0] - cycle_arr[0] - filter_hit_latency

            ofmap_demand_line = ofmap_demand_mat[i, :].reshape((1, ofmap_demand_mat.shape[1]))
            ofmap_cycle_out = self.ofmap_buf.service_writes(incoming_requests_arr_np=ofmap_demand_line,
                                                             incoming_cycles_arr_np=cycle_arr)
            ofmap_serviced_cycles += [ofmap_cycle_out[0]]
            ofmap_stalls = ofmap_cycle_out[0] - cycle_arr[0] - 1

            self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))

            #maz
            this_step_stall = int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))

            # reset the activity of all cells at the beginning of each step
            # for k in range(systolic_array_row_number):
            #     for j in range(systolic_array_col_number):
            #         per_mac_compute_array_for_one_step[k][j]['is_active'] = 0

            for k in range(this_step_stall):
                per_mac_compute_array_for_all_steps.append(dc(per_mac_compute_array_for_one_step))
                
            per_mac_compute_array_for_one_step = self.compute_step(per_mac_compute_array_for_one_step, systolic_array_row_number, systolic_array_col_number)

            for idx, val in enumerate(ifmap_demand_line[0]):
                per_mac_compute_array_for_one_step[idx][0]['ifmap'] = val
            for idx, val in enumerate(filter_demand_line[0]):
                per_mac_compute_array_for_one_step[0][idx]['filter'] = val
            
            for k in range(systolic_array_row_number):
                for j in range(systolic_array_col_number):
                    if (per_mac_compute_array_for_one_step[k][j]['ifmap'] != -1 and 
                        per_mac_compute_array_for_one_step[k][j]['filter'] != -1):
                        per_mac_compute_array_for_one_step[k][j]['is_active'] = 1
                        activityArray[math.floor(curr_cycle/self.cycles_per_sample)][k][j] += 1

                        # if sample_cycle_count > self.cycles_per_sample: 
                        #     activityArray[sample_count] = activityArray_per_sample
                        #     activityArray_per_sample = np.zeros((systolic_array_row_number, systolic_array_col_number), dtype=np.int8)
                        #     sample_count += 1
                        #     sample_cycle_count = 0
                            

                        # activityArray[i][k][j] = 1
                    
            self.steps += 1
            curr_cycle += 1
            # ax = sns.heatmap(sumArray, cmap="GnBu", linewidths=.5, cbar=False)
            # ax.figure.savefig('../figs/activity_' + str(i).zfill(5) + '.png', format='png', bbox_inches='tight', pad_inches=0)
            
            # per_mac_compute_array_for_all_steps.append(dc(per_mac_compute_array_for_one_step))
            #breakpoint()
        cwd = os.getcwd()
        outputDir = self.output_path + "/activityArrays"
        if os.path.exists(outputDir) and self.layer_id == 0:
            shutil.rmtree(outputDir)
            os.makedirs(outputDir)
        os.chdir(self.output_path)
        activityArray_name = "activityArray.npy"
        np.save(activityArray_name, activityArray)
        np.save("curr_cycle.npy", [curr_cycle])
        os.chdir(cwd)
        print("size of activityArray")
        print(activityArray.shape)

        if self.estimate_bandwidth_mode:
            # IDE shows warning as complete_all_prefetches is not implemented in read_buffer class
            # It is harmless since, in estimate bandwidth mode, read_buffer_estimate_bw is instantiated
            self.ifmap_buf.complete_all_prefetches()
            self.filter_buf.complete_all_prefetches()

        self.ofmap_buf.empty_all_buffers(ofmap_serviced_cycles[-1])

        # Prepare the traces
        ifmap_services_cycles_np = np.asarray(ifmap_serviced_cycles).reshape((len(ifmap_serviced_cycles), 1))
        self.ifmap_trace_matrix = np.concatenate((ifmap_services_cycles_np, ifmap_demand_mat), axis=1)

        filter_services_cycles_np = np.asarray(filter_serviced_cycles).reshape((len(filter_serviced_cycles), 1))
        self.filter_trace_matrix = np.concatenate((filter_services_cycles_np, filter_demand_mat), axis=1)

        ofmap_services_cycles_np = np.asarray(ofmap_serviced_cycles).reshape((len(ofmap_serviced_cycles), 1))
        self.ofmap_trace_matrix = np.concatenate((ofmap_services_cycles_np, ofmap_demand_mat), axis=1)
        self.total_cycles = int(ofmap_serviced_cycles[-1][0])

        # END of serving demands from memory
        #sumArray = np.zeros((32, 32))
        #for i in range(systolic_array_row_number):
        #    for j in range(systolic_array_col_number):
        #        for k in range(ofmap_lines):
        #            sumArray[i][j] += per_mac_compute_array_for_all_steps[k][i][j]['is_active']
        # breakpoint()
        self.traces_valid = True



    # This is the trace computation logic of this memory system
    # Anand: This is too complex, perform the serve cycle by cycle for the requests
    def service_memory_requests_old(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        # TODO: assert sanity check
        assert self.params_valid_flag, 'Memories not initialized yet'

        # Logic:
        # Stalls can occur in both read and write portions and interfere with each other
        # We mitigate interference by picking a window in which there are no write stall,
        # ie, there is sufficient free space in the write buffer

        ofmap_lines_remaining = ofmap_demand_mat.shape[0]       # The three demand mats have the same shape though
        start_line_idx = 0
        end_line_idx = 0

        first = True
        cycle_offset = 0
        self.total_cycles = 0
        self.stall_cycles = 0

        # Status bar
        pbar_disable = not self.verbose #or True
        pbar = tqdm(total=ofmap_lines_remaining, disable=pbar_disable)

        avg_read_time_series = []

        while ofmap_lines_remaining > 0:
            loop_start_time = time.time()
            ofmap_free_space = self.ofmap_buf.get_free_space()

            # Find the number of lines till the ofmap_free_space is filled up
            count = 0
            while not count > ofmap_free_space:
                this_line = ofmap_demand_mat[end_line_idx]
                for elem in this_line:
                    if not elem == -1:
                        count += 1

                if not count > ofmap_free_space:
                    end_line_idx += 1
                    # Limit check
                    if not end_line_idx < ofmap_demand_mat.shape[0]:
                        end_line_idx = ofmap_demand_mat.shape[0] - 1
                        count = ofmap_free_space + 1
                else:   # Send request with minimal data ie one line of the requests
                    end_line_idx += 1
            # END of line counting

            num_lines = end_line_idx - start_line_idx + 1
            this_req_cycles_arr = [int(x + cycle_offset) for x in range(num_lines)]
            this_req_cycles_arr_np = np.asarray(this_req_cycles_arr).reshape((num_lines,1))

            this_req_ifmap_demands = ifmap_demand_mat[start_line_idx:(end_line_idx + 1), :]
            this_req_filter_demands = filter_demand_mat[start_line_idx:(end_line_idx + 1), :]
            this_req_ofmap_demands = ofmap_demand_mat[start_line_idx:(end_line_idx + 1), :]

            no_stall_cycles = num_lines     # Since the cycles are consecutive at this point

            time_start = time.time()
            ifmap_cycles_out = self.ifmap_buf.service_reads(incoming_requests_arr_np=this_req_ifmap_demands,
                                                            incoming_cycles_arr=this_req_cycles_arr_np)
            time_end = time.time()
            delta = time_end - time_start
            avg_read_time_series.append(delta)

            # Take care of the incurred stalls when launching demands for filter_reads
            # Note: Stalls incurred on reading line i in ifmap reflect the request cycles for line i+1 in filter
            ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
            ifmap_stalls = ifmap_cycles_out - this_req_cycles_arr_np - ifmap_hit_latency    # Vec - vec - scalar
            ifmap_stalls = np.concatenate((np.zeros((1,1)), ifmap_stalls[0:-1]), axis=0)    # Shift by one row
            this_req_cycles_arr_np = this_req_cycles_arr_np + ifmap_stalls

            time_start = time.time()
            filter_cycles_out = self.filter_buf.service_reads(incoming_requests_arr_np=this_req_filter_demands,
                                                              incoming_cycles_arr=this_req_cycles_arr_np)
            time_end = time.time()
            delta = time_end - time_start
            avg_read_time_series.append(delta)

            # Take care of stalls again --> The entire array stops when there is a stall
            filter_hit_latency = self.filter_buf.get_hit_latency()
            filter_stalls = filter_cycles_out - this_req_cycles_arr_np - filter_hit_latency  # Vec - vec - scalar
            filter_stalls = np.concatenate((np.zeros((1, 1)), filter_stalls[0:-1]), axis=0)  # Shift by one row
            this_req_cycles_arr_np = this_req_cycles_arr_np + filter_stalls

            ofmap_cycles_out = self.ofmap_buf.service_writes(incoming_requests_arr_np=this_req_ofmap_demands,
                                                             incoming_cycles_arr_np=this_req_cycles_arr_np)

            # Make the trace matrices
            this_req_ifmap_trace_matrix = np.concatenate((ifmap_cycles_out, this_req_ifmap_demands), axis=1)
            this_req_filter_trace_matrix = np.concatenate((filter_cycles_out, this_req_filter_demands), axis=1)
            this_req_ofmap_trace_matrix = np.concatenate((ofmap_cycles_out, this_req_ofmap_demands), axis=1)

            actual_cycles = ofmap_cycles_out[-1][0] - this_req_cycles_arr_np[0][0] + 1
            num_stalls = actual_cycles - no_stall_cycles

            self.stall_cycles += num_stalls
            self.total_cycles = ofmap_cycles_out[-1][0] + 1         # OFMAP is served the last

            if first:
                first = False
                self.ifmap_trace_matrix = this_req_ifmap_trace_matrix
                self.filter_trace_matrix = this_req_filter_trace_matrix
                self.ofmap_trace_matrix = this_req_ofmap_trace_matrix
            else:
                self.ifmap_trace_matrix = np.concatenate((self.ifmap_trace_matrix, this_req_ifmap_trace_matrix), axis=0)
                self.filter_trace_matrix = np.concatenate((self.filter_trace_matrix, this_req_filter_trace_matrix), axis=0)
                self.ofmap_trace_matrix = np.concatenate((self.ofmap_trace_matrix, this_req_ofmap_trace_matrix), axis=0)

            # Update the local variable for another iteration of the while loop
            cycle_offset = ofmap_cycles_out[-1][0] + 1
            start_line_idx = end_line_idx + 1

            pbar.update(num_lines)
            ofmap_lines_remaining = max(ofmap_demand_mat.shape[0] - (end_line_idx + 1), 0)    # Cutoff at 0
            #print("DEBUG: " + str(end_line_idx))

            if end_line_idx > ofmap_demand_mat.shape[0]:
                print('Trap')

            #if int(ofmap_lines_remaining % 1000) == 0:
            #    print("DEBUG: " + str(ofmap_lines_remaining))

            loop_end_time = time.time()
            loop_time = loop_end_time - loop_start_time
            #print('DEBUG: Time taken in one iteration: ' + str(loop_time))

        # At this stage there might still be some data in the active buffer of the OFMAP scratchpad
        # The following drains it and generates the OFMAP
        drain_start_cycle = self.ofmap_trace_matrix[-1][0] + 1
        self.ofmap_buf.empty_all_buffers(drain_start_cycle)

        #avg_read_time = sum(avg_read_time_series) / len(avg_read_time_series)
        #print('DEBUG: Avg time to service reads= ' + str(avg_read_time))

        pbar.close()
        # END of serving demands from memory
        self.traces_valid = True

    #
    def get_total_compute_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.total_cycles

    #
    def get_stall_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.stall_cycles

    #
    def get_ifmap_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.ifmap_trace_matrix.shape[0]):
            if done:
                break
            row = self.ifmap_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.ifmap_sram_start_cycle = self.ifmap_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.ifmap_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.ifmap_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.ifmap_sram_stop_cycle  = self.ifmap_trace_matrix[ridx][0]
                    done = True
                    break

        return self.ifmap_sram_start_cycle, self.ifmap_sram_stop_cycle

    #
    def get_filter_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.filter_trace_matrix.shape[0]):
            if done:
                break
            row = self.filter_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.filter_sram_start_cycle = self.filter_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.filter_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.filter_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.filter_sram_stop_cycle = self.filter_trace_matrix[ridx][0]
                    done = True
                    break

        return self.filter_sram_start_cycle, self.filter_sram_stop_cycle

    #
    def get_ofmap_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.ofmap_trace_matrix.shape[0]):
            if done:
                break
            row = self.ofmap_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.ofmap_sram_start_cycle = self.ofmap_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.ofmap_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.ofmap_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.ofmap_sram_stop_cycle = self.ofmap_trace_matrix[ridx][0]
                    done = True
                    break

        return self.ofmap_sram_start_cycle, self.ofmap_sram_stop_cycle

    #
    def get_ifmap_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.ifmap_dram_reads = self.ifmap_buf.get_num_accesses()
        self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle \
            = self.ifmap_buf.get_external_access_start_stop_cycles()

        return self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle, self.ifmap_dram_reads

    #
    def get_filter_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.filter_dram_reads = self.filter_buf.get_num_accesses()
        self.filter_dram_start_cycle, self.filter_dram_stop_cycle \
            = self.filter_buf.get_external_access_start_stop_cycles()

        return self.filter_dram_start_cycle, self.filter_dram_stop_cycle, self.filter_dram_reads

    #
    def get_ofmap_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.ofmap_dram_writes = self.ofmap_buf.get_num_accesses()
        self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle \
            = self.ofmap_buf.get_external_access_start_stop_cycles()

        return self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle, self.ofmap_dram_writes

    #
    def get_ifmap_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix

    #
    def get_filter_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.filter_trace_matrix

    #
    def get_ofmap_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ofmap_trace_matrix

    #
    def get_sram_trace_matrices(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix, self.filter_trace_matrix, self.ofmap_trace_matrix

    #
    def get_ifmap_dram_trace_matrix(self):
        return self.ifmap_buf.get_trace_matrix()

    #
    def get_filter_dram_trace_matrix(self):
        return self.filter_buf.get_trace_matrix()

    #
    def get_ofmap_dram_trace_matrix(self):
        return self.ofmap_buf.get_trace_matrix()

    #
    def get_dram_trace_matrices(self):
        dram_ifmap_trace = self.ifmap_buf.get_trace_matrix()
        dram_filter_trace = self.filter_buf.get_trace_matrix()
        dram_ofmap_trace = self.ofmap_buf.get_trace_matrix()

        return dram_ifmap_trace, dram_filter_trace, dram_ofmap_trace

        #
    def print_ifmap_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ifmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_filter_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.filter_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ofmap_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ofmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ifmap_dram_trace(self, filename):
        self.ifmap_buf.print_trace(filename)

    #
    def print_filter_dram_trace(self, filename):
        self.filter_buf.print_trace(filename)

    #
    def print_ofmap_dram_trace(self, filename):
        self.ofmap_buf.print_trace(filename)





