# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Gossip Buffer

:author: Mido Assran
:description: Class defines a shared-memory Gossip-Buffer, which allows
    multi-processed asynchronous agents on the same machine to communicate
    tensors to on one-another
"""

import copy
import torch
import time
import pickle
import base64
import os
def encode(content):
    content = pickle.dumps(content)
    result = base64.b64encode(content).decode()
    return result

def decode(content):
    temp = base64.b64decode(content)
    result = pickle.loads(temp)
    return result

class GossipBuffer():

    def __init__(self, topology, model, read_events,
                 write_events, sync_list, proc_manager ,sync_freq=0, num_nodes = 1, msg_buffer = 0):
        """ GossipBuffer """

        self.topology = topology
        self.num_learners = len(topology) # all learners in the cluster
        self.sync_list = sync_list
        self.sync_freq = sync_freq

        self.aggregate_time_start = 0
        self.aggregate_time_end = 0
        self.aggregate_circle_time_start = 0
        self.aggregate_circle_time_end = 0
        self.num_nodes = num_nodes
        self.num_learner_per_node = int(self.num_learners/self.num_nodes)
        # Initialize message buffer (4-item object):
        # [0] -> Msg-Tensor
        # [1] -> Events recording peers that have read the message
        # [2] -> Events recording peer that has written the message
        # [3] -> Lock for safe access of Msg-Tensor
        #self.msg_buffer = []
        self.read_events = read_events
        self.write_events = write_events
        self.msg_buffer = proc_manager.get_msg_buffer()
        self.aggregate_counter = 0
        print('num_learners', self.num_learners)

    def write_message(self, rank, model, rotate=False):
        """
        Write agent 'rank's 'model' to a local 'boradcast buffer' that will be
        read by the out-neighbours defined in 'self.topology'.

        :param rank: Agent's rank in multi-agent graph topology
        :param model: Agent's torch neural network model
        :param rotate: Whether to alternate peers in graph topology

        Agents should only write to their own broadcast buffer:
            i.e., ensure 'model' belongs to agent 'rank'
        WARNING: setting rotate=True with sync_freq > 1 not currently supported
        """
        with torch.no_grad():
            # Get local broadcast-buffer
            #MAYBE WE NEED to LOAD event from server directly
            broadcast_buffer = copy.deepcopy(self.msg_buffer[rank])
            broadcast_buffer = decode(broadcast_buffer)

            read_event_list = self.read_events[rank*self.num_learners : (rank+1)*self.num_learners]
            write_event_list = self.write_events[rank*self.num_learners : (rank+1)*self.num_learners]

            # Check if out-peers finished reading our last message
            out_peers, _ = self.topology[rank].get_peers()
            #print(os.getpid(),'outpeers', out_peers)
            read_complete = True
            for peer in out_peers:
                if not read_event_list[peer] == True: #todo this only define it's true or false so we don't need to
                    read_complete = False
                    break
            # If peers done reading our last message, wait and clear events
            if read_complete:
                for peer in out_peers:
                    #todo use itself to simulate event
                    while True:
                        if read_event_list[peer] == True:
                            break
                        else:
                            pass
                    read_event_list[peer] = False
                    self.read_events[rank*self.num_learners+peer] = False #broadcast the change to network
            # If not done reading, cannot write another message right now
            else:
                return

            # Update broadcast-buffer with new message
            # -- flatten params and multiply by mixing-weight
            num_peers = self.topology[rank].peers_per_itr
            for bp, p in zip(broadcast_buffer.parameters(),
                             model.parameters()):
                bp.data.copy_(p)
                bp.data.div_(num_peers + 1)
            # -- mark message as 'written'
            self.msg_buffer[rank] = encode(broadcast_buffer)
            out_peers, _ = self.topology[rank].get_peers(rotate)
            torch.cuda.current_stream().synchronize()
            for peer in out_peers:
                #todo use itself simulate event
                write_event_list[peer] = True
                self.write_events[rank*self.num_learners + peer] = True


    def aggregate_message(self, rank, model):
        """
        Average messages with local model:
        Average all in-neighbours' (defined in 'self.topology') parameters with
        agent 'rank's 'model' and copy the result into 'model'.

        Agents should only aggregate messages into their own model:
            i.e., ensure 'model belongs to agent 'rank'
        """
        with torch.no_grad():
            # Check if in-peers finished writing messages to broadcast buffers
            _, in_peers = self.topology[rank].get_peers()
            write_complete = True
            self.aggregate_circle_time_start = time.time()
            for peer in in_peers:
                write_event = self.write_events[peer*self.num_learners + rank]
                #todo use itself to simulate event
                if not write_event == True:
                    write_complete = False #如果节点还没有set，则没写完
                    break
            # Check if any messages are excessively stale
            stale_assert = self.sync_list[rank] >= self.sync_freq
            
            self.aggregate_circle_time_end = time.time()
            # If peers done writing or message too stale, wait and clear events
            if write_complete or stale_assert:
                for peer in in_peers:
                    while True:
                         #todo use peerbuffer itself to simulate event
                        if self.write_events[peer*self.num_learners + rank] == True:
                            break
                        else:
                            pass
                    self.write_events[peer*self.num_learners + rank] = False
                    self.sync_list[rank] = 0
            # Not done writing, but staleness is still tolerable
            else:
                self.sync_list[rank] += 1
                print('%s: staleness %s' % (rank, self.sync_list[rank]))
                return
            
            self.aggregate_time_start = time.time()
            # Lazy-mixing of local params
            num_peers = self.topology[rank].peers_per_itr
            for p in model.parameters():
                p.data.div_(num_peers + 1)

            # Aggregate received messages
            for peer in in_peers:
                # Read message and update 'params'
                peer_msg = copy.deepcopy(self.msg_buffer[peer])
                peer_msg = decode(peer_msg)
                peer_msg.to('cuda:0')
                model.to('cuda:0')
                for p, bp in zip(model.parameters(),
                                 peer_msg.parameters()):
                    p.data.add_(bp.to(p.device, non_blocking=True))
                self.aggregate_counter += 1
                torch.cuda.current_stream().synchronize()
                # Mark message as 'read'
                 #聚合完成，设置为可写入
                #todo use itself to simulate
                self.read_events[peer*self.num_learners + rank] = True
            self.aggregate_time_end = time.time()

