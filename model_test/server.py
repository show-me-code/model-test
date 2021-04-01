# -*-coding:utf-8-*-
import torch.multiprocessing as mp
from multiprocessing.managers import ListProxy, BarrierProxy, AcquirerProxy, EventProxy
from gala.arguments import get_args
import torch
from gala.model import Policy
import base64
import json
import pickle
mp.current_process().authkey = b'abc'


def encode(content):
    content = pickle.dumps(content)
    result = base64.b64encode(content).decode()
    return result

def decode(content):
    temp = base64.b64decode(content)
    result = pickle.loads(temp)
    return result

def server(manager,host, port, key, args):
    #barrier = manager.Barrier(4)
    '''sync_list = manager.list()
    buffer_locks = manager.list()
    read_events = manager.list()
    write_events = manager.list()'''
    num_learners = args.num_learners * args.num_nodes

    sync_list = manager.list([0 for _ in range(num_learners)])

    #buffer_locks = manager.list([manager.Lock() for _ in range(num_learners)])


    #read_events = manager.list([manager.list([manager.Event() for _ in range(num_learners)])
    #                        for _ in range(num_learners)]) #2 dim array is supported
    #read_events = manager.list([
    #    manager.list([False for _ in range(num_learners)])
    #    for _ in range(num_learners)]) #2 dim array is supported
    read_events = manager.list([True for _ in range(num_learners * num_learners)])
    #write_events = manager.list([
    #    manager.list([manager.Event() for _ in range(num_learners)])
    #    for _ in range(num_learners)])
    write_events = manager.list([True for _ in range(num_learners * num_learners)])
    #write_events = manager.list([
    #    manager.list([False for _ in range(num_learners)])
    #    for _ in range(num_learners)])
    msg_buffer = manager.list()
    for i in range(num_learners):
        actor_critic = Policy(
            (4, 84, 84),
            base_kwargs={'recurrent': args.recurrent_policy},
            env_name=args.env_name)
        actor_critic.load_state_dict(torch.load('0.207.1774.8.pt')[0].state_dict())
        actor_critic.to('cpu')
        actor_critic = encode(actor_critic)
        msg_buffer.append(actor_critic)

    #manager.register('get_barrier', callable=lambda: barrier, proxytype=BarrierProxy)
    manager.register('get_sync_list', callable=lambda :sync_list, proxytype=ListProxy)
    #manager.register('get_buffer_locks', callable=lambda : buffer_locks, proxytype=ListProxy)
    manager.register('get_read_events', callable=lambda : read_events, proxytype=ListProxy)
    manager.register('get_write_events', callable= lambda : write_events, proxytype=ListProxy)
    manager.register('get_msg_buffer', callable=lambda :msg_buffer, proxytype=ListProxy)
    manager.__init__(address=(host, port), authkey=key)
    print('start service at', host)
    s = manager.get_server()
    s.serve_forever()

if __name__ == '__main__':
    mp.set_start_method('spawn') #need to set start method into spawn to transmate cuda tensor
    args = get_args()
    manager = mp.Manager()
    server(manager,'127.0.0.1', 5000, b'abc', args)