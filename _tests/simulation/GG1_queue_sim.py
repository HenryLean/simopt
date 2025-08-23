import numpy as np
import matplotlib.pyplot as plt
from simenvs.queue_sys import Arrival, Server, AdmissionCtrl, ServiceCtrl

queue_node_option = {
    "admission": AdmissionCtrl,
    "service": ServiceCtrl,
}



def sim_queue(arrival_rate, service_rate, max_T=10, n=100, mode="service", *, arrival_kwargs={}, server_kwargs={}):
    arrival = Arrival(arrival_rate, **arrival_kwargs)
    server = Server(service_rate, **server_kwargs)
    queue_node = queue_node_option[mode](arrival, server, max_T=max_T)


    for k in range(n):
        print(f'{k+1}-th arrival:')
        print(f"next_in={queue_node.next_in:.4f}, next_out={queue_node.next_out:.4f}")
        t, state, data = queue_node()
        print(f"time={t:.4f}, state={state:d}")
        for key, value in data.items():
            print(f"* {key} \t= {value:.4f}")
        print(f"next_in_new={queue_node.next_in:.4f}, next_out_new={queue_node.next_out:.4f}\n")
    return np.array(queue_node.state_trace)



def plot_queue_states(trace, *, figsize=(10, 7.5), dpi=100, save_path:str=""):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.step(trace[:,0], trace[:,1], where="post")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    arrival_rate = 1.0
    service_rate = 1.5
    arrival_kwargs = {
        "Type": "M",
    }
    server_kwargs = {
        "Type": "gamma",
        "scale": 0.5,
        "alph": 3.0,
    }

    trace = sim_queue(arrival_rate, service_rate, max_T=10, n=50, mode="service", arrival_kwargs=arrival_kwargs, server_kwargs=server_kwargs)
    plot_queue_states(trace)