import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_sim_results(csv_file):
    df = pd.read_csv(csv_file)
    df["sys_id"] = df["sys_id"].astype(int)
    df["execution_cycles"] = df["execution_cycles"].astype(float)
    df["exposed_communication_cycles"] = df["exposed_communication_cycles"].astype(float)

    df = df.sort_values(by='sys_id', ascending=True)
    chunk_size = 100
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    plots = []
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        
        fig, ax = plt.subplots(figsize=(20, 6))
        y_pos = np.arange(len(chunk))

        # Plot total execution_cycles as bars
        ax.bar(y_pos, chunk['execution_cycles'], color='blue', label='Exec Cycles')

        # Overlay exposed_communication_cycles portion
        ax.bar(y_pos, chunk['exposed_communication_cycles'], color='red', label='Comm Cycles')

        # Labels and formatting
        ax.set_xticks(y_pos)
        ax.set_xticklabels(chunk['sys_id'], rotation=45)
        ax.set_ylabel('Cycles')
        ax.set_title('Execution Cycles with Communication Breakdown')
        ax.legend()

        plots.append(fig)
        plt.close(fig)
    
    return plots
        
