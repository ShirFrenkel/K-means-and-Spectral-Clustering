######################################################################################################
# Replace the comment in run() function below with your algorithm call.
#
# To run this script on Nova, use the following command:
#   nohup python3.8.5 runtimematrix.py > output.txt &
# It will run in the background (even if the session is closed or the VPN disconnects)
# and save the results into output.txt file.
#
# You need to run this twice (for 2 and 3 dimensions, separately)
######################################################################################################

import os
import sys
sys.path.append(os.getcwd())

from datetime import datetime
from main import testing_main
import numpy as np
import pandas as pd

D = 2
TIMEOUT = 60 * 6    # After this time we use stop running (set runtime to `inf`)
n_list = [500, 520]
k_list = [249]

data = pd.DataFrame(index=k_list, columns=n_list, dtype=np.float64)


def run(k: int, n: int):
    if data.loc[k, n] == float('inf'):
        print(f"data.loc[k, n]: {data.loc[k, n]} - skipping")
        return k, n, float('inf')
    best_time = 60 * 8
    for i in range(3):
        start_time = datetime.now()
        if n > k:
            testing_main(D, n, k)

        end_time = datetime.now()
        running_time = end_time - start_time
        best_time = min(best_time, running_time.total_seconds())
    return k, n, best_time


if __name__ == '__main__':
    print(f"{data.to_string(show_dimensions=True)}\n")

    for n in n_list:
        print("n = {}".format(n))
        for k in k_list:
            if data.loc[k, n] != float('inf'):
                result = run(k, n)
                data.loc[result[0], result[1]] = result[2]
                if data.loc[k, n] > TIMEOUT:
                    data.loc[k, n:].iloc[1:] = float('inf')

            #print(f"{data.to_string(show_dimensions=True)}\n")
            #print("\n", data.to_dict(), "\n")

        print(f"{data.to_string(show_dimensions=True)}\n")
        print(data.to_dict())

    print(f"\n== FINISHED ==\n")
    print(f"{data.to_string(show_dimensions=True)}\n")
    print(data.to_dict())

