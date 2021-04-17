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


TIMEOUT = 60 * 7    # After this time we use stop running (set runtime to `inf`)
n_list = [50, 100, 150, 180, 200, 210, 220, 240, 245, 250, 255, 260, 265, 270, 280, 300, 310, 400, 500, 530, 550, 600]
k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 20, 32, 64, 128]

data = pd.DataFrame(index=k_list, columns=n_list, dtype=np.float64)


def run(k: int, n: int):
    if data.loc[k, n] == float('inf'):
        print(f"data.loc[k, n]: {data.loc[k, n]} - skipping")
        return k, n, float('inf')

    start_time = datetime.now()

    testing_main(3, n, k)

    end_time = datetime.now()
    running_time = end_time - start_time
    return k, n, running_time.total_seconds()


if __name__ == '__main__':
    print(f"{data.to_string(show_dimensions=True)}\n")

    for n in n_list:
        for k in k_list:
            if data.loc[k, n] != float('inf'):
                result = run(k, n)
                data.loc[result[0], result[1]] = result[2]
                if data.loc[k, n] > TIMEOUT:
                    data.loc[k, n:].iloc[1:] = float('inf')

            print(f"{data.to_string(show_dimensions=True)}\n")
            print("\n", data.to_dict(), "\n")

        print(f"{data.to_string(show_dimensions=True)}\n")

    print(f"\n== FINISHED ==\n")
    print(data.to_dict())

