"""
Recording trajectories and storing them into a databaseself.
"""
from Learning_from_demonstration import Learning_from_Demonstration
import sys
import os

if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: " + os.path.basename(__file__) + " <trajectory_file_name>")
        sys.exit(1)

    lfd = Learning_from_Demonstration()
    lfd.traj_rec()
    lfd.save(arg1)