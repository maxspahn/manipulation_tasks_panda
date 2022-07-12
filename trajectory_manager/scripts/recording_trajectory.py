"""
Recording trajectories and storing them into a databaseself.
"""
from Learning_from_demonstration import LfD
import sys
import os

if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: " + os.path.basename(__file__) + " <trajectory_file_name>")
        sys.exit(1)

    lfd = LfD()
    lfd.traj_rec()
    lfd.save(arg1)