"""
Recording trajectories and storing them into a databaseself.
"""
from Learning_from_demonstration import LfD
import sys

if __name__ == '__main__':
    lfd = LfD()
    lfd.traj_rec()
    lfd.save(sys.argv[1])