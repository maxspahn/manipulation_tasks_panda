"""
Playback of trajectories and storing them into a databaseself.
"""
from Learning_from_demonstration import LfD
import sys
import os

if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: " + os.path.basename(__file__) + " <name_of_traj>")
        sys.exit(1)

    lfd = LfD()
    lfd.load(arg1)
    
    lfd.execute()

    save = None
    while not (save in [0,1]):
        print("SAVE CURRENT RECORDING? 0 = NO, 1 = YES")
        try:
            save = int(input('\n'))
        except:
            print("INVALID INPUT")
    if save:
        lfd.save(arg1)