"""
Playback of trajectories and storing them into a databaseself.
"""
from Learning_from_demonstration import LfD
import sys

if __name__ == '__main__':
    lfd = LfD()
    lfd.load(sys.argv[1])
    
    lfd.execute()

    save = None
    while not (save in [0,1]):
        print("SAVE CURRENT RECORDING? 0 = NO, 1 = YES")
        try:
            save = int(input('\n'))
        except:
            print("INVALID INPUT")
    if save:
        lfd.save(sys.argv[1])