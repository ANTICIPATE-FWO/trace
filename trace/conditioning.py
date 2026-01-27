from json import load
import os
os.chdir('../')

from trace.utils import TrajectoryManager

def main():
    #filepath = "data/38_dst_ipro.json"
    filepath = "data/2_mc_ipro.json"
    manager = TrajectoryManager()
    manager.load_file(filepath)


if __name__ == "__main__":
    main()