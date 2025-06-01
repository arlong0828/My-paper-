from nsga_vrp.CW_LS_NSGA2_vrp import *
import argparse

def main():
    days = 31
    start = 0
    end = 300
    m = 0
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_customer_number', type=int, default=0, required=False,
                        help="Number of generations to run")
    parser.add_argument('--end_customer_number', type=int, default=300, required=False,
                        help="Number of generations to run")

    args = parser.parse_args()

    # Initializing instance
    
    # Setting internal variables
    for i in range(days):
        print(start , end)
        nsgaObj = nsgaAlgo()
        # print(args.start_customer_number , args.end_customer_number , m)
        nsgaObj.start_customer_number = start 
        nsgaObj.end_customer_number = end
        nsgaObj.day = m
        # Running Algorithm
        nsgaObj.runMain()
        start += 300+1
        end += 300+1
        m += 1

if __name__ == '__main__':
    main()
