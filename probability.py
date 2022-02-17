from power import main_power
from MCP_script import main_mcp
import numpy as np

def calc_p90(yields):

    p90 = np.percentile(yields, 10)

    return p90

def calc_p50(yields):

    p50 = np.percentile(yields, 50)

    return p50

if __name__ == '__main__':

    predictions_mcp = main_mcp()
    yields_mcp = main_power(predictions_mcp)

    p90_mcp = calc_p90(yields_mcp)
    print(p90_mcp/10**9)

    p50_mcp = calc_p50(yields_mcp)
    print(p50_mcp/10**9)