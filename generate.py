import numpy as np
import pandas as pd

if __name__ == "__main__":
    GEN = 10000
    weights = np.random.normal(30.59, 4, GEN)

    df = pd.DataFrame(weights, columns=["Peso"])
    df.to_csv("./athl_rnd.csv", index_label="INDEXED")