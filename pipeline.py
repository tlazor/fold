from pathlib import Path
from joblib import Memory
from sklearn.pipeline import Pipeline
import torch

from TSVToDataFrame import TSVToDataFrame
from LikelihoodEstimator import LikelihoodEstimator
import fold_globals

if __name__ == "__main__":
    cachedir = Path(".cache/joblib")
    memory = Memory(cachedir, verbose=0)

    torch.set_float32_matmul_precision('high')

    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device('mps')
    else:
        fold_globals.DEVICE = torch.device('cpu')
    print('Using device:', fold_globals.DEVICE)

    pipeline = Pipeline([
        ("load_tsv", TSVToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
        ("est_likelihood", LikelihoodEstimator()),
        # ... more pipeline steps (vectorizers, classifiers, etc.) ...
    ])

    # Running the pipeline
    output = pipeline.fit_transform(None)  
    # In this case, we pass None because TSVToDataFrame ignores X and reads from file_path
    print(output.shape)