Search.setIndex({"docnames": ["api/embeddings/dense/all2all", "api/embeddings/dense/index", "api/embeddings/index", "api/embeddings/saver", "api/embeddings/sparse/distributed", "api/embeddings/sparse/index", "api/embeddings/tf_distributed_embedding", "api/index", "api/init", "api/optimizers/opts", "api/utils/index", "api/utils/opt_scope", "api/utils/opt_utils", "env_vars/env_vars", "examples/amp", "examples/dense_demo", "examples/dlrm", "examples/index", "features/features", "get_started/get_started", "index", "intro_link", "known_issues/issues", "performance/dense_demo", "performance/dlrm", "performance/index", "release_notes/release_notes"], "filenames": ["api/embeddings/dense/all2all.rst", "api/embeddings/dense/index.rst", "api/embeddings/index.rst", "api/embeddings/saver.rst", "api/embeddings/sparse/distributed.rst", "api/embeddings/sparse/index.rst", "api/embeddings/tf_distributed_embedding.rst", "api/index.rst", "api/init.rst", "api/optimizers/opts.rst", "api/utils/index.rst", "api/utils/opt_scope.rst", "api/utils/opt_utils.rst", "env_vars/env_vars.md", "examples/amp.md", "examples/dense_demo.md", "examples/dlrm.md", "examples/index.rst", "features/features.md", "get_started/get_started.md", "index.rst", "intro_link.md", "known_issues/issues.md", "performance/dense_demo.md", "performance/dlrm.md", "performance/index.rst", "release_notes/release_notes.md"], "titles": ["All2All Dense Embedding", "SparseOperationKit Dense Embeddings", "SparseOperationKit Embeddings", "SparseOperationKit Embedding Saver", "Distributed Sparse Embedding", "SparseOperationKit Sparse Embeddings", "TF Distributed Embedding", "SparseOperationKit API", "SparseOperationKit Initialize", "SparseOperationKit Optimizers", "SparseOperationKit Utilities", "SparseOperationKit Optimizer Scope", "SparseOperationKit Optimizer Utils", "Environment Variables", "Mixed Precision", "Demo model using Dense Embedding Layer", "DLRM using SparseOperationKit", "Examples and Tutorials", "Features in SparseOperationKit", "Get Started With SparseOperationKit", "Welcome to SparseOperationKit\u2019s documentation!", "SparseOperationKit", "Known Issues", "Performance of demo model using Dense Embedding Layer", "SOK DLRM Benchmark", "Performance", "SparseOperationKit Release Notes"], "terms": {"class": [0, 3, 4, 6, 9, 11, 12, 19], "sparse_operation_kit": [0, 3, 4, 6, 8, 9, 11, 12, 14, 15, 16, 19, 21, 24], "all2all_dense_embed": 0, "all2alldenseembed": [0, 1, 19, 26], "arg": [0, 4, 6, 8, 9, 12, 19], "kwarg": [0, 4, 6, 8, 9, 12, 19], "sourc": [0, 3, 4, 6, 8, 9, 11, 12, 15, 16], "base": [0, 4, 14, 18, 24], "layer": [0, 3, 4, 6, 12, 19, 20, 21, 22], "abbrevi": [0, 3, 4, 8, 9, 11, 12], "sok": [0, 3, 4, 8, 9, 11, 12, 13, 14, 18, 19, 21, 22, 23, 26], "thi": [0, 3, 4, 6, 8, 9, 11, 12, 13, 14, 18, 19, 24], "i": [0, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23, 24], "wrapper": [0, 4, 6, 14], "It": [0, 4, 6, 11, 13, 14, 21], "can": [0, 4, 6, 8, 9, 12, 13, 14, 15, 16, 18, 19, 21, 22, 24, 26], "us": [0, 3, 4, 6, 8, 9, 11, 12, 13, 14, 18, 21, 22, 24, 26], "creat": [0, 4, 6, 12, 13, 19], "which": [0, 3, 4, 8, 13, 15, 18, 19, 21, 26], "distribut": [0, 2, 5, 7, 8, 21, 26], "kei": [0, 4, 6, 18, 19, 26], "gpu_id": [0, 4, 18], "gpu_num": [0, 4, 18], "each": [0, 3, 4, 6, 8, 15, 18, 19], "gpu": [0, 4, 6, 8, 9, 13, 14, 19, 20, 21, 22, 23, 24], "paramet": [0, 3, 4, 6, 8, 11, 12, 14, 21], "max_vocabulary_size_per_gpu": [0, 4, 15, 19], "integ": [0, 4, 6], "first": [0, 4, 6, 8, 19], "dimens": [0, 4, 6], "variabl": [0, 3, 4, 6, 11, 12, 14, 19, 20], "whose": [0, 4, 6], "shape": [0, 3, 4, 6, 19], "embedding_vec_s": [0, 3, 4, 6, 15, 16, 19, 23], "second": [0, 4, 6, 24], "slot_num": [0, 4, 15, 19, 23], "number": [0, 4, 14, 19], "featur": [0, 4, 15, 19, 20], "file": [0, 3, 4, 6, 15, 16, 19], "process": [0, 3, 4, 8, 12, 14, 15, 16, 18, 19, 21, 23], "same": [0, 4, 13, 14, 18, 19, 22], "time": [0, 4, 24], "iter": [0, 4, 19, 24], "where": [0, 3, 4, 15, 18, 19, 21], "all": [0, 1, 3, 4, 5, 8, 9, 12, 18, 19, 21, 23, 24], "produc": [0, 4, 19], "vector": [0, 4, 6, 12, 18, 19], "nnz_per_slot": [0, 15, 19, 23], "valid": [0, 4, 19], "slot": [0, 4, 15, 18], "The": [0, 3, 4, 5, 6, 8, 9, 11, 14, 15, 18, 19, 21, 22, 23, 24, 26], "dynamic_input": 0, "boolean": [0, 4], "fals": [0, 11, 14, 19], "whether": [0, 4, 13], "input": [0, 4, 6, 11, 12, 18, 19, 21, 26], "dynam": [0, 4, 14, 26], "For": [0, 3, 8, 13, 15, 18, 19], "exampl": [0, 3, 4, 6, 8, 11, 12, 13, 15, 18, 19, 20, 23], "tensor": [0, 3, 4, 6, 19], "com": [0, 16, 21, 24], "from": [0, 3, 12, 14, 15, 18, 24], "tf": [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 23, 26], "uniqu": [0, 9, 26], "when": [0, 3, 4, 6, 8, 9, 11, 14, 15, 18, 21, 22, 26], "true": [0, 4, 12, 19, 23, 24], "lookup": [0, 26], "gather": [0, 18, 26], "pattern": [0, 14, 26], "By": [0, 4, 13, 15, 19], "default": [0, 4, 13, 15, 19], "mean": [0, 4, 5, 13, 18, 21, 26], "size": [0, 18, 24], "must": [0, 3, 4, 6, 8, 19, 22], "replica_batchs": 0, "use_hasht": [0, 4], "hashtabl": [0, 4], "embeddingvari": [0, 3, 4], "insert": [0, 4, 26], "otherwis": [0, 4, 13, 22], "index": [0, 4, 19], "look": [0, 4, 18, 21], "up": [0, 3, 4, 18, 21], "so": [0, 4, 6, 12, 19], "rang": [0, 4, 14, 19], "0": [0, 4, 13, 18, 19, 20, 22, 24], "key_dtyp": [0, 4], "dtype": [0, 4, 6, 19, 26], "int64": [0, 4, 6, 19], "data": [0, 4, 14, 15, 18, 19, 21, 22, 24], "type": [0, 3, 4, 6, 8, 11, 14, 19], "embedding_initi": [0, 4], "string": [0, 3, 4, 6, 8], "an": [0, 4, 14], "instanc": [0, 4, 15, 19, 21], "kera": [0, 4, 9, 11, 12, 14, 19], "initi": [0, 4, 6, 7, 19, 20, 26], "gener": [0, 4, 6, 24], "valu": [0, 3, 4, 6, 14], "random_uniform": [0, 4], "minval": [0, 4], "05": [0, 4], "maxval": [0, 4], "randomuniform": [0, 4], "emb_lay": [0, 4], "function": [0, 3, 4, 6, 8, 11, 12, 13, 14, 21, 26], "def": [0, 4, 6, 11, 12, 14, 19], "_train_step": [0, 4, 6, 11, 12, 19], "label": [0, 4, 6, 11, 12, 19], "emb_vector": [0, 4, 6, 19], "enumer": [0, 4, 6, 12, 19], "dataset": [0, 4, 6, 12, 19], "call": [0, 3, 4, 6, 8, 12, 14, 18, 19], "train": [0, 4, 12, 14, 16, 19, 21, 24, 26], "forward": [0, 4, 6, 18], "logic": [0, 4, 6], "ar": [0, 1, 3, 4, 5, 6, 8, 9, 12, 13, 15, 16, 18, 19, 21, 22], "store": [0, 4, 6], "row": [0, 4], "major": 0, "If": [0, 3, 13, 19, 21], "none": [0, 3, 6, 19, 23, 24], "batchsiz": [0, 4, 19, 23], "return": [0, 3, 4, 6, 8, 11, 12, 14, 19], "its": [0, 6, 8, 9, 13, 15, 18, 19], "equal": [0, 3, 4, 13], "float": [0, 4, 14], "There": [1, 5, 9, 13, 18, 21, 22], "sever": [1, 5, 9, 16, 18, 21, 22], "implement": [1, 5, 9, 15, 18, 21, 26], "them": [1, 5], "equival": [1, 5, 18], "nn": [1, 5, 18, 19], "embedding_lookup": [1, 18], "all2al": [1, 2], "spars": [2, 7, 21], "dens": [2, 4, 7, 12, 19], "saver": [2, 7], "tfdistributedembed": [2, 6], "current": [3, 6, 8, 21], "you": [3, 13, 14, 15, 16, 19, 21, 22], "explicitli": 3, "dump": [3, 19], "load": 3, "trainabl": [3, 11, 12, 14], "those": [3, 12, 22], "And": [3, 8, 14, 15, 16, 18, 19, 21], "other": [3, 8, 12, 13, 14, 19, 21], "still": 3, "manag": [3, 11], "dl": [3, 21, 24, 26], "framework": [3, 21, 26], "dump_to_fil": 3, "embedding_vari": [3, 12], "filepath": 3, "save": [3, 4, 6, 15, 16, 24], "specifi": [3, 4, 6, 8, 13, 15, 19], "host": 3, "multipl": [3, 6, 13, 18, 19, 21, 22, 23], "cpu": [3, 8, 15, 19, 23], "within": [3, 21], "distributedvari": 3, "need": [3, 11, 12, 14, 21], "directori": [3, 24], "statu": [3, 8], "op": [3, 13, 24, 26], "execut": [3, 8, 22], "successfulli": [3, 8, 19], "ok": [3, 8], "restore_from_fil": 3, "restor": 3, "load_embedding_valu": 3, "assign": 3, "": [3, 6, 8, 13, 14, 16, 19, 22], "list": [3, 11, 12, 19], "tupl": [3, 11, 12], "2": [3, 4, 8, 9, 15, 20, 23, 24], "rank": [3, 4, 16], "make": [3, 13, 14, 19, 22, 24], "big": 3, "just": [3, 14], "like": [3, 15], "thei": [3, 14, 15], "stack": 3, "bs_0": 3, "bs_1": 3, "bs_2": 3, "treat": 3, "distributed_embed": 4, "distributedembed": [4, 5, 12], "combin": [4, 5, 18], "how": [4, 13, 14, 15, 16, 19], "intra": [4, 15, 18], "sum": [4, 5, 18, 19], "max_nnz": 4, "maximum": 4, "max_feature_num": 4, "sampl": [4, 15, 19, 23, 24], "memori": [4, 14, 21], "statist": 4, "known": [4, 18, 20], "max": 4, "_featur": 4, "_num": 4, "_nnz": 4, "sparsetensor": 4, "dense_shap": 4, "dim": 4, "denot": 4, "therefor": [4, 14, 15, 19, 21], "indic": [4, 13], "column": 4, "correspond": 4, "Its": 4, "embedding_lookup_spars": [5, 18], "support": [5, 18, 19, 21, 26], "build": [6, 13, 15, 16, 24], "model": [6, 11, 12, 14, 16, 20, 21], "parallel": [6, 15, 19, 20, 21], "total": [6, 24, 26], "tensorflow": [6, 8, 9, 11, 13, 20, 21, 23, 24, 26], "api": [6, 8, 14, 20, 26], "util": [6, 7, 20, 21], "strategi": [6, 8, 11, 12, 21], "do": [6, 8], "commun": [6, 19, 26], "among": [6, 18, 21, 22], "differ": [6, 12, 14, 15, 18, 19, 22, 26], "tf_distributed_embed": 6, "leverag": [6, 8, 19, 21], "vocabulary_s": [6, 15], "numpi": 6, "arrai": 6, "glorotnorm": 6, "comm_opt": 6, "experiment": [6, 14], "communicationopt": 6, "see": [6, 14, 21], "doc": [6, 19, 20], "scope": [6, 7, 8, 10, 12, 19], "embedding_lay": [6, 12, 16, 19], "run": [6, 8, 12, 14, 19], "note": [6, 11, 19, 20, 21, 24], "correctli": 6, "int32": 6, "replica_output": 6, "replica": 6, "float32": [6, 14], "init": [7, 8, 19], "embed": [7, 11, 12, 14, 19, 20, 21, 22, 24, 26], "optim": [7, 10, 14, 15, 19, 20, 26], "adam": [7, 11, 14, 15, 19, 26], "local": [7, 24], "updat": [7, 14], "core": [8, 11, 19], "avail": [8, 13, 14, 19, 21], "pleas": [8, 9, 13, 19, 21], "set": [8, 19], "cuda_visible_devic": [8, 19], "config": [8, 19, 21, 26], "set_visible_devic": [8, 19, 26], "befor": [8, 14, 19, 23, 24], "launch": [8, 14, 19, 22], "runtim": [8, 20], "In": [8, 14, 18, 19, 21, 22], "x": [8, 20, 21, 24], "horovod": [8, 21, 26], "under": [8, 19], "import": [8, 14, 20, 21], "hvd": [8, 14, 19], "1": [8, 12, 13, 15, 16, 18, 20, 22, 23, 24], "15": [8, 20, 24, 26], "onli": [8, 11, 13, 14, 19, 21], "work": [8, 18, 21], "retur": 8, "evalu": [8, 24], "sess": [8, 19], "step": [8, 12, 19, 21, 24], "ani": [8, 13, 18], "sok_init": 8, "global_batch_s": [8, 15, 16, 19, 24], "session": [8, 19], "dictionari": 8, "keyword": 8, "argument": [8, 9, 14, 23], "contain": 8, "content": 8, "insid": 9, "unsorted_segment_sum": 9, "replac": 9, "version": [9, 19, 20], "4": [9, 15, 16, 23, 24, 26], "obtain": 9, "perform": [9, 19, 20], "gain": 9, "while": 9, "5": [9, 19, 23, 24], "should": [9, 14, 18], "ident": [9, 19], "refer": [9, 19], "http": [9, 16, 21, 24], "googl": 9, "cn": 9, "api_doc": 9, "python": [9, 14, 15, 19, 21], "document": [9, 14, 15, 16, 19, 24], "split_embedding_variable_from_oth": [10, 11, 12, 14, 19], "optimizerscop": [10, 11, 19], "context_scop": 11, "trainable_vari": [11, 12, 14, 19], "context": 11, "along": [11, 12], "nativ": [11, 19], "context_manag": 11, "switch": [11, 13], "handl": 11, "emb_opt": [11, 19], "other_opt": 11, "gradienttap": [11, 12, 14, 19], "tape": [11, 12, 14, 19], "logit": [11, 19], "loss": [11, 12, 19], "loss_fn": [11, 12, 19], "emb_var": [11, 12, 19], "other_var": [11, 12, 19], "emb_grad": [11, 19], "other_grad": [11, 19], "gradient": [11, 14, 18, 19], "apply_gradi": [11, 14, 19], "zip": [11, 14, 19], "experimental_aggregate_gradi": [11, 14, 19], "dense_opt": [11, 19], "mai": [11, 21], "next": 11, "releas": [11, 13, 20], "split": 12, "automat": [12, 19], "sinc": 12, "aggreg": 12, "we": [12, 15, 22], "wai": [12, 14, 21], "other_vari": [12, 14], "normal": 12, "__init__": [12, 19], "self": [12, 19], "super": [12, 19], "dense_lay": [12, 19], "unit": [12, 15, 19], "out": 12, "two": [13, 14], "kind": 13, "sparseoperationkit": [13, 14, 15, 22], "respect": [13, 15], "These": [13, 18, 21], "dure": [13, 14, 26], "configur": [13, 26], "comput": 13, "capabl": 13, "target": [13, 21, 24], "devic": [13, 22, 26], "70": [13, 24], "75": 13, "80": [13, 24], "want": [13, 19, 21], "env": [13, 21], "var": 13, "semicolon": 13, "separ": 13, "sm": 13, "export": [13, 16], "60": 13, "65": 13, "enabl": 13, "dedic": [13, 21, 26], "cuda": [13, 23, 26], "stream": [13, 26], "off": 13, "nvtx": [13, 15, 23], "mark": 13, "disabl": 13, "ON": 13, "On": 13, "debug": 13, "mode": 13, "built": [13, 19], "some": [13, 19], "assert": 13, "synchroniz": 13, "cudaeventsynchron": 13, "wait": 13, "complet": 13, "cudastreamwaitev": 13, "depend": [13, 22], "between": [13, 22], "except": [13, 18], "synchron": [13, 19, 21, 26], "applic": 14, "faster": 14, "less": 14, "both": 14, "16": [14, 19, 24], "bit": 14, "32": [14, 16], "point": 14, "follow": [14, 16, 18, 21], "link": 14, "section": [14, 18, 19], "illustr": 14, "To": [14, 18], "your": [14, 19, 21], "add": [14, 15, 22, 23, 26], "line": 14, "script": [14, 23], "keep": 14, "part": 14, "untouch": 14, "polici": 14, "mixed_precis": 14, "mixed_float16": 14, "set_global_polici": 14, "float16": 14, "ha": [14, 15, 21], "narrow": 14, "compar": 14, "might": [14, 19, 22], "lead": 14, "underflow": 14, "overflow": 14, "problem": [14, 22], "techniqu": 14, "avoid": 14, "numer": 14, "provid": [14, 18, 21], "could": [14, 19], "well": 14, "lossscaleoptim": 14, "loop": 14, "calcul": 14, "after": [14, 19], "backward": [14, 18], "propag": [14, 18], "get": [14, 20], "correct": 14, "becaus": [14, 19, 21], "also": [14, 18], "back": 14, "train_step": [14, 19], "y": 14, "predict": 14, "loss_object": 14, "scaled_loss": 14, "get_scaled_loss": 14, "emb_vari": 14, "scaled_emb_gradi": 14, "scaled_other_gradi": 14, "emb_gradi": 14, "get_unscaled_gradi": 14, "other_gradi": 14, "find": [14, 15, 16, 21], "tutori": [14, 15, 16, 20, 24], "mixedprecis": 14, "command": [14, 15, 21], "cd": [14, 21, 24], "amp_tf2": 14, "py": [14, 15, 16, 19, 21, 23, 24], "four": 14, "engin": 14, "base_layer_util": 14, "enable_v2_dtype_behavior": 14, "set_polici": 14, "similarili": 14, "here": [14, 21, 24], "unscal": 14, "scaled_gradi": 14, "distinguish": 14, "variabel": 14, "len": [14, 19], "allreduc": [14, 19], "grad": [14, 19], "emb_train_op": [14, 19], "other_train_op": [14, 19], "control_depend": [14, 19, 22], "identifi": 14, "mpiexec": [14, 15, 16, 19], "allow": [14, 15, 16], "root": [14, 15, 16], "np": [14, 15, 16, 19, 24], "amp_tf1": 14, "demonstr": [15, 16], "dnn": [15, 19, 21], "reduct": [15, 18, 19], "conduct": [15, 18, 19], "code": [15, 16, 21], "densedemo": [15, 17, 20, 23, 25], "modul": 15, "cupi": 15, "mpi4pi": 15, "construct": 15, "7": [15, 19], "fulli": [15, 19, 21], "connect": 15, "former": 15, "6": [15, 24], "have": [15, 21], "1024": [15, 16, 19], "output": [15, 19], "last": 15, "one": [15, 18, 19, 21], "randomli": 15, "filenam": 15, "ad": [15, 26], "xxx": 15, "python3": [15, 16, 19, 21, 23, 24], "gen_data": 15, "65536": [15, 19, 23, 24], "100": [15, 23], "10": [15, 19, 23, 24], "iter_num": 15, "30": [15, 24], "d": 15, "let": 15, "own": [15, 19], "dataread": 15, "read": 15, "splite": [15, 24], "name": 15, "save_prefix": [15, 16], "split_id": 15, "data_0": 15, "data_1": 15, "linearli": 15, "arrang": 15, "s0": 15, "s1": 15, "s2": 15, "s3": 15, "s4": 15, "s5": 15, "s6": 15, "s7": 15, "split_data": 15, "split_num": 15, "8": [15, 18, 19, 23, 24], "data_": 15, "method": 15, "n": 15, "run_tf": 15, "data_filenam": 15, "8192": [15, 23], "num_dense_lay": 15, "data_split": 15, "run_sok_mirroredstrategi": 15, "oversubscrib": 15, "enough": 15, "run_sok_multiworker_mpi": 15, "horovodrun": [15, 19, 24], "h": [15, 19], "localhost": [15, 19], "run_sok_horovod": 15, "data_filename_prefix": 15, "criteo": [16, 24], "terabyt": [16, 24], "download": [16, 21], "option": 16, "instruct": 16, "csv": 16, "hugectr": [16, 24, 26], "Then": 16, "convert": 16, "binari": [16, 24], "bin2csv": 16, "input_fil": 16, "yourbinaryfilepath": 16, "bin": [16, 24], "num_output_fil": 16, "output_path": 16, "train_": 16, "test": [16, 24], "64": [16, 24], "test_": 16, "embedding_dim": 16, "main": [16, 19, 24], "16384": 16, "train_file_pattern": 16, "test_file_pattern": 16, "bottom_stack": 16, "512": 16, "256": 16, "top_stack": 16, "distribute_strategi": 16, "multiwork": 16, "tf_mp": 16, "arxiv": 16, "org": 16, "pdf": 16, "1906": 16, "00091": 16, "lab": 16, "2013": 16, "12": [16, 24], "click": [16, 21], "log": 16, "github": [16, 21, 24], "tree": 16, "master": [16, 24], "offici": 16, "recommend": [16, 19, 21], "dlrm": [17, 20, 25, 26], "mix": [17, 20, 21, 26], "precis": [17, 20, 26], "detail": 18, "inform": 18, "about": [18, 21], "As": 18, "describ": [18, 19], "introduct": [18, 20], "manner": [18, 21], "doe": [18, 21], "requir": 18, "further": [18, 21], "transform": [18, 21], "paral": 18, "algorithm": [18, 21, 26], "singl": [18, 19, 21, 23, 26], "machin": [18, 19, 21], "mp": [18, 21], "1000": 18, "1001": 18, "pictur": [18, 21], "depict": [18, 19, 21], "reduc": [18, 19], "overhead": 18, "tabl": [18, 24], "huge": [18, 21], "tini": 18, "field": 18, "unifi": 18, "encod": 18, "collect": [18, 23], "oper": [18, 21], "scatter": 18, "exchang": 18, "anoth": 18, "top": 18, "walk": 19, "through": [19, 21], "simpl": 19, "demo": 19, "familiar": 19, "expert": 19, "more": [19, 21], "assum": 19, "relat": 19, "tool": [19, 26], "system": [19, 21], "now": 19, "detect": 19, "signatur": 19, "structur": 19, "fig": 19, "demomodel": 19, "embedding_vector_s": 19, "num_of_dense_lay": 19, "per": 19, "concaten": 19, "_": 19, "activ": 19, "relu": 19, "append": 19, "out_lay": 19, "reshap": 19, "hidden": 19, "create_demomodel": 19, "placehold": 19, "input_tensor": 19, "type_spec": 19, "tensorspec": 19, "target_shap": 19, "compat": [19, 21], "specificli": 19, "exist": [19, 22], "program": [19, 22], "multi": [19, 26], "thread": 19, "But": [19, 21], "due": [19, 21], "gil": 19, "cpython": 19, "interpret": 19, "hard": [19, 21], "impact": 19, "end": [19, 24], "infer": [19, 21], "mani": 19, "use_tf_opt": 19, "learning_r": 19, "els": 19, "initil": 19, "binarycrossentropi": 19, "from_logit": 19, "_replica_loss": 19, "compute_average_loss": 19, "labl": 19, "replica_loss": 19, "total_loss": 19, "reduceop": 19, "axi": 19, "print": 19, "info": 19, "format": 19, "control": 19, "o": [19, 21, 23], "json": 19, "worker_num": 19, "task_id": 19, "environ": [19, 20], "str": 19, "procecss": 19, "port": 19, "12345": 19, "arbitrari": 19, "unus": 19, "tf_config": 19, "cluster": [19, 21], "worker": 19, "task": 19, "check": [19, 21], "mpi": [19, 23], "similar": 19, "local_rank": 19, "first_batch": 19, "broadcast_vari": 19, "root_rank": 19, "restrict": 19, "sok_init_op": 19, "colocate_gradients_with_op": 19, "init_op": 19, "group": 19, "global_variables_initi": 19, "local_variables_initi": 19, "loss_v": 19, "even": 19, "instal": [20, 24, 26], "start": 20, "what": 20, "new": 20, "compil": 20, "issu": [20, 26], "nccl": 20, "conflict": 20, "packag": 21, "wrap": 21, "acceler": 21, "case": 21, "design": 21, "common": [21, 26], "deeplearn": 21, "most": 21, "extract": 21, "across": 21, "node": [21, 26], "estim": 21, "rate": 21, "ctr": 21, "veri": 21, "effici": 21, "solut": [21, 22], "scenario": 21, "amount": 21, "fit": 21, "whole": 21, "avaiabl": 21, "matter": 21, "locat": 21, "dp": 21, "minim": 21, "chang": 21, "With": 21, "consum": 21, "littl": 21, "resourc": 21, "integr": [21, 26], "hous": 21, "feed": 21, "scale": 21, "workflow": 21, "docker": [21, 24], "imag": 21, "nvcr": [21, 24], "io": [21, 24], "nvidia": [21, 24], "merlin": [21, 24], "22": 21, "02": [21, 24], "sparseopeationkit": 21, "alreadi": 21, "directrli": 21, "via": [21, 26], "sparse_opeation_kit": 21, "pip": [21, 26], "user": 21, "been": 21, "upload": 21, "take": 21, "yourself": 21, "setuptool": 21, "sy": 21, "subprocess": 21, "shutil": 21, "git": [21, 24], "clone": [21, 24], "setup": 21, "sdist": 21, "copi": 21, "cp": [21, 24], "dist": 21, "tar": 21, "gz": 21, "yourtargetpath": 21, "souc": 21, "our": 21, "try": 22, "fix": [22, 26], "shose": 22, "futur": 22, "transfer": 22, "order": 22, "determinist": 22, "hang": [22, 26], "forc": 22, "introduc": 23, "trace": [23, 24], "fork": [23, 24], "exec": [23, 24], "timelin": 23, "nsy": [23, 24], "backtrac": [23, 24], "cudabacktrac": [23, 24], "cpuctxsw": [23, 24], "f": 23, "profiling_filenam": 23, "dgx": 23, "a100": [23, 24], "nsightsystem": 23, "linux": 23, "cli": 23, "public": 23, "2021": 23, "58": 23, "origin": 23, "179": 23, "85": 23, "25": [23, 24], "90": 23, "45": [23, 24], "36": 23, "45548": 23, "316269": 23, "1444925": 23, "gitlab": 24, "dlrm_benchmark": 24, "train_data": 24, "test_data": 24, "split_bin": 24, "privileg": 24, "rm": 24, "21": 24, "11": 24, "mkdir": 24, "dsm": 24, "v100": 24, "cmake": 24, "j": 24, "r": 24, "usr": 24, "lib": 24, "fp32": 24, "result": [24, 26], "hvd_wrapper": 24, "sh": 24, "data_dir": 24, "xla": 24, "compress": 24, "custom_interact": 24, "eval_in_last": 24, "amp": 24, "82gb": 24, "batch": 24, "exit": 24, "criteria": 24, "frequent": 24, "custom": [24, 26], "interact": 24, "minut": 24, "averag": 24, "m": 24, "throughput": 24, "epoch": 24, "ye": 24, "93": 24, "09": 24, "55": 24, "08m": 24, "06": 24, "07": 24, "13": 24, "74": 24, "14": 24, "51m": 24, "55296": 24, "auc": 24, "8025": 24, "everi": 24, "3793": 24, "23": 24, "44": 24, "67": 24, "87": 24, "66m": 24, "99": 24, "26": 24, "50m": 24, "17": 24, "52": 24, "19": 24, "71": 24, "42": 24, "02m": 24, "20": 24, "35": 24, "9": 24, "56": 24, "99m": 24, "3": 24, "59": 24, "04": 24, "85m": 24, "69": 24, "54": 24, "62m": 24, "seen": 24, "care": 24, "repo": 24, "deeplearningexampl": 24, "instead": 24, "wravean": 24, "dot": 24, "written": 24, "readm": 24, "early_stop": 24, "auto": 26, "uint32": 26, "benchmark": 26, "visibl": 26, "mirroredstrategi": 26, "greater": 26, "than": 26, "identityhasht": 26, "hash": 26, "map": 26, "easili": 26, "distributedsparseembed": 26}, "objects": {"sparse_operation_kit.core.context_scope": [[11, 0, 1, "", "OptimizerScope"]], "sparse_operation_kit.core": [[8, 1, 0, "-", "initialize"]], "sparse_operation_kit.core.initialize": [[8, 2, 1, "", "Init"]], "sparse_operation_kit.embeddings.all2all_dense_embedding": [[0, 0, 1, "", "All2AllDenseEmbedding"]], "sparse_operation_kit.embeddings.all2all_dense_embedding.All2AllDenseEmbedding": [[0, 3, 1, "", "call"]], "sparse_operation_kit.embeddings.distributed_embedding": [[4, 0, 1, "", "DistributedEmbedding"]], "sparse_operation_kit.embeddings.distributed_embedding.DistributedEmbedding": [[4, 3, 1, "", "call"]], "sparse_operation_kit.embeddings.tf_distributed_embedding": [[6, 0, 1, "", "TFDistributedEmbedding"]], "sparse_operation_kit.embeddings.tf_distributed_embedding.TFDistributedEmbedding": [[6, 3, 1, "", "call"]], "sparse_operation_kit.optimizers": [[12, 1, 0, "-", "utils"]], "sparse_operation_kit.optimizers.utils": [[12, 2, 1, "", "split_embedding_variable_from_others"]], "sparse_operation_kit.saver.Saver": [[3, 0, 1, "", "Saver"]], "sparse_operation_kit.saver.Saver.Saver": [[3, 3, 1, "", "dump_to_file"], [3, 3, 1, "", "load_embedding_values"], [3, 3, 1, "", "restore_from_file"]], "sparse_operation_kit.tf.keras.optimizers.adam": [[9, 0, 1, "", "Adam"]]}, "objtypes": {"0": "py:class", "1": "py:module", "2": "py:function", "3": "py:method"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "module", "Python module"], "2": ["py", "function", "Python function"], "3": ["py", "method", "Python method"]}, "titleterms": {"all2al": [0, 18], "dens": [0, 1, 15, 18, 23], "embed": [0, 1, 2, 3, 4, 5, 6, 15, 18, 23], "sparseoperationkit": [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 16, 18, 19, 20, 21, 26], "saver": 3, "distribut": [4, 6, 18, 19], "spars": [4, 5, 18], "tf": [6, 19], "api": [7, 19], "initi": 8, "optim": [9, 11, 12], "adam": 9, "local": 9, "updat": 9, "util": [10, 12], "scope": 11, "environ": [13, 24], "variabl": 13, "compil": 13, "sok_compile_gpu_sm": 13, "sok_compile_async": 13, "valu": 13, "accept": 13, "sok_compile_use_nvtx": 13, "sok_compile_build_typ": 13, "runtim": 13, "sok_event_sync": 13, "mix": 14, "precis": 14, "tensorflow": [14, 15, 16, 19], "2": [14, 19, 26], "x": [14, 19], "enabl": 14, "loss": 14, "scale": 14, "exampl": [14, 17], "code": 14, "1": [14, 19, 26], "15": [14, 19], "demo": [15, 23], "model": [15, 18, 19, 23], "us": [15, 16, 19, 23], "layer": [15, 18, 23], "requir": 15, "structur": 15, "step": [15, 16], "gener": [15, 16], "dataset": [15, 16, 24], "split": 15, "whole": 15, "multipl": 15, "shard": 15, "run": [15, 16, 24], "thi": [15, 21], "writen": 15, "sok": [15, 16, 24], "mirroredstrategi": [15, 19], "multiworkermirroredstrategi": [15, 19], "mpi": 15, "horovod": [15, 19], "dlrm": [16, 24], "option1": 16, "option2": 16, "set": 16, "common": 16, "param": 16, "refer": 16, "tutori": 17, "featur": [18, 21], "parallel": 18, "gpu": 18, "get": 19, "start": 19, "With": 19, "see": 19, "also": 19, "import": 19, "instal": [19, 21], "defin": 19, "via": 19, "subclass": 19, "function": 19, "strategi": 19, "caution": 19, "tip": 19, "welcom": 20, "": [20, 26], "document": [20, 21], "modul": 21, "along": 21, "hugectr": 21, "from": 21, "pypi": 21, "build": 21, "sourc": 21, "known": 22, "issu": 22, "nccl": 22, "conflict": 22, "perform": [23, 24, 25], "profil": [23, 24], "command": 23, "infrastructur": 23, "number": 23, "end2end": 23, "elaps": 23, "time": 23, "milisecond": 23, "queri": 23, "per": 23, "second": 23, "benchmark": 24, "prepar": 24, "releas": 26, "note": 26, "what": 26, "new": 26, "version": 26, "0": 26}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"All2All Dense Embedding": [[0, "all2all-dense-embedding"], [18, "all2all-dense-embedding"]], "SparseOperationKit Dense Embeddings": [[1, "sparseoperationkit-dense-embeddings"]], "SparseOperationKit Embeddings": [[2, "sparseoperationkit-embeddings"]], "SparseOperationKit Embedding Saver": [[3, "sparseoperationkit-embedding-saver"]], "Distributed Sparse Embedding": [[4, "distributed-sparse-embedding"], [18, "distributed-sparse-embedding"]], "SparseOperationKit Sparse Embeddings": [[5, "sparseoperationkit-sparse-embeddings"]], "TF Distributed Embedding": [[6, "tf-distributed-embedding"]], "SparseOperationKit API": [[7, "sparseoperationkit-api"]], "SparseOperationKit Initialize": [[8, "module-sparse_operation_kit.core.initialize"]], "SparseOperationKit Optimizers": [[9, "sparseoperationkit-optimizers"]], "Adam optimizer": [[9, "adam-optimizer"]], "Local update Adam optimizer": [[9, "local-update-adam-optimizer"]], "SparseOperationKit Utilities": [[10, "sparseoperationkit-utilities"]], "SparseOperationKit Optimizer Scope": [[11, "sparseoperationkit-optimizer-scope"]], "SparseOperationKit Optimizer Utils": [[12, "module-sparse_operation_kit.optimizers.utils"]], "Environment Variables": [[13, "environment-variables"]], "Compile Variables": [[13, "compile-variables"]], "SOK_COMPILE_GPU_SM": [[13, "sok-compile-gpu-sm"]], "SOK_COMPILE_ASYNC": [[13, "sok-compile-async"]], "values accepted": [[13, "values-accepted"], [13, "id1"], [13, "id2"], [13, "id3"]], "SOK_COMPILE_USE_NVTX": [[13, "sok-compile-use-nvtx"]], "SOK_COMPILE_BUILD_TYPE": [[13, "sok-compile-build-type"]], "Runtime Variables": [[13, "runtime-variables"]], "SOK_EVENT_SYNC": [[13, "sok-event-sync"]], "Mixed Precision": [[14, "mixed-precision"]], "TensorFlow 2.x": [[14, "tensorflow-2-x"], [19, "tensorflow-2-x"]], "enable mixed precision": [[14, "enable-mixed-precision"], [14, "id1"]], "loss scaling": [[14, "loss-scaling"], [14, "id2"]], "example codes": [[14, "example-codes"], [14, "id3"]], "TensorFlow 1.15": [[14, "tensorflow-1-15"], [19, "tensorflow-1-15"]], "Demo model using Dense Embedding Layer": [[15, "demo-model-using-dense-embedding-layer"]], "requirements": [[15, "requirements"]], "model structure": [[15, "model-structure"]], "steps": [[15, "steps"], [16, "steps"]], "Generate datasets": [[15, "generate-datasets"], [16, "generate-datasets"]], "Split the whole dataset into multiple shards": [[15, "split-the-whole-dataset-into-multiple-shards"]], "Run this demo writen with TensorFlow": [[15, "run-this-demo-writen-with-tensorflow"]], "Run this demo writen with SOK + MirroredStrategy": [[15, "run-this-demo-writen-with-sok-mirroredstrategy"]], "Run this demo writen with SOK + MultiWorkerMirroredStrategy + MPI": [[15, "run-this-demo-writen-with-sok-multiworkermirroredstrategy-mpi"]], "Run this demo writen with SOK + Horovod": [[15, "run-this-demo-writen-with-sok-horovod"]], "DLRM using SparseOperationKit": [[16, "dlrm-using-sparseoperationkit"]], "[Option1]": [[16, "option1"]], "[Option2]": [[16, "option2"]], "Set common params": [[16, "set-common-params"]], "Run DLRM with TensorFlow": [[16, "run-dlrm-with-tensorflow"]], "Run DLRM with SOK": [[16, "run-dlrm-with-sok"]], "reference": [[16, "reference"]], "Examples and Tutorials": [[17, "examples-and-tutorials"]], "Features in SparseOperationKit": [[18, "features-in-sparseoperationkit"]], "Model-Parallelism GPU Embedding Layer": [[18, "model-parallelism-gpu-embedding-layer"]], "Sparse Embedding Layer": [[18, "sparse-embedding-layer"]], "Dense Embedding Layer": [[18, "dense-embedding-layer"]], "Get Started With SparseOperationKit": [[19, "get-started-with-sparseoperationkit"]], "See also": [[19, null]], "Important": [[19, null], [19, null]], "Install SparseOperationKit": [[19, "install-sparseoperationkit"]], "Import SparseOperationKit": [[19, "import-sparseoperationkit"]], "Define a model with TensorFlow": [[19, "define-a-model-with-tensorflow"]], "Via Subclassing": [[19, "via-subclassing"]], "Via Functional API": [[19, "via-functional-api"]], "Use SparseOperationKit with tf.distribute.Strategy": [[19, "use-sparseoperationkit-with-tf-distribute-strategy"]], "with tf.distribute.MirroredStrategy": [[19, "with-tf-distribute-mirroredstrategy"]], "Caution": [[19, null], [19, null]], "Tip": [[19, null]], "With tf.distribute.MultiWorkerMirroredStrategy": [[19, "with-tf-distribute-multiworkermirroredstrategy"]], "Use SparseOperationKit with Horovod": [[19, "use-sparseoperationkit-with-horovod"]], "Using SparseOperationKit with Horovod": [[19, "using-sparseoperationkit-with-horovod"]], "Welcome to SparseOperationKit\u2019s documentation!": [[20, "welcome-to-sparseoperationkit-s-documentation"]], "SparseOperationKit": [[21, "sparseoperationkit"]], "Features": [[21, "features"]], "Installation": [[21, "installation"]], "Install this module along with HugeCTR": [[21, "install-this-module-along-with-hugectr"]], "Install this module from pypi": [[21, "install-this-module-from-pypi"]], "Build from source": [[21, "build-from-source"]], "Documents": [[21, "documents"]], "Known Issues": [[22, "known-issues"]], "NCCL conflicts": [[22, "nccl-conflicts"]], "Performance of demo model using Dense Embedding Layer": [[23, "performance-of-demo-model-using-dense-embedding-layer"]], "Profiling commands": [[23, "profiling-commands"]], "Infrastructure": [[23, "infrastructure"]], "Performance Numbers": [[23, "performance-numbers"]], "end2end elapsed time (miliseconds)": [[23, "end2end-elapsed-time-miliseconds"]], "Query per seconds": [[23, "query-per-seconds"]], "SOK DLRM Benchmark": [[24, "sok-dlrm-benchmark"]], "Prepare Dataset": [[24, "prepare-dataset"]], "Environment": [[24, "environment"]], "Run Benchmark": [[24, "run-benchmark"]], "Performance": [[24, "performance"], [25, "performance"]], "Profile": [[24, "profile"]], "SparseOperationKit Release Notes": [[26, "sparseoperationkit-release-notes"]], "What\u2019s new in Version 1.1.2": [[26, "whats-new-in-version-1-1-2"]], "What\u2019s new in Version 1.1.1": [[26, "whats-new-in-version-1-1-1"]], "What\u2019s new in Version 1.1.0": [[26, "whats-new-in-version-1-1-0"]], "What\u2019s new in Version 1.0.1": [[26, "whats-new-in-version-1-0-1"]], "What\u2019s new in Version 1.0.0": [[26, "whats-new-in-version-1-0-0"]]}, "indexentries": {"all2alldenseembedding (class in sparse_operation_kit.embeddings.all2all_dense_embedding)": [[0, "sparse_operation_kit.embeddings.all2all_dense_embedding.All2AllDenseEmbedding"]], "call() (sparse_operation_kit.embeddings.all2all_dense_embedding.all2alldenseembedding method)": [[0, "sparse_operation_kit.embeddings.all2all_dense_embedding.All2AllDenseEmbedding.call"]], "saver (class in sparse_operation_kit.saver.saver)": [[3, "sparse_operation_kit.saver.Saver.Saver"]], "dump_to_file() (sparse_operation_kit.saver.saver.saver method)": [[3, "sparse_operation_kit.saver.Saver.Saver.dump_to_file"]], "load_embedding_values() (sparse_operation_kit.saver.saver.saver method)": [[3, "sparse_operation_kit.saver.Saver.Saver.load_embedding_values"]], "restore_from_file() (sparse_operation_kit.saver.saver.saver method)": [[3, "sparse_operation_kit.saver.Saver.Saver.restore_from_file"]], "distributedembedding (class in sparse_operation_kit.embeddings.distributed_embedding)": [[4, "sparse_operation_kit.embeddings.distributed_embedding.DistributedEmbedding"]], "call() (sparse_operation_kit.embeddings.distributed_embedding.distributedembedding method)": [[4, "sparse_operation_kit.embeddings.distributed_embedding.DistributedEmbedding.call"]], "tfdistributedembedding (class in sparse_operation_kit.embeddings.tf_distributed_embedding)": [[6, "sparse_operation_kit.embeddings.tf_distributed_embedding.TFDistributedEmbedding"]], "call() (sparse_operation_kit.embeddings.tf_distributed_embedding.tfdistributedembedding method)": [[6, "sparse_operation_kit.embeddings.tf_distributed_embedding.TFDistributedEmbedding.call"]], "init() (in module sparse_operation_kit.core.initialize)": [[8, "sparse_operation_kit.core.initialize.Init"]], "module": [[8, "module-sparse_operation_kit.core.initialize"], [12, "module-sparse_operation_kit.optimizers.utils"]], "sparse_operation_kit.core.initialize": [[8, "module-sparse_operation_kit.core.initialize"]], "adam (class in sparse_operation_kit.tf.keras.optimizers.adam)": [[9, "sparse_operation_kit.tf.keras.optimizers.adam.Adam"]], "optimizerscope (class in sparse_operation_kit.core.context_scope)": [[11, "sparse_operation_kit.core.context_scope.OptimizerScope"]], "sparse_operation_kit.optimizers.utils": [[12, "module-sparse_operation_kit.optimizers.utils"]], "split_embedding_variable_from_others() (in module sparse_operation_kit.optimizers.utils)": [[12, "sparse_operation_kit.optimizers.utils.split_embedding_variable_from_others"]]}})